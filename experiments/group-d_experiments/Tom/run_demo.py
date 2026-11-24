"""Run probly MC-Dropout and shapiq interaction analyses on example images."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tom import ProblyImageExplainer, ProblyModelWrapper
from Tom.models import ViTShapleyBackbone
from Tom.visualization import plot_vision_summary, plot_model_heatmap_comparison


DEFAULT_IMAGES=Path("/home/tom/probly/experiments/group-d_experiments/Tom/Pictures")
IMAGE_NAMES=("GoldenRetriever.jpeg", "DogCat.jpeg", "blurryFrog.jpeg")
SAMPLES=15
DROPOUT_P=.3
BUDGET=512
SEED=42
# Four super-patches make the 2^N coalition space fully enumerable (16 masks).
# The configured 512 budget is therefore an upper bound; shapiq can compute exactly.
GRID=(2,2)


def load_images(folder):
    from PIL import Image
    images=[]
    mean=np.asarray([.485,.456,.406],dtype=np.float32)
    std=np.asarray([.229,.224,.225],dtype=np.float32)
    for name in IMAGE_NAMES:
        path=Path(folder)/name
        if not path.exists(): raise FileNotFoundError(f"Required demo image is missing: {path}")
        image=Image.open(path).convert("RGB").resize((224,224),Image.Resampling.BICUBIC)
        array=np.asarray(image,dtype=np.float32)/255
        array=(array-mean)/std
        images.append((name,array.transpose(2,0,1)))
    return images


def class_topology(wrapper, image):
    samples=wrapper.predict_samples(image[None,...])
    probs=samples.mean(axis=1)[0]
    top1=int(np.argmax(probs)); top3=np.argsort(probs)[-3:][::-1]
    eps=1e-12; entropy=lambda p: float(-np.sum(p*np.log(np.clip(p,eps,1))))
    h_total=entropy(probs)
    h_alea=float(np.mean([-np.sum(p*np.log(np.clip(p,eps,1))) for p in samples[0]]))
    h_epi=h_total-h_alea
    lo=float(samples[0,:,top1].min()); hi=float(samples[0,:,top1].max())
    return {"samples":samples,"mean_probs":probs,"top1":top1,"top3":top3,
            "confidence":float(probs[top1]),"lo":lo,"hi":hi,"width":hi-lo,
            "h_total":h_total,"h_alea":h_alea,"h_epi":h_epi}


def class_name(categories, index):
    return categories[index] if categories and index < len(categories) else f"class_{index}"


def score_for(wrapper, image, target_class, notion):
    return float(np.asarray(wrapper.predict(image[None,...],notion,target_class)).reshape(-1)[0])


def run_explanation(image, wrapper, target_class, notion):
    explainer=ProblyImageExplainer(
        image,wrapper,index="k-SII",max_order=2,approximator="svarmiq",budget=BUDGET,
        notion=notion,target_class=target_class,segmentation="grid",grid=GRID,batch_size=16,
        random_state=SEED,
    )
    iv=explainer.explain()
    singles={interaction[0]:float(value) for interaction,value in iv.interactions.items() if len(interaction)==1}
    pairs={interaction:float(value) for interaction,value in iv.interactions.items() if len(interaction)==2}
    first=np.asarray([singles.get(i,0.) for i in range(GRID[0]*GRID[1])])
    game=explainer.game
    full=float(game(np.ones((1,game.n_players),dtype=bool))[0])
    empty=float(game(np.zeros((1,game.n_players),dtype=bool))[0])
    efficiency=float(first.sum()+sum(pairs.values()))
    return {"iv":iv,"singles":first,"pairs":pairs,"efficiency_sum":efficiency,
            "efficiency_delta":full-empty,"game":game}


def patch_label(player):
    return f"({player//GRID[1]},{player%GRID[1]})"


def print_header():
    print("\nMC-Dropout image classification and interaction analysis")
    print(f"T={SAMPLES}, dropout p={DROPOUT_P}, k-SII max_order=2, budget={BUDGET}, seed={SEED}, grid={GRID[0]}x{GRID[1]}")
    print("Credal intervals are min/max MC probabilities for each model's predicted top-1 class; entropy is in nats.\n",flush=True)
    print("Image / model | Top-1 (mean p) | Credal [lo, hi], Δp | H_total / H_alea / H_epi | φ_target min..max [patch] | φ_epi min..max [patch] | max |φij| [pair] | efficiency sum vs Δv | seconds")
    print("-"*236,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained",action="store_true",help="load ImageNet pretrained timm ViT and torchvision VGG16 weights")
    parser.add_argument("--images",type=Path,default=DEFAULT_IMAGES)
    parser.add_argument("--device",default="cpu",choices=("cpu","cuda"),help="inference device (default: cpu)")
    args=parser.parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    if not args.pretrained:
        parser.error("This detailed demo is intended for real models. Pass --pretrained to load ImageNet weights.")
    import torchvision
    torch.set_num_threads(min(8,torch.get_num_threads()))
    from torchvision.models import VGG16_Weights
    categories=VGG16_Weights.DEFAULT.meta["categories"]
    vit=ViTShapleyBackbone(model_name="vit_small_patch16_224",pretrained=True,device=args.device)
    vgg=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).to(args.device).eval()
    for module in vgg.modules():
        if isinstance(module,torch.nn.Dropout): module.p=DROPOUT_P
    models=[("ViT",ProblyModelWrapper(vit.model,transformation="dropout",num_samples=SAMPLES,transform_kwargs={"p":DROPOUT_P},mc_chunk_size=4)),
            ("VGG-16",ProblyModelWrapper(vgg,transformation="dropout",num_samples=SAMPLES,transform_kwargs={"p":DROPOUT_P},mc_chunk_size=4))]
    images=load_images(args.images)
    print_header()
    vit_rows=[]; comparison_rows=[]
    total_start=time.perf_counter()
    for image_name,image in images:
        image_started=time.perf_counter()
        masks=None; image_result={"image_name":image_name,"image":image}
        for model_name,wrapper in models:
            started=time.perf_counter()
            prediction=class_topology(wrapper,image)
            target=prediction["top1"]
            target_exp=run_explanation(image,wrapper,target,"target_probability")
            epistemic_exp=run_explanation(image,wrapper,target,"epistemic")
            if masks is None: masks=target_exp["game"].masks
            target_values=target_exp["singles"]; epi_values=epistemic_exp["singles"]
            tmin,tmax=int(np.argmin(target_values)),int(np.argmax(target_values))
            emin,emax=int(np.argmin(epi_values)),int(np.argmax(epi_values))
            strongest=max(epistemic_exp["pairs"].items(),key=lambda item:abs(item[1]),default=((),0.))
            (i,j),interaction_value=strongest
            elapsed=time.perf_counter()-started
            line=(f"{image_name} / {model_name} | {class_name(categories,target)} ({prediction['confidence']:.4f}) | "
                  f"[{prediction['lo']:.4f}, {prediction['hi']:.4f}], {prediction['width']:.4f} | "
                  f"{prediction['h_total']:.3f} / {prediction['h_alea']:.3f} / {prediction['h_epi']:.3f} | "
                  f"{target_values[tmin]:+.4g} {patch_label(tmin)} .. {target_values[tmax]:+.4g} {patch_label(tmax)} | "
                  f"{epi_values[emin]:+.4g} {patch_label(emin)} .. {epi_values[emax]:+.4g} {patch_label(emax)} | "
                  f"{abs(interaction_value):.4g} {patch_label(i)}-{patch_label(j)} | "
                  f"{epistemic_exp['efficiency_sum']:+.5f} vs {epistemic_exp['efficiency_delta']:+.5f} | {elapsed:.1f}s")
            print(line,flush=True)
            if model_name=="ViT":
                top3=prediction["top3"]
                vit_rows.append({"image_name":image_name,"image":image,"masks":masks,
                    "target_values":target_values,"epistemic_values":epi_values,"interactions":epistemic_exp["pairs"],
                    "samples":prediction["samples"],"top3_indices":top3,
                    "top3_names":[class_name(categories,int(c)) for c in top3],
                    "info":f"Top-1: {class_name(categories,target)}\np̄={prediction['confidence']:.3f}\nH_epi={prediction['h_epi']:.3f}"})
            image_result["vit_target" if model_name=="ViT" else "vgg_target"]=target_values
            image_result["vit_epistemic" if model_name=="ViT" else "vgg_epistemic"]=epi_values
        comparison_rows.append({**image_result,"masks":masks})
        print(f"  per-image total: {time.perf_counter()-image_started:.1f}s\n",flush=True)
    outdir=Path(__file__).resolve().parent
    plot_vision_summary(vit_rows,outdir/"vit_detailed_analysis.png")
    plot_vision_summary(vit_rows,outdir/"vit_vs_models_summary.png")
    plot_model_heatmap_comparison(comparison_rows,outdir/"vit_vs_vgg_comparison.png")
    print(f"\nTotal elapsed: {time.perf_counter()-total_start:.1f}s")
    print("Saved vit_vs_models_summary.png, vit_detailed_analysis.png, and vit_vs_vgg_comparison.png")


if __name__=="__main__": main()
