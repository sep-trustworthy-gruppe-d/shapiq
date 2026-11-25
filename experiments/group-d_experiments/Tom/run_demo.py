"""Run probly MC-Dropout and shapiq interaction analyses on example images."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tom import ProblyModelWrapper
from Tom.games import ProblyImageGame
from Tom.models import ViTShapleyBackbone
from Tom.visualization import plot_vision_summary, plot_model_heatmap_comparison
from shapiq import ExactComputer


DEFAULT_IMAGES=Path("~/probly/experiments/group-d_experiments/Tom/Pictures").expanduser()
IMAGE_NAMES=("GoldenRetriever.jpeg", "DogCat.jpeg", "blurryFrog.jpeg")
SAMPLES=10
DROPOUT_P=.3
BUDGET=512
SEED=42
GRID=(3,3)


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


def run_explanation(image, wrapper, target_class, notion):
    game=ProblyImageGame(image,wrapper,notion=notion,target_class=target_class,
        segmentation="grid",grid=GRID,batch_size=16)
    # With N=9 the coalition space has exactly 512 rows; enumerate it once.
    iv=ExactComputer(game)(index="k-SII",order=2)
    singles={interaction[0]:float(value) for interaction,value in iv.interactions.items() if len(interaction)==1}
    pairs={interaction:float(value) for interaction,value in iv.interactions.items() if len(interaction)==2}
    first=np.asarray([singles.get(i,0.) for i in range(GRID[0]*GRID[1])])
    full=float(game(np.ones((1,game.n_players),dtype=bool))[0])
    empty=float(game(np.zeros((1,game.n_players),dtype=bool))[0])
    efficiency=float(sum(value for interaction,value in iv.interactions.items() if interaction))
    return {"iv":iv,"singles":first,"pairs":pairs,"efficiency_sum":efficiency,
            "efficiency_delta":full-empty,"game":game}


def patch_label(player):
    return f"({player//GRID[1]},{player%GRID[1]})"


def top_pairs(pairs, count=5):
    return sorted(pairs.items(),key=lambda item:abs(item[1]),reverse=True)[:count]


def matrix_markdown(values):
    values=np.asarray(values).reshape(GRID)
    rows=["| " + " | ".join(f"{v:+.6f}" for v in row) + " |" for row in values]
    return "\n".join(["| φ(row,col) | 0 | 1 | 2 |","|---:|---:|---:|---:|"]+rows)


def pairs_markdown(pairs):
    if not pairs: return "| Patchpaar | φᵢⱼ |\n|---|---:|\n| – | 0.000000 |"
    return "\n".join(["| Patchpaar (row,col) | φᵢⱼ |","|---|---:|"]+
        [f"| {patch_label(i)} ↔ {patch_label(j)} | {value:+.6f} |" for (i,j),value in top_pairs(pairs)])


def write_results(path, records):
    lines=["# Probly × shapiq: Vision-Ergebnisse", "",
        f"Konfiguration: vortrainierte ImageNet-Modelle, MC-Dropout T={SAMPLES}, p={DROPOUT_P}, "
        f"3×3-Grid (N=9), exaktes k-SII bis Ordnung 2 via `ExactComputer` (2^9={2**9} Koalitionen), "
        f"Budgetobergrenze={BUDGET}, Seed={SEED}, CPU-Threads=2.", "",
        "Credal-Intervalle beziehen sich auf die vorhergesagte Top-1-Klasse; Entropien sind in Nats. "
        "Die φ-Matrizen sind zeilenweise nach den neun Grid-Feldern geordnet.", ""]
    for rec in records:
        m=rec["metrics"]
        lines += [f"## {rec['image_name']} — {rec['model_name']}", "",
            f"- Top-1: **{m['class_name']}** (ImageNet-Index {m['top1']})",
            f"- Mittlere Konfidenz p̄: `{m['confidence']:.6f}`",
            f"- Credal-Intervall: `[{m['lo']:.6f}, {m['hi']:.6f}]`; Δp: `{m['width']:.6f}`",
            f"- Entropie: H_total `{m['h_total']:.6f}`; H_alea `{m['h_alea']:.6f}`; H_epi `{m['h_epi']:.6f}`", "",
            "### 1st-Order φ für `target_probability`", "", matrix_markdown(rec["target"]["singles"]), "",
            "### 1st-Order φ für `epistemic`", "", matrix_markdown(rec["epistemic"]["singles"]), "",
            "### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)", "", pairs_markdown(rec["target"]["pairs"]), "",
            "### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)", "", pairs_markdown(rec["epistemic"]["pairs"]), "",
            "### Effizienzprüfung", "",
            "| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |", "|---|---:|---:|---:|"]
        for notion in ("target","epistemic"):
            e=rec[notion]; err=abs(e["efficiency_sum"]-e["efficiency_delta"])
            lines.append(f"| {notion} | {e['efficiency_sum']:+.9f} | {e['efficiency_delta']:+.9f} | {err:.3e} |")
        lines += ["",f"Gemessene Laufzeit für Vorhersage und beide Erklärungen: **{rec['elapsed']:.2f} s**.",""]
    Path(path).write_text("\n".join(lines),encoding="utf-8")


def print_header():
    print("\nMC-Dropout image classification and interaction analysis")
    print(f"T={SAMPLES}, dropout p={DROPOUT_P}, exact k-SII max_order=2, budget={BUDGET}, seed={SEED}, grid={GRID[0]}x{GRID[1]} (N=9)")
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
    vit_rows=[]; comparison_rows=[]; records=[]
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
            strongest=max(epistemic_exp["pairs"].items(),key=lambda item:abs(item[1]),default=((0,1),0.))
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
            metrics={**prediction,"class_name":class_name(categories,target)}
            records.append({"image_name":image_name,"model_name":model_name,"metrics":metrics,
                "target":target_exp,"epistemic":epistemic_exp,"elapsed":elapsed})
            image_result["vit_metrics" if model_name=="ViT" else "vgg_metrics"]=metrics
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
    plot_model_heatmap_comparison(comparison_rows,outdir/"vit_vs_vgg_comparison.png")
    write_results(outdir/"results_summary.md",records)
    print(f"\nTotal elapsed: {time.perf_counter()-total_start:.1f}s")
    print("Saved vit_detailed_analysis.png, vit_vs_vgg_comparison.png, and results_summary.md")


if __name__=="__main__": main()
