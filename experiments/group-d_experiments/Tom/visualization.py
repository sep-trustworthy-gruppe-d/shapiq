"""Image-space plots for Shapley values, pair interactions, and credal sets."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def display_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert a CHW float image (optionally ImageNet-normalized) to display RGB."""
    a=np.asarray(image,dtype=float)
    if a.ndim==3 and a.shape[0] in (1,3): a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[-1]==3:
        # Normalize only when values look like ImageNet-normalized tensors.
        if a.min() < -0.05 or a.max() > 1.05: a=a*np.asarray(std)+np.asarray(mean)
        a=np.clip(a,0,1)
    return a


def _centers(masks):
    centers=[]
    for mask in np.asarray(masks,dtype=bool):
        y,x=np.nonzero(mask); centers.append((x.mean(),y.mean()) if len(x) else (0.,0.))
    return np.asarray(centers)


def _heatmap(image, values, masks, ax, *, title, cmap="coolwarm"):
    ax.imshow(display_image(image))
    v=np.asarray(values,dtype=float); heat=np.zeros(np.asarray(masks).shape[1:],float)
    for value,mask in zip(v,masks): heat[np.asarray(mask,dtype=bool)]=value
    limit=max(float(np.max(np.abs(v))) if len(v) else 0.,1e-12)
    ax.imshow(heat,cmap=cmap,alpha=.58,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    ax.set_title(title); ax.set_axis_off()
    return heat


def plot_image_heatmap(image, values, masks, ax=None, *, title="Patch attribution"):
    """Overlay signed first-order scores on the original image."""
    ax=ax or plt.gca()
    return _heatmap(image,values,masks,ax,title=title)


def plot_partition(image, masks, ax=None, *, title="Image patches", info=None):
    """Show the original image, region boundaries, and a compact prediction panel."""
    ax=ax or plt.gca(); ax.imshow(display_image(image))
    for mask in masks:
        ax.contour(np.asarray(mask,dtype=float),levels=[.5],colors="white",linewidths=.55,alpha=.9)
    if info:
        ax.text(.02,.02,info,transform=ax.transAxes,color="white",fontsize=8,va="bottom",
                bbox={"facecolor":"black","alpha":.72,"pad":4})
    ax.set_title(title); ax.set_axis_off(); return ax


def plot_image_interaction_graph(image, interactions, masks, ax=None, threshold=0.0, *, top_k=5, title="Epistemic attribution + interactions", show_image=True):
    """Overlay the strongest pairwise Shapley interactions as centroid edges."""
    ax=ax or plt.gca()
    if show_image: ax.imshow(display_image(image))
    centers=_centers(masks)
    pairs=[(ij,float(value)) for ij,value in interactions.items() if len(ij)==2 and abs(value)>=threshold]
    pairs=sorted(pairs,key=lambda item:abs(item[1]),reverse=True)[:top_k]
    for rank,((i,j),value) in enumerate(pairs):
        if max(i,j)>=len(centers): continue
        (xi,yi),(xj,yj)=centers[i],centers[j]
        ax.plot([xi,xj],[yi,yj],color="#ffdf00" if value>=0 else "#00e5ff",
                linewidth=1.1+2.2*abs(value)/(abs(pairs[0][1])+1e-12),alpha=.95,zorder=4)
        ax.scatter([xi,xj],[yi,yj],s=17,color="white",edgecolor="black",linewidth=.4,zorder=5)
        ax.text((xi+xj)/2,(yi+yj)/2,f"{rank+1}: {value:+.3g}",color="white",fontsize=7,
                bbox={"facecolor":"black","alpha":.65,"pad":1},zorder=6)
    ax.set_title(title); ax.set_axis_off(); return ax


def plot_credal_simplex(samples, ax=None, *, class_indices=None, class_names=None, title="MC-Dropout credal set"):
    """Plot top-three-class MC samples and the convex hull as a 2D simplex."""
    from scipy.spatial import ConvexHull
    p=np.asarray(samples,dtype=float)
    if p.ndim==3: p=p[0]
    if class_indices is None: class_indices=np.argsort(p.mean(axis=0))[-3:][::-1]
    q=p[:,np.asarray(class_indices,dtype=int)]
    q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
    xy=np.c_[q[:,1]+.5*q[:,2],np.sqrt(3)/2*q[:,2]]
    ax=ax or plt.gca()
    ax.plot([0,1,.5,0],[0,0,np.sqrt(3)/2,0],color="0.2",linewidth=1.2)
    ax.scatter(xy[:,0],xy[:,1],s=16,alpha=.62,color="#2864dc",label="MC samples")
    if len(xy)>=3:
        try:
            hull=ConvexHull(xy); vertices=xy[hull.vertices]; vertices=np.vstack([vertices,vertices[0]])
            ax.fill(vertices[:,0],vertices[:,1],color="#72a4ff",alpha=.22,label="Credal hull")
            ax.plot(vertices[:,0],vertices[:,1],color="#1746a2",linewidth=1.5)
        except Exception: pass
    names=class_names or [str(i) for i in class_indices]
    ax.text(-.03,-.06,names[0],ha="right",va="top",fontsize=7)
    ax.text(1.03,-.06,names[1],ha="left",va="top",fontsize=7)
    ax.text(.5,np.sqrt(3)/2+.045,names[2],ha="center",va="bottom",fontsize=7)
    ax.set_xlim(-.18,1.18); ax.set_ylim(-.16,1.02); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    return ax


def plot_vision_summary(rows, output_path):
    """Create the ViT-only four-column image overview for a list of result records."""
    n=len(rows); fig,axes=plt.subplots(n,4,figsize=(16,4.2*n),squeeze=False)
    for r,row in enumerate(rows):
        plot_partition(row["image"],row["masks"],axes[r,0],title=row["image_name"],info=row["info"])
        _heatmap(row["image"],row["target_values"],row["masks"],axes[r,1],title="Target probability $\\varphi_P$")
        epi_heat=_heatmap(row["image"],row["epistemic_values"],row["masks"],axes[r,2],title="Epistemic $\\varphi_{epi}$ + top interactions")
        plot_image_interaction_graph(row["image"],row["interactions"],row["masks"],axes[r,2],top_k=5,show_image=False)
        plot_credal_simplex(row["samples"],axes[r,3],class_indices=row["top3_indices"],class_names=row["top3_names"])
    fig.tight_layout(); fig.savefig(output_path,dpi=160,bbox_inches="tight"); plt.close(fig)


def plot_model_heatmap_comparison(rows, output_path):
    """Compare ViT/VGG target-probability and epistemic maps for every image."""
    fig,axes=plt.subplots(len(rows),4,figsize=(17,4.2*len(rows)),squeeze=False)
    for i,row in enumerate(rows):
        for j,key in enumerate(("vit_target","vit_epistemic","vgg_target","vgg_epistemic")):
            title={"vit_target":"ViT target probability","vit_epistemic":"ViT epistemic",
                   "vgg_target":"VGG-16 target probability","vgg_epistemic":"VGG-16 epistemic"}[key]
            _heatmap(row["image"],row[key],row["masks"],axes[i,j],title=f"{row['image_name']}\n{title}")
    fig.tight_layout(); fig.savefig(output_path,dpi=160,bbox_inches="tight"); plt.close(fig)
