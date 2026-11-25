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


def _heatmap(image, values, masks, ax, *, title, fig=None, cax=None, cmap="coolwarm"):
    ax.imshow(display_image(image))
    v=np.asarray(values,dtype=float); heat=np.zeros(np.asarray(masks).shape[1:],float)
    for value,mask in zip(v,masks): heat[np.asarray(mask,dtype=bool)]=value
    limit=max(float(np.max(np.abs(v))) if len(v) else 0.,1e-12)
    vmin,vmax=-limit,limit
    im=ax.imshow(heat,cmap=cmap,alpha=.58,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
    centers=_centers(masks)
    for i,(cx,cy) in enumerate(centers):
        value=float(v[i]) if i<len(v) else 0.
        ax.text(cx,cy,f"{value:+.3f}",ha="center",va="center",color="white",fontsize=8,weight="bold",
                bbox={"facecolor":"black","alpha":.65,"edgecolor":"white","linewidth":.3,"pad":2})
    if cax is not None:
        cb=(fig or ax.figure).colorbar(im,cax=cax)
        cb.set_ticks([vmin,vmax]); cb.set_ticklabels([f"{vmin:+.3f}",f"{vmax:+.3f}"])
    ax.set_title(f"{title}\nvmin={vmin:+.3f}, vmax={vmax:+.3f}"); ax.set_axis_off()
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
        # The separate 9-node graph carries numeric edge labels; keep this image overlay uncluttered.
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


def plot_interaction_network(interactions, ax=None, *, top_k=5):
    """Draw all nine grid players and the strongest pairwise k-SII edges."""
    ax=ax or plt.gca()
    coords={i:(i%3,2-i//3) for i in range(9)}
    pairs=sorted(((ij,float(v)) for ij,v in interactions.items() if len(ij)==2),key=lambda x:abs(x[1]),reverse=True)[:top_k]
    scale=max((abs(v) for _,v in pairs),default=1.) or 1.
    for rank,((i,j),value) in enumerate(pairs):
        xi,yi=coords[i]; xj,yj=coords[j]
        ax.plot([xi,xj],[yi,yj],color="#c62828" if value>=0 else "#1565c0",linewidth=1+3*abs(value)/scale,alpha=.8,zorder=1)
        dx,dy=xj-xi,yj-yi; norm=(dx*dx+dy*dy)**.5 or 1
        side=1 if rank%2==0 else -1
        tx=(xi+xj)/2+side*(-dy/norm)*.19
        ty=(yi+yj)/2+side*(dx/norm)*.19
        ax.text(tx,ty,f"{value:+.3f}",fontsize=6.5,ha="center",va="center",
                bbox={"facecolor":"white","alpha":.85,"edgecolor":"none","pad":1},zorder=3)
    for i,(x,y) in coords.items():
        ax.scatter(x,y,s=230,color="#fff3e0",edgecolor="#263238",linewidth=1.2,zorder=2)
        ax.text(x,y,str(i),ha="center",va="center",fontsize=9,weight="bold",zorder=4)
    ax.set_xlim(-.5,2.5); ax.set_ylim(-.5,2.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("9-node k-SII graph\n(top pair interactions)")
    return ax


def plot_vision_summary(rows, output_path):
    """Create the ViT overview: original, numeric heatmaps, interaction graph, simplex."""
    n=len(rows)
    fig=plt.figure(figsize=(22,5.3*n))
    gs=fig.add_gridspec(n,7,width_ratios=(1.2,1.2,.12,1.2,.12,.95,1.1),wspace=.18,hspace=.35)
    for r,row in enumerate(rows):
        ax0=fig.add_subplot(gs[r,0]); ax1=fig.add_subplot(gs[r,1]); cb1=fig.add_subplot(gs[r,2])
        ax2=fig.add_subplot(gs[r,3]); cb2=fig.add_subplot(gs[r,4]); ax3=fig.add_subplot(gs[r,5]); ax4=fig.add_subplot(gs[r,6])
        plot_partition(row["image"],row["masks"],ax0,title=row["image_name"],info=row["info"])
        _heatmap(row["image"],row["target_values"],row["masks"],ax1,title="Target probability $\\varphi_i$",fig=fig,cax=cb1)
        _heatmap(row["image"],row["epistemic_values"],row["masks"],ax2,title="Epistemic $\\varphi_i$ + top 5 $\\phi_{ij}$",fig=fig,cax=cb2)
        epi_limit=max(float(np.max(np.abs(row["epistemic_values"]))),1e-12)
        plot_image_interaction_graph(row["image"],row["interactions"],row["masks"],ax2,top_k=5,show_image=False,
            title=f"Epistemic $\\varphi_i$ + top 5 $\\phi_{{ij}}$\nvmin={-epi_limit:+.3f}, vmax={epi_limit:+.3f}")
        plot_interaction_network(row["interactions"],ax3,top_k=5)
        plot_credal_simplex(row["samples"],ax4,class_indices=row["top3_indices"],class_names=row["top3_names"])
    fig.savefig(output_path,dpi=155,bbox_inches="tight"); plt.close(fig)


def plot_model_heatmap_comparison(rows, output_path):
    """Compare numeric ViT/VGG heatmaps, with per-panel metrics and exact scales."""
    fig=plt.figure(figsize=(23,5.4*len(rows)))
    gs=fig.add_gridspec(len(rows),8,width_ratios=(1.1,.10,1.1,.10,1.1,.10,1.1,.10),wspace=.18,hspace=.32)
    for i,row in enumerate(rows):
        for j,(key,meta_key,title) in enumerate((("vit_target","vit_metrics","ViT target probability"),
                ("vit_epistemic","vit_metrics","ViT epistemic"),("vgg_target","vgg_metrics","VGG-16 target probability"),
                ("vgg_epistemic","vgg_metrics","VGG-16 epistemic"))):
            ax=fig.add_subplot(gs[i,2*j]); cax=fig.add_subplot(gs[i,2*j+1]); m=row[meta_key]
            info=(f"{row['image_name']} — {title}\nTop-1: {m['class_name']} | p̄={m['confidence']:.3f} | Δp={m['width']:.3f}\n"
                  f"Htotal={m['h_total']:.3f} | Halea={m['h_alea']:.3f} | Hepi={m['h_epi']:.3f}")
            _heatmap(row["image"],row[key],row["masks"],ax,title=info,fig=fig,cax=cax)
    fig.savefig(output_path,dpi=155,bbox_inches="tight"); plt.close(fig)
