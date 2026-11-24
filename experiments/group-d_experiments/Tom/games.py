"""Image coalitional game backed by a ProblyModelWrapper."""
from __future__ import annotations
import numpy as np
from shapiq import Game
from .probly_adapter import ProblyModelWrapper

def _segments(shape, segmentation="grid", grid=(4,4), labels=None, n_segments=16):
    h,w=shape
    if segmentation=="custom":
        if labels is None: raise ValueError("custom segmentation requires labels")
        lab=np.asarray(labels)
        if lab.ndim==3: return lab.astype(bool)
        ids=np.unique(lab); return np.stack([lab==i for i in ids])
    if segmentation in ("patches","grid"):
        gh,gw=grid; yy,xx=np.indices((h,w)); return np.stack([(yy*gh//h==i)&(xx*gw//w==j) for i in range(gh) for j in range(gw)])
    if segmentation=="slic":
        from skimage.segmentation import slic
        lab=slic(np.zeros((h,w,3),dtype=float),n_segments=n_segments,channel_axis=-1,start_label=0)
        return np.stack([lab==i for i in np.unique(lab)])
    raise ValueError(f"Unknown segmentation: {segmentation}")

class ProblyImageGame(Game):
    def __init__(self, image, model, *, notion="target_probability", target_class=0, segmentation="grid", grid=(4,4), labels=None, n_segments=16, baseline=None, batch_size=32, **kwargs):
        self.image=np.asarray(image); self.wrapper=model if isinstance(model,ProblyModelWrapper) else ProblyModelWrapper(model)
        self.notion,self.target_class,self.batch_size=notion,target_class,batch_size
        self.segmentation=segmentation
        self.masks=_segments(self.image.shape[-2:],segmentation,grid,labels,n_segments)
        self._cache={}
        empty=self._evaluate(np.zeros(self.masks.shape[0],bool)) if baseline is None else float(baseline)
        super().__init__(self.masks.shape[0],normalization_value=empty,**kwargs)
    def _evaluate(self, coalition):
        key=tuple(np.asarray(coalition,dtype=bool));
        if key not in self._cache:
            native = (self.segmentation == "patches" and self.wrapper.transformation is None
                      and callable(getattr(self.wrapper.model, "predict_tokens", None)))
            if native:
                logits=np.asarray(self.wrapper.model.predict_tokens(self.image[None,...],np.asarray(key)[None,:]))
                logits=logits-logits.max(axis=-1,keepdims=True); probs=np.exp(logits); probs/=probs.sum(axis=-1,keepdims=True)
                samples=probs[:,None,:]
                value=self.wrapper.predict_from_samples(samples,self.notion,self.target_class)
                self._cache[key]=float(np.asarray(value).reshape(-1)[0])
                return self._cache[key]
            mask=np.any(self.masks[np.asarray(key)],axis=0) if np.any(key) else np.zeros(self.image.shape[-2:],bool)
            x=self.image.copy(); x[...,~mask]=0
            self._cache[key]=float(np.asarray(self.wrapper.predict(x[None,...],self.notion,self.target_class)).reshape(-1)[0])
        return self._cache[key]
    def value_function(self, coalitions):
        coalitions=np.asarray(coalitions,dtype=bool)
        if (self.segmentation == "patches" and self.wrapper.transformation is None
                and callable(getattr(self.wrapper.model, "predict_tokens", None))):
            return np.asarray([self._evaluate(c) for c in coalitions],dtype=float)
        missing=[]
        for coalition in coalitions:
            key=tuple(coalition)
            if key not in self._cache and key not in missing: missing.append(key)
        for start in range(0,len(missing),max(1,self.batch_size)):
            keys=missing[start:start+max(1,self.batch_size)]
            images=[]
            for key in keys:
                mask=np.any(self.masks[np.asarray(key)],axis=0) if any(key) else np.zeros(self.image.shape[-2:],bool)
                image=self.image.copy(); image[...,~mask]=0; images.append(image)
            values=np.asarray(self.wrapper.predict(np.stack(images),self.notion,self.target_class)).reshape(-1)
            self._cache.update(zip(keys,values.tolist()))
        return np.asarray([self._cache[tuple(c)] for c in coalitions],dtype=float)
