"""Normalize probabilistic predictors to sampled predictions and uncertainty notions."""
from __future__ import annotations
import importlib
import hashlib
import numpy as np
from .models import ModelAdapter

class ProblyModelWrapper:
    TRANSFORMATIONS = {"dropout": "probly.transformation.dropout.torch", "ensemble": "probly.transformation.ensemble.torch",
      "credal_wrapper": "probly.representer", "credal_ensembling": "probly.representer", "laplace": "probly.predictor.laplace", "evidential": "probly.method.evidential"}
    def __init__(self, model=None, transformation=None, *, representer=None, num_samples=16, transform_kwargs=None, predict_fn=None, mc_chunk_size=4):
        self.model, self.transformation, self.representer = model, transformation, representer
        self.num_samples, self.transform_kwargs = int(num_samples), transform_kwargs or {}
        self.mc_chunk_size=max(1,int(mc_chunk_size))
        self._sample_cache = {}
        self.adapter = None if representer is not None else ModelAdapter(model, predict_fn=predict_fn)
        self.transformed = None
        if transformation:
            modname = self.TRANSFORMATIONS.get(transformation)
            if modname:
                try:
                    mod = importlib.import_module(modname)
                    funcs = [getattr(mod, n) for n in dir(mod) if n.startswith("torch_") and callable(getattr(mod,n))]
                    if funcs and model is not None: self.transformed = funcs[0](model, **self.transform_kwargs)
                except (ImportError, TypeError, AttributeError): pass

    def predict_samples(self, x):
        if self.representer is not None:
            out = self.representer(x)
        else:
            obj = self.transformed or self.model
            if self.transformation == "dropout" and self.adapter:
                try:
                    import torch
                    if isinstance(obj, torch.nn.Module):
                        array=np.asarray(x,dtype=np.float32)
                        keys=[hashlib.blake2b(np.ascontiguousarray(row).view(np.uint8),digest_size=16).digest() for row in array]
                        missing=[]; missing_idx=[]
                        for i,key in enumerate(keys):
                            if key not in self._sample_cache and key not in missing:
                                missing.append(key); missing_idx.append(i)
                        if missing:
                            device=next(obj.parameters(),torch.empty(0)).device
                            xx=torch.as_tensor(array[missing_idx],dtype=torch.float32,device=device)
                            previous=obj.training; obj.train(); chunks=[]
                            with torch.no_grad():
                                for start in range(0,self.num_samples,self.mc_chunk_size):
                                    count=min(self.mc_chunk_size,self.num_samples-start)
                                    repeated=torch.cat([xx]*count,dim=0)
                                    prediction=obj(repeated)
                                    if isinstance(prediction,(tuple,list)): prediction=prediction[0]
                                    chunks.append(prediction.detach().cpu().numpy().reshape(count,len(missing),-1).transpose(1,0,2))
                            obj.train(previous)
                            sampled=np.concatenate(chunks,axis=1)
                            sampled=self._to_probabilities(sampled)
                            for key,row in zip(missing,sampled): self._sample_cache[key]=row
                        out=np.stack([self._sample_cache[key] for key in keys],axis=0)
                    else: out=self.adapter.predict(x)
                except ImportError: out=self.adapter.predict(x)
            elif self.transformation in ("ensemble", "credal_ensembling") and isinstance(obj,(list,tuple)):
                out=np.stack([ModelAdapter(m).predict(x) for m in obj],axis=1)
            else: out=self.adapter.predict(x)
        out=np.asarray(out)
        if out.ndim == 1: out=out[:,None]
        if out.ndim == 2: out=out[:,None,:]
        return self._to_probabilities(out)

    @staticmethod
    def _to_probabilities(out):
        out=np.asarray(out)
        if np.any(out < 0) or not np.allclose(out.sum(-1),1,atol=1e-4):
            z=out-out.max(axis=-1,keepdims=True); e=np.exp(z); out=e/e.sum(axis=-1,keepdims=True)
        return out

    def predict(self, x, notion="total", target_class=0):
        return self.predict_from_samples(self.predict_samples(x), notion, target_class)

    @staticmethod
    def predict_from_samples(a, notion="total", target_class=0):
        a=np.asarray(a)
        eps=1e-12; mean=a.mean(axis=1); ent=lambda p: -(p*np.log(np.clip(p,eps,1))).sum(-1)
        if notion=="target_probability": return mean[:,target_class]
        if notion=="total": return ent(mean)
        if notion=="aleatoric": return ent(a).mean(axis=1)
        if notion=="epistemic": return ent(mean)-ent(a).mean(axis=1)
        if notion=="credal_width": return a[:,:,target_class].max(axis=1)-a[:,:,target_class].min(axis=1)
        if notion=="credal_area":
            from scipy.spatial import ConvexHull
            result=[]
            for row in a:
                top=np.argsort(row.mean(axis=0))[-3:]; pts=row[:,top]
                try: result.append(float(ConvexHull(pts[:,:2]).volume))
                except Exception: result.append(0.)
            return np.asarray(result)
        raise ValueError(f"Unknown notion: {notion}")
