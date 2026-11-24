"""Model adapters for torch, sklearn and arbitrary prediction callables."""
from __future__ import annotations

from typing import Any, Callable
import numpy as np


class ModelAdapter:
    def __init__(self, model: Any, *, predict_fn: Callable | None = None, device: str | None = None):
        self.model, self.predict_fn, self.device = model, predict_fn, device

    def predict(self, x: Any) -> np.ndarray:
        if self.predict_fn is not None:
            return np.asarray(self.predict_fn(x))
        if callable(getattr(self.model, "predict_proba", None)):
            return np.asarray(self.model.predict_proba(np.asarray(x)))
        if callable(getattr(self.model, "predict", None)) and not hasattr(self.model, "parameters"):
            return np.asarray(self.model.predict(np.asarray(x)))
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                dev = self.device or str(next(self.model.parameters(), torch.empty(0)).device)
                tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
                with torch.no_grad(): out = self.model(tensor)
                if isinstance(out, (tuple, list)): out = out[0]
                return out.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(self.model(x))


class ViTShapleyBackbone(ModelAdapter):
    """Optional timm ViT with pixel masking and token/patch masking support."""
    def __init__(self, model=None, *, checkpoint_path=None, model_name="vit_small_patch16_224", device=None, pretrained=False):
        import torch
        if model is None:
            import timm
            model = timm.create_model(model_name, pretrained=pretrained)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("state_dict", state), strict=False)
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        super().__init__(model, device=self.device)

    def predict_tokens(self, images, active_patches):
        """Predict after zeroing inactive patch tokens when timm exposes token APIs."""
        import torch
        model = self.model
        x = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        if not all(hasattr(model, name) for name in ("patch_embed", "pos_drop", "blocks", "norm")):
            return self.predict(images)
        with torch.no_grad():
            tokens = model.patch_embed(x)
            if getattr(model, "cls_token", None) is not None:
                cls = model.cls_token.expand(tokens.shape[0], -1, -1)
                tokens = torch.cat((cls, tokens), dim=1)
            if getattr(model, "pos_embed", None) is not None: tokens = tokens + model.pos_embed
            mask = torch.as_tensor(active_patches, dtype=tokens.dtype, device=tokens.device)
            if mask.ndim == 1: mask = mask[None, :]
            offset = tokens.shape[1] - mask.shape[1]
            tokens[:, offset:] *= mask[:, :, None]
            tokens = model.pos_drop(tokens)
            for block in model.blocks: tokens = block(tokens)
            tokens = model.norm(tokens)
            if getattr(model, "fc_norm", None) is not None: tokens = model.fc_norm(tokens.mean(dim=1))
            elif getattr(model, "cls_token", None) is not None: tokens = tokens[:, 0]
            else: tokens = tokens.mean(dim=1)
            head = getattr(model, "head", None)
            return (head(tokens) if head is not None else tokens).cpu().numpy()


ViTAdapter = ViTShapleyBackbone
