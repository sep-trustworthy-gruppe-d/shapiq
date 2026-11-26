"""Explain the uncertainty of a model with Shapley values and interactions.

The :class:`ProblyExplainer` shows which players (features, image patches, ...) raise or lower the
uncertainty (total, aleatoric or epistemic) of a model with several predictions per input, for
example a probly model. A masker defines the players, so tabular data, images and custom data work.
Requires ``torch``, ``transformers`` and ``probly`` only for :class:`ProblyViT`.
"""

from .explainer import ProblyExplainer
from .imputer import ProblyImputer
from .maskers import CallableMasker, FeatureMasker, Masker, PatchMasker
from .measures import uncertainty
from .vit import ProblyViT

__all__ = [
    "CallableMasker",
    "FeatureMasker",
    "Masker",
    "PatchMasker",
    "ProblyExplainer",
    "ProblyImputer",
    "ProblyViT",
    "uncertainty",
]
