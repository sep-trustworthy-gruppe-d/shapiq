"""Convenience explainers returning native shapiq InteractionValues."""
from shapiq.explainer.agnostic import AgnosticExplainer
from shapiq import ExactComputer
from .games import ProblyImageGame

class _ProblyExplainer:
    def __init__(self, game, *, index="SV", max_order=1, approximator="auto", budget=None, random_state=None, **kwargs):
        self.game=game; self.index=index; self.max_order=max_order; self.approximator=approximator
        self.exact = ExactComputer(game) if approximator == "exact" else None
        aliases={"kernelshap":"regression","kernelshapiq":"regression","svarmiq":"svarm"}
        resolved=aliases.get(approximator,approximator)
        self.explainer=None if self.exact else AgnosticExplainer(game,index=index,max_order=max_order,approximator=resolved,random_state=random_state,**kwargs)
        self.budget=budget
    def explain(self, budget=None, **kwargs):
        if self.exact is not None: return self.exact(index=self.index,order=self.max_order)
        return self.explainer.explain(budget=budget if budget is not None else self.budget,**kwargs)

class ProblyImageExplainer(_ProblyExplainer):
    def __init__(self, image, model, **kwargs):
        game_keys={k:kwargs.pop(k) for k in list(kwargs) if k in {"notion","target_class","segmentation","grid","labels","n_segments","baseline","batch_size"}}
        game=ProblyImageGame(image,model,**game_keys)
        super().__init__(game,**kwargs)

