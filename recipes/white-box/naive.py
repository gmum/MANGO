from typing import Optional

import textattack.transformations
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
)

from goal_functions.fluency_bert_score_untargeted_classification import FluencyBertScoreUntargetedClassification
from search_methods.gradient_searchers.gradient_searcher import GradientSearcher
from search_methods.gradient_searchers.utils.optimizers import AdamInputOptimizer
from search_methods.gradient_searchers.utils.step_scheduler import ExponentialStepScheduler


def build_attack(model_wrapper, logging_path: str, disable_column_name: Optional[str] = None):
    transformation = textattack.transformations.WordSwapEmbedding()

    constraints = [RepeatModification()]

    goal_function = FluencyBertScoreUntargetedClassification(model_wrapper)

    search_method = GradientSearcher(
        model_wrapper=model_wrapper,
        logging_path=logging_path,
        optimizer_cls=AdamInputOptimizer,
        optimizer_kwargs={'lr': 1.0},
        optimization_steps=100,
        step_scheduler_cls=ExponentialStepScheduler,
        quantization_method=('naive', None),
        compute_gradient_method='white-box',
        init_method='original',
        mutable_column_name=disable_column_name,
        input_type=('token', 'logit'),
    )

    return Attack(goal_function, constraints, transformation, search_method)
