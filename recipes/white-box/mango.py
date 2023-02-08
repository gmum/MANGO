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

    goal_function = FluencyBertScoreUntargetedClassification(model_wrapper, bert_score_lambda=20.0, use_idf=True)

    search_method = GradientSearcher(
        model_wrapper=model_wrapper,
        logging_path=logging_path,
        optimizer_cls=AdamInputOptimizer,
        optimizer_kwargs={'lr': 1.0},
        optimization_steps=100,
        step_scheduler_cls=ExponentialStepScheduler,
        step_scheduler_kwargs={'min_n_steps': 0},
        quantization_method=('mango', {'direction': 'max', 'function': 'entropy', 'method': 'mixed'}),
        compute_gradient_method='white-box',
        quantization_proposals_threshold=0.5,
        reset_optimizer_every_quantization=True,
        check_for_ones=False,
        init_method='original',
        input_type=('token', 'logit'),
        mutable_column_name=disable_column_name,
    )

    return Attack(goal_function, constraints, transformation, search_method)
