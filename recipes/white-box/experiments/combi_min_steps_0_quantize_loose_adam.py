import textattack.transformations
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
)

from goal_functions.fluency_bert_score_untargeted_classification import FluencyBertScoreUntargetedClassification
from search_methods.gradient_searchers.gradient_searcher import GradientSearcher
from search_methods.gradient_searchers.utils.optimizers import AdamInputOptimizer
from search_methods.gradient_searchers.utils.step_scheduler import ExponentialStepScheduler


def build_attack(model_wrapper, logging_path: str):
    transformation = textattack.transformations.WordSwapEmbedding()

    constraints = [RepeatModification()]

    goal_function = FluencyBertScoreUntargetedClassification(model_wrapper)

    search_method = GradientSearcher(
        model_wrapper=model_wrapper,
        logging_path=logging_path,
        optimizer_cls=AdamInputOptimizer(lr=1.0, scale=False),
        optimization_steps=40,
        step_scheduler_cls=ExponentialStepScheduler,
        step_scheduler_kwargs={'min_n_steps': 0},
        reset_optimizer_every_quantization=True,
        quantization_proposals_threshold=0.5,
        use_logger_first_samples=0,
        quantization_method=('combi', {'direction': 'max', 'function': 'entropy', 'method': 'mixed'}),
        compute_gradient_method='white-box',
        init_method='original',
        input_type=('token', 'logit'),
    )

    return Attack(goal_function, constraints, transformation, search_method)
