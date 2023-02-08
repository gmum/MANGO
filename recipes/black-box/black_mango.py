from typing import Optional

import textattack.transformations
import transformers
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
)

from goal_functions.fluency_bert_score_untargeted_classification import FluencyBertScoreUntargetedClassification
from search_methods.gradient_searchers.gradient_searcher import GradientSearcher
from search_methods.gradient_searchers.utils.index_constraints import SpecialIndexConstraint, StopwordIndexConstraint, \
    ComposedIndexConstraint
from search_methods.gradient_searchers.utils.optimizers import SGDInputOptimizer, AdamInputOptimizer
from search_methods.gradient_searchers.utils.step_scheduler import IdentityStepScheduler, ExponentialStepScheduler
from search_methods.gradient_searchers.utils.token_constraints import SpecialTokenTokenConstraint, \
    StopwordTokenConstraint, EmbeddingSimilarityTokenConstraint, ComposedTokenConstraint


def build_attack(model_wrapper, logging_path: str, disable_column_name: Optional[str] = None):
    transformation = textattack.transformations.WordSwapEmbedding()

    constraints = [RepeatModification()]

    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    index_constraints = [
        SpecialIndexConstraint(tokenizer),
    ]
    token_constraints = [
        SpecialTokenTokenConstraint(tokenizer),
        EmbeddingSimilarityTokenConstraint(tokenizer, embedding_path='glove_embeddings.pt', min_cos_sim=0.0)
    ]

    goal_function = FluencyBertScoreUntargetedClassification(model_wrapper, bert_score_lambda=80.0)

    search_method = GradientSearcher(
        model_wrapper=model_wrapper,
        logging_path=logging_path,
        optimizer_cls=AdamInputOptimizer,
        optimizer_kwargs={'lr': 1.0, 'ams_grad': True},
        optimization_steps=150,
        step_scheduler_cls=ExponentialStepScheduler,
        step_scheduler_kwargs={'min_n_steps': 0},
        quantization_method=('combi', {
            'direction': 'max', 'function': 'entropy', 'method': 'one_hot'}),
        compute_gradient_method='zoo',
        batch_size=5,
        fake_batch_multiplier=2,
        quantization_proposals_threshold=0.3,
        init_method='original',
        mi=0.1,
        input_type=('token', 'logit'),
        mutable_column_name=disable_column_name,
        index_constraint=ComposedIndexConstraint(index_constraints),
        token_constraint=ComposedTokenConstraint(token_constraints)
    )

    return Attack(goal_function, constraints, transformation, search_method)
