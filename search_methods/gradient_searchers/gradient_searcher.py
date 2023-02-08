import itertools
import os.path
from collections import OrderedDict
from functools import partial
from itertools import groupby
from typing import Type, List, Tuple, Any, Optional, Dict

import torch
import torch.nn.functional as F
from textattack import shared
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult
from textattack.goal_function_results import GoalFunctionResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BatchEncoding

from goal_functions.fluency_bert_score_untargeted_classification import FluencyBertScoreUntargetedClassification
from search_methods.gradient_searchers.utils.index_constraints import IndexConstraintBase, SpecialIndexConstraint
from search_methods.gradient_searchers.utils.initialization import initialize_logit_at_original_value, \
    initialize_logit_normal, initialize_one_hot_normal
from search_methods.gradient_searchers.utils.optimizers import InputOptimizerBase, SGDInputOptimizer
from search_methods.gradient_searchers.utils.other import FakeSummaryWriter, compute_gradient_cosine_similarity_fast
from search_methods.gradient_searchers.utils.step_scheduler import StepSchedulerBase, ExponentialStepScheduler
from search_methods.gradient_searchers.utils.token_constraints import TokenConstraintBase, SpecialTokenTokenConstraint


class GradientSearcher(SearchMethod):
    def __init__(self,
                 model_wrapper: HuggingFaceModelWrapper,
                 logging_path: str,
                 use_logger_first_samples: int = 0,
                 optimizer_cls: Type[InputOptimizerBase] = SGDInputOptimizer,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 optimization_steps: int = 4,
                 step_scheduler_cls: Type[StepSchedulerBase] = ExponentialStepScheduler,
                 step_scheduler_kwargs: Optional[Dict[str, Any]] = None,
                 quantization_method: Tuple[str, Optional[Dict[str, Any]]] = ('naive', None),
                 compute_gradient_method: str = 'white-box',
                 init_method: str = 'normal',
                 input_type: Tuple[str, str] = ('token', 'one-hot'),
                 batch_size: int = 1,
                 fake_batch_multiplier: int = 1,
                 mi: float = 0.01,
                 noise_sampling_method: str = 'normal',
                 check_for_ones: bool = False,
                 max_quantization_proposals: int = 5,
                 quantization_proposals_threshold: Optional[float] = None,
                 reset_optimizer_every_quantization: bool = False,
                 mutable_column_name: Optional[str] = None,
                 index_constraint: Optional[IndexConstraintBase] = None,
                 token_constraint: Optional[TokenConstraintBase] = None,
                 dump_score: bool = False
                 ):
        self.tokenizer: BertTokenizer = model_wrapper.tokenizer
        self.goal_function: FluencyBertScoreUntargetedClassification = ...
        self.special_tokens: List[str] = ...
        self._encodings: BatchEncoding = ...
        self.tricky_words_indices = ...
        self.mi = mi
        self.noise_sampling_method = noise_sampling_method
        self.max_quantization_proposals = max_quantization_proposals
        self.quantization_proposals_threshold = quantization_proposals_threshold
        self.optimization_steps = optimization_steps
        optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.optimizer = optimizer_cls(**optimizer_kwargs)
        self.step_scheduler_cls = step_scheduler_cls
        self.step_scheduler_kwargs = step_scheduler_kwargs if step_scheduler_kwargs else {}
        self.quantization_method = quantization_method
        self.compute_gradient_method = compute_gradient_method
        self.input_type = input_type
        self.init_method = init_method
        self.batch_size = batch_size
        self.logging_path = logging_path
        self.fake_batch_multiplier = fake_batch_multiplier
        self.use_logger_first_samples = use_logger_first_samples
        self.logger = SummaryWriter(logging_path) if use_logger_first_samples > 0 else FakeSummaryWriter()
        self.samples_count = 0
        self.tokens_count = 0
        self.steps_count = 0
        self.check_for_ones = check_for_ones
        self.reset_optimizer_every_quantization = reset_optimizer_every_quantization
        self.mutable_token_type_id = {'premise': 0, 'hypothesis': 1}[
            mutable_column_name] if mutable_column_name else None
        self.index_constraint = index_constraint if index_constraint else SpecialIndexConstraint(self.tokenizer)
        self.token_constraint = token_constraint if token_constraint else SpecialTokenTokenConstraint(self.tokenizer)
        self.dump_score = dump_score
        self.dumped_score = []

    def prepare(self, attacked_text: AttackedText):
        self.special_tokens = list(",.-'") + [self.tokenizer.cls_token, self.tokenizer.sep_token]
        self.tricky_words_indices = [i for i, w in enumerate(attacked_text.words) if '-' in w]
        self._encodings = None
        self.samples_count += 1
        if self.samples_count > self.use_logger_first_samples:
            self.logger = FakeSummaryWriter()
        self.tokens_count = 0
        self.steps_count = 0
        self.dumped_score = []

    def get_input_masks(self, attack_text: AttackedText) -> Tuple[Tensor, Tensor, Tensor]:
        word_or_token, _ = self.input_type
        if word_or_token == 'token':
            return self._get_token_input_masks(attack_text)
        else:
            raise NotImplementedError()

    def _get_token_input_masks(self, attacked_text: AttackedText) -> Tuple[Tensor, Tensor, Tensor]:
        encodings = self.get_tokenizer_original_encodings(attacked_text)
        token_type_ids = encodings['token_type_ids'].squeeze(0)
        input_ids = encodings['input_ids'].squeeze(0)

        if self.mutable_token_type_id is not None:
            disabled_tokens = token_type_ids != self.mutable_token_type_id
        else:
            disabled_tokens = torch.zeros_like(token_type_ids).bool()

        vocab_size = self.tokenizer.vocab_size
        n = len(input_ids)
        stochastic_mask = torch.zeros(size=(n, vocab_size), dtype=torch.bool)
        quantized_mask = torch.zeros(size=(n, vocab_size), dtype=torch.bool)
        original_mask = torch.zeros(size=(n, vocab_size), dtype=torch.bool)
        for i, (input_id, disabled) in enumerate(zip(input_ids, disabled_tokens)):
            original_mask[i, input_id] = True
            if disabled or not self.index_constraint.is_allowed(input_id):
                stochastic_mask[i] = False
                quantized_mask[i, input_id] = True
            else:
                allowed_mask = self.token_constraint.get_allowed_mask(input_id)
                if torch.sum(allowed_mask) <= 1:
                    stochastic_mask[i] = False
                    quantized_mask[i, input_id] = True
                else:
                    quantized_mask[i] = False
                    stochastic_mask[i, :] = allowed_mask
        # print(f'Tokens to optimize: {torch.sum(stochastic_mask).item()}.')
        return stochastic_mask, quantized_mask, original_mask

    def initialize_input(self, stochastic_mask: Tensor, quantized_mask: Tensor, original_mask: Tensor) -> Tensor:
        _, one_hot_or_logit = self.input_type
        if one_hot_or_logit == 'one-hot':
            if self.init_method == 'normal':
                return initialize_one_hot_normal(stochastic_mask, quantized_mask)
            else:
                raise NotImplementedError()
        elif one_hot_or_logit in ['logit', 'gumbel-logit']:
            if self.init_method == 'normal':
                return initialize_logit_normal(stochastic_mask, quantized_mask)
            elif self.init_method == 'original':
                return initialize_logit_at_original_value(stochastic_mask, quantized_mask, original_mask)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def input_to_bert_one_hot(self, input: Tensor) -> Tensor:
        word_or_token, _ = self.input_type
        if word_or_token == 'token':
            input = input
        else:
            raise NotImplementedError()

        return self.input_to_one_hot(input)

    def input_to_one_hot(self, input: Tensor) -> Tensor:
        _, one_hot_or_logit = self.input_type
        if one_hot_or_logit == 'one-hot':
            return input
        elif one_hot_or_logit == 'logit':
            return torch.softmax(input, dim=-1)
        elif one_hot_or_logit == 'gumbel-logit':
            return F.gumbel_softmax(input, dim=-1, hard=False)
        else:
            raise NotImplementedError()

    def compute_tokenizer_encodings(self, attacked_text: AttackedText) -> BatchEncoding:
        word_or_token, _ = self.input_type
        if word_or_token == 'token':
            if 'premise' in attacked_text._text_input:
                return self.tokenizer(attacked_text._text_input['premise'],
                                      attacked_text._text_input['hypothesis'], return_tensors='pt',
                                      max_length=512,
                                      truncation=True)
            else:
                return self.tokenizer([attacked_text.text], return_tensors='pt', truncation=True,
                                      max_length=512)
        else:
            raise NotImplementedError()

    def get_tokenizer_original_encodings(self, attacked_text: AttackedText) -> BatchEncoding:
        if self._encodings is not None:
            return self._encodings.to(shared.utils.device)
        else:
            self._encodings = self.compute_tokenizer_encodings(attacked_text)
            return self._encodings.to(shared.utils.device)

    def compute_grad(self, attacked_text: AttackedText, input: Tensor, stochastic_mask: Tensor,
                     quantized_mask: Tensor) -> Tuple[Tensor, Tensor]:
        if self.compute_gradient_method == 'zoo':
            with torch.no_grad():
                score_original = self.compute_scores(attacked_text, input.unsqueeze(0)).squeeze(0)
            compute_lambda_fn = partial(self._compute_gradient_zoo, score_original=score_original)
        elif self.compute_gradient_method == 'white-box':
            compute_lambda_fn = self._compute_gradient_white_box
        else:
            raise NotImplementedError()
        grads_and_scores = [compute_lambda_fn(attacked_text, input, stochastic_mask, quantized_mask) for _ in
                            range(self.fake_batch_multiplier)]
        grads, scores = zip(*grads_and_scores)
        return torch.mean(torch.stack(grads), dim=0), torch.mean(torch.tensor(scores)).cpu().item()

    def _compute_gradient_zoo(self, attacked_text: AttackedText, input: Tensor, stochastic_mask: Tensor,
                              quantized_mask: Tensor, reduce: str = 'mean', score_original: Optional[Tensor] = None) -> \
            Tuple[Tensor, Tensor]:

        with torch.no_grad():
            batched_inputs = input.repeat(self.batch_size, 1, 1)
            if self.noise_sampling_method == 'normal':
                perturbation = torch.randn(size=batched_inputs.shape).to(shared.utils.device)
            elif self.noise_sampling_method == 'uniform':
                perturbation = (torch.rand(size=batched_inputs.shape).to(shared.utils.device) - 0.5) * 2
            else:
                raise NotImplementedError()

            perturbation[:, ~stochastic_mask] = 0.0
            perturbed_inputs = batched_inputs + (self.mi * perturbation)
            score_perturbed = self.compute_scores(attacked_text, perturbed_inputs)
            grads = ((score_perturbed - score_original) / self.mi).view(-1, 1, 1) * perturbed_inputs
            if reduce == 'mean':
                grad = torch.mean(grads, dim=0)
                grad[~stochastic_mask] = 0.0
            elif reduce == 'none':
                grad = grads
                grad[:, ~stochastic_mask] = 0.0
            else:
                raise NotImplementedError
            return grad, score_original.cpu().item()

    def compute_scores(self, attacked_text: AttackedText, inputs: Tensor, is_one_hot: bool = False) -> Tensor:
        original_encodings = self.get_tokenizer_original_encodings(attacked_text)
        one_hot_encodings = self.input_to_bert_one_hot(inputs) if not is_one_hot else inputs
        score = self.goal_function.get_score_from_one_hots(one_hot_encodings, original_encodings, attacked_text.text)
        return score

    def _compute_gradient_white_box(self, attacked_text: AttackedText, input: Tensor, stochastic_mask: Tensor,
                                    quantized_mask: Tensor) -> Tuple[Tensor, Tensor]:
        self.goal_function.zero_grad()
        if not input.requires_grad:
            input.requires_grad = True
        input.grad = None
        batched_inputs = input.repeat(self.batch_size, 1, 1)
        scores = self.compute_scores(attacked_text, batched_inputs)
        scores = torch.mean(scores)
        scores.backward()
        grad = input.grad
        grad[~stochastic_mask] = 0.0
        self.goal_function.zero_grad()
        input.grad = None
        return grad, scores.detach().cpu().item()

    def quantize_input(self, attacked_text: AttackedText, input: Tensor, stochastic_mask: Tensor,
                       quantized_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        quantization_method, _ = self.quantization_method
        if quantization_method == 'naive':
            return self._quantize_input_naive(input, stochastic_mask, quantized_mask)
        elif quantization_method == 'sample':
            return self._quantize_input_sample(input, stochastic_mask, quantized_mask)
        elif quantization_method == 'mango':
            return self._quantize_input_with_selection(attacked_text, input, stochastic_mask, quantized_mask)

        else:
            raise NotImplementedError()

    @torch.no_grad()
    def _quantize_input_sample(self, input: Tensor,
                               stochastic_mask: Tensor,
                               quantized_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _, quantization_arguments = self.quantization_method
        sample_method = quantization_arguments['method']
        max_num_samples = quantization_arguments['max_num_samples']
        if sample_method == 'gumbel-softmax':
            sample_fn = lambda x: F.gumbel_softmax(x, hard=True, dim=-1)
        else:
            raise NotImplementedError()

        for _ in range(max_num_samples):
            sampled_one_hot = sample_fn(input)
            sampled_text = self.retrieve_text_from_one_hot(sampled_one_hot)
            result = self.get_goal_result(sampled_text)
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                break

        quantized_mask[sampled_one_hot == 1.0] = True
        stochastic_mask[:] = False
        return sampled_one_hot, stochastic_mask, quantized_mask

    def _quantize_input_with_selection(self, attacked_text: AttackedText, input: Tensor, stochastic_mask: Tensor,
                                       quantized_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        position_idx = self._select_index_for_quantization(input, stochastic_mask, quantized_mask)
        indices = self._propose_quantizations(position_idx, attacked_text, input, stochastic_mask, quantized_mask)
        best_token_idx = self._find_best_quantization(position_idx, indices, attacked_text, input, stochastic_mask,
                                                      quantized_mask)
        return self._quantize_indices([(position_idx, best_token_idx)], input, stochastic_mask, quantized_mask)

    def _select_index_for_quantization(self, input: Tensor, stochastic_mask: Tensor, quantized_mask: Tensor) -> int:
        with torch.no_grad():
            _, quantization_arguments = self.quantization_method
            selection_direction = quantization_arguments['direction']
            selection_value_fn = quantization_arguments['function']
            one_hot = self.input_to_one_hot(input).cpu()
            if selection_value_fn == 'entropy':
                value_fn = lambda x: torch.sum(-torch.log(x) * x)
            else:
                raise NotImplementedError()

            values = []
            for x, x_stochastic_mask, x_quantized_mask in zip(one_hot, stochastic_mask, quantized_mask):
                if torch.sum(x_stochastic_mask) == 0:
                    values.append(float('nan'))
                else:
                    value = value_fn(x[x_stochastic_mask])
                    if value.isnan():
                        value = 0.0
                    values.append(value)

            values = torch.tensor(values)
            if values[~values.isnan()].numel() == 0:
                print(torch.sum(stochastic_mask))
            self.logger.add_scalars(f'sample_{str(self.samples_count)}',
                                    {'entropy_mean': torch.mean(values[~values.isnan()]),
                                     'entropy_max': torch.max(values[~values.isnan()]),
                                     'entropy_min': torch.min(values[~values.isnan()])}, self.tokens_count)

            values = values if selection_direction == 'max' else -values
            values[values.isnan()] = -float('inf')
            idx = torch.argmax(values).item()

            return idx

    def _propose_quantizations(self, position_idx: int, attacked_text: AttackedText, input: Tensor,
                               stochastic_mask: Tensor,
                               quantized_mask: Tensor) -> List[int]:
        _, quantization_arguments = self.quantization_method
        quantization_method = quantization_arguments['method']
        one_hot = self.input_to_one_hot(input).detach()
        if quantization_method == 'one_hot':
            scores = one_hot[position_idx]
        elif quantization_method == 'gradient_cos':
            scores = self._compute_gradient_cos_scores(position_idx, attacked_text, input, stochastic_mask,
                                                       quantized_mask)
        elif quantization_method == 'mixed':
            one_hot_lambda = quantization_arguments[
                'one_hot_lambda'] if 'one_hot_lambda' in quantization_arguments else 0.5
            one_hot_scores = one_hot[position_idx]
            if one_hot_lambda < 1.0:
                if self.check_for_ones and torch.sum(
                        torch.isclose(one_hot_scores, torch.ones_like(one_hot_scores), atol=1e-4, rtol=1e-4)) == 1:
                    gradient_cos_scores = one_hot_scores
                else:
                    gradient_cos_scores = self._compute_gradient_cos_scores(position_idx, attacked_text, input,
                                                                            stochastic_mask,
                                                                            quantized_mask)
                    gradient_cos_scores = torch.clamp(gradient_cos_scores, min=-1, max=1)
                    gradient_cos_scores[gradient_cos_scores.isnan()] = 0.0
            else:
                gradient_cos_scores = 0
            scores = one_hot_lambda * one_hot_scores + (1 - one_hot_lambda) * gradient_cos_scores
            scores = (scores + 1 - one_hot_lambda) / (2 - one_hot_lambda)  # to be in [0, 1]
        else:
            raise NotImplementedError()

        scores[~stochastic_mask[position_idx]] = -float('inf')
        # scores[scores.isnan()] = -float('inf')
        indices = torch.topk(scores, self.max_quantization_proposals).indices
        if self.quantization_proposals_threshold is None:
            return list(indices)

        best_scores = scores[indices]
        diffs = best_scores[0] - best_scores
        good_indices = diffs < self.quantization_proposals_threshold
        return list(indices[good_indices])

    def _compute_gradient_cos_scores(self, position_idx: int, attacked_text: AttackedText, input: Tensor,
                                     stochastic_mask: Tensor, quantized_mask: Tensor) -> Tensor:
        one_hot_input = self.input_to_one_hot(input)
        original_input_type = self.input_type
        self.input_type = (original_input_type[0], 'one-hot')  # to treat input as one-hot
        if self.compute_gradient_method == 'white-box':
            one_hot_input.retain_grad()
        grad, _ = self.compute_grad(attacked_text, one_hot_input, stochastic_mask, quantized_mask)
        self.input_type = original_input_type

        one_hot_input = one_hot_input.detach()[position_idx]
        grad = grad[position_idx]
        scores = compute_gradient_cosine_similarity_fast(one_hot_input, grad)
        scores[~stochastic_mask[position_idx]] = -float('inf')
        return torch.tensor(scores)

    def _find_best_quantization(self, position_idx: int, indices: List[int], attacked_text: AttackedText, input: Tensor,
                                stochastic_mask: Tensor, quantized_mask: Tensor) -> int:
        if len(indices) == 1:
            return indices[0]
        with torch.no_grad():
            inputs = [self._quantize_indices([(position_idx, idx)], input.clone(), stochastic_mask.clone(),
                                             quantized_mask.clone())[0] for idx in indices]
            inputs = torch.stack(inputs)
            scores = self.compute_scores(attacked_text, inputs)

            list_idx = torch.argmax(scores)
            self.logger.add_scalars(f'quantization_score_sample_{self.samples_count}', {'score': scores[list_idx]},
                                    self.tokens_count)
            return indices[list_idx]

    def _quantize_input_naive(self, input: Tensor, stochastic_mask: Tensor, quantized_mask: Tensor) -> Tuple[
        Tensor, Tensor, Tensor]:
        token_indices = torch.argmax(input, dim=1)
        indices = [(pos_idx, token_idx.item()) for pos_idx, token_idx in enumerate(token_indices)]
        return self._quantize_indices(indices, input, stochastic_mask, quantized_mask)

    def _quantize_indices(self, indices: List[Tuple[int, int]], input: Tensor, stochastic_mask: Tensor,
                          quantized_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _, one_hot_or_logit = self.input_type
        fill_value = -float('inf') if one_hot_or_logit == 'logit' else 0.0
        for position_idx, token_idx in indices:
            input.data[position_idx, :] = fill_value
            input.data[position_idx, token_idx] = 1.0
            quantized_mask[position_idx, token_idx] = True
            stochastic_mask[position_idx, :] = False
        return input, stochastic_mask, quantized_mask

    @torch.no_grad()
    def retrieve_text_from_one_hot(self, one_hot: Tensor) -> AttackedText:
        word_or_token, _ = self.input_type
        if word_or_token == 'token':
            ids = list(torch.argmax(one_hot, dim=1))
            ids = ids[1:-1]
            groups = [list(items) for key, items in groupby(ids, lambda x: x == self.tokenizer.sep_token_id) if not key]
            if len(groups) == 1:
                text = self.tokenizer.decode(ids)
                return AttackedText(text)
            elif len(groups) == 2:
                premise, hypothesis = groups
                premise = self.tokenizer.decode(premise)
                hypothesis = self.tokenizer.decode(hypothesis)
                text = OrderedDict({'premise': premise, 'hypothesis': hypothesis})
                return AttackedText(text)
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    @torch.no_grad()
    def retrieve_text_after_quantization(self, quantized_input: Tensor) -> AttackedText:
        word_or_token, _ = self.input_type
        quantization_method, _ = self.quantization_method
        if word_or_token == 'token':
            if quantization_method == 'sample':
                return self.retrieve_text_from_one_hot(quantized_input)
            else:
                one_hot = self.input_to_bert_one_hot(quantized_input)
                return self.retrieve_text_from_one_hot(one_hot)
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def get_goal_result(self, updated_text: AttackedText) -> GoalFunctionResult:
        result, _ = self.get_goal_results([updated_text])
        return result[0]

    def get_gradients(self, initial_result: GoalFunctionResult, step_points: List[int], arguments: Dict[str, Any],
                      repeats: int = 100) -> Dict:
        attacked_text: AttackedText = initial_result.attacked_text
        self.prepare(attacked_text)

        stochastic_mask, quantized_mask, original_mask = self.get_input_masks(attacked_text)
        input = self.initialize_input(stochastic_mask, quantized_mask, original_mask)

        stochastic_mask = stochastic_mask.to(shared.utils.device)
        quantized_mask = quantized_mask.to(shared.utils.device)
        input = input.to(shared.utils.device)

        keys, values = zip(*arguments.items())
        argument_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

        self.optimizer.reset()
        steps = 0
        results_dict = {}
        for step_point in step_points:
            self.batch_size = 1
            if isinstance(step_point, int):
                for _ in range(step_point - steps):
                    grad, score = self.compute_grad(attacked_text, input, stochastic_mask, quantized_mask)
                    input.data = self.optimizer.updated(input.data, grad, stochastic_mask, quantized_mask)
            else:
                input.data.normal_(*step_point)

            original_grad, _ = self._compute_gradient_white_box(attacked_text, input, stochastic_mask, quantized_mask)
            original_grad = original_grad.squeeze(0)
            original_grad[~stochastic_mask] = float('nan')
            results_dict[step_point] = {}
            original_grad = original_grad.cpu()
            results_dict[step_point]['original_grad'] = original_grad
            self.batch_size = 10
            for i, special_dict in enumerate(argument_dict_list):
                for key, value in special_dict.items():
                    setattr(self, key, value)
                gradients = [
                    self._compute_gradient_zoo(attacked_text, input, stochastic_mask, quantized_mask, reduce='none')[
                        0].cpu()
                    for _ in range(repeats // self.batch_size)]
                gradients = torch.cat(gradients)
                gradients[:, ~stochastic_mask] = float('nan')

                gradients_cum_sum = torch.cumsum(gradients, 0)
                norm = torch.arange(gradients.shape[0], device=gradients.device).view(-1, 1, 1) + 1
                gradients_cum_mean = gradients_cum_sum / norm

                cosines = []
                norms = []
                means = []
                for j in range(gradients_cum_mean.shape[0]):
                    cosine = torch.cosine_similarity(original_grad[stochastic_mask],
                                                     gradients_cum_mean[j, stochastic_mask], dim=0)
                    norm = gradients_cum_mean[j, stochastic_mask].norm()
                    mean = gradients_cum_mean[j, stochastic_mask].mean()
                    cosines.append(cosine)
                    norms.append(norm)
                    means.append(mean)
                cosines = torch.stack(cosines)
                norms = torch.stack(norms)
                means = torch.stack(means)
                results_dict[step_point][f'combination_{i}'] = {
                    'arguments': special_dict,
                    # 'gradients': gradients.cpu(),
                    'cosines': cosines.cpu(),
                    'norms': norms.cpu(),
                    'means': means.cpu()
                }

        return results_dict

    def perform_search(self, initial_result: GoalFunctionResult) -> GoalFunctionResult:
        attacked_text: AttackedText = initial_result.attacked_text
        self.prepare(attacked_text)

        stochastic_mask, quantized_mask, original_mask = self.get_input_masks(attacked_text)
        input = self.initialize_input(stochastic_mask, quantized_mask, original_mask)
        n_tokens_to_optimize = (quantized_mask.shape[0] - torch.sum(quantized_mask)).cpu().item()
        step_scheduler = self.step_scheduler_cls(init_n_steps=self.optimization_steps,
                                                 init_n_tokens_to_optimize=n_tokens_to_optimize,
                                                 **self.step_scheduler_kwargs)

        stochastic_mask = stochastic_mask.to(shared.utils.device)
        quantized_mask = quantized_mask.to(shared.utils.device)
        input = input.to(shared.utils.device)

        self.optimizer.reset()
        while n_tokens_to_optimize > 0:
            steps = step_scheduler.get_n_steps(n_tokens_to_optimize)
            if self.reset_optimizer_every_quantization:
                self.optimizer.reset()
            for i in range(steps):
                grad, score = self.compute_grad(attacked_text, input, stochastic_mask, quantized_mask)
                if self.dump_score:
                    self.dumped_score.append(score)
                self.logger.add_scalars(f'steps_gradient_sample_{self.samples_count}',
                                        {'grad_mean': torch.abs(grad[stochastic_mask]).mean()}, self.steps_count)
                self.logger.add_scalars(f'steps_score_sample_{self.samples_count}', {'score': score}, self.steps_count)
                input.data = self.optimizer.updated(input.data, grad, stochastic_mask, quantized_mask)
                self.steps_count += 1
            if steps == 0:
                self.steps_count += 1
            else:
                self.logger.add_scalars(f'tokens_gradient_sample_{self.samples_count}',
                                        {'grad_mean': torch.abs(grad[stochastic_mask]).mean()}, self.tokens_count)
                self.logger.add_scalars(f'tokens_score_sample_{self.samples_count}', {'score': score},
                                        self.tokens_count)
            if steps == 0 and self.dump_score:
                score = self.compute_scores(attacked_text, input.unsqueeze(0))
                self.dumped_score.append(score)
            input, stochastic_mask, quantized_mask = \
                self.quantize_input(attacked_text, input, stochastic_mask, quantized_mask)
            n_tokens_to_optimize = (quantized_mask.shape[0] - torch.sum(quantized_mask)).cpu().item()
            self.tokens_count += 1

        if self.dump_score:
            with torch.no_grad():
                is_one_hot = self.quantization_method[0] == 'sample'
                score = self.compute_scores(attacked_text, input.unsqueeze(0), is_one_hot=is_one_hot)
            self.dumped_score.append(score)
            torch.save(self.dumped_score, os.path.join(self.logging_path, f'scores_sample_{self.samples_count}.pt'))
        self.optimizer.reset()
        updated_text = self.retrieve_text_after_quantization(input)
        result = self.get_goal_result(updated_text)
        return result

    @property
    def is_black_box(self):
        """Returns `True` if search method does not require access to victim
            model's internal states."""
        return self.compute_gradient_method == 'zoo'
