import copy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from textattack import shared
from textattack.goal_functions import UntargetedClassification
from torch import Tensor
from transformers import BertForSequenceClassification, BertTokenizer, BatchEncoding

from src.utils import load_gpt2_from_dict


class FluencyBertScoreUntargetedClassification(UntargetedClassification):
    def __init__(self, model, bert_score_lambda: float = 20.0, fluency_score_lambda: float = 1.0, kappa: float = 5.0,
                 use_idf: bool = True):
        super().__init__(model, use_cache=False)
        self.bert_model: BertForSequenceClassification = self.model.model
        self.reference_model = load_gpt2_from_dict("models/transformer_wikitext-103.pth",
                                                   output_hidden_states=True)
        self.reference_model_outputs = {}
        self.tokenizer: BertTokenizer = self.model.tokenizer
        self.bert_model.eval()
        self.reference_model.eval()
        with torch.no_grad():
            tokens = torch.arange(0, self.tokenizer.vocab_size).long().to(shared.utils.device)
            self.bert_model_embeddings_weight = self.bert_model.get_input_embeddings()(tokens)
            self.reference_model_embeddings_weight = self.reference_model.get_input_embeddings()(tokens)
        self.kappa = kappa
        self.bert_score_lambda = bert_score_lambda
        self.fluency_score_lambda = fluency_score_lambda
        self.idf_dict: Optional[Dict[int, Any]] = None
        self.use_idf = use_idf

    def add_idf_dict(self, idf_dict: Dict[int, Any]):
        self.idf_dict = idf_dict

    def zero_grad(self):
        parameters = list(self.bert_model.parameters()) + list(self.reference_model.parameters())
        for p in parameters:
            p.grad = None
            p.required_grad = True

    def _adversarial_goal(self, outputs: Any) -> Tensor:
        logits = outputs['logits']
        logits_at_ground_truth = logits[:, self.ground_truth_output]

        indices = torch.zeros_like(logits.detach()).bool()
        indices[:, self.ground_truth_output] = True
        logits = torch.masked_fill(logits, indices, -float('inf'))

        margin_loss = (logits_at_ground_truth - torch.max(logits, dim=-1).values + self.kappa).clamp(min=0)
        return -margin_loss

    def _bert_score_goal(self, outputs: Any, original_encodings: BatchEncoding, original_text: str) -> Tensor:
        new_output = outputs.hidden_states[-1]
        old_output = self._get_reference_model_output(original_text, original_encodings)

        old_output_norm = old_output / old_output.norm(2, -1).unsqueeze(-1)
        new_output_norm = new_output / new_output.norm(2, -1).unsqueeze(-1)

        if self.idf_dict and self.use_idf:
            idfs = [self.idf_dict[idx.item()] for idx in original_encodings['input_ids'][0]]
            weights = torch.tensor(idfs).float().to(new_output.device)
            weights /= weights.sum()
            old_output_norm *= weights.unsqueeze(0).unsqueeze(-1)
        else:
            old_output_norm /= old_output.size(1)

        cosines = torch.matmul(old_output_norm, new_output_norm.transpose(1, 2))
        cosines = cosines[:, 1:-1, 1:-1]
        similarities = cosines.max(-1)[0].sum(1)
        return similarities

    def _fluency_goal(self, outputs: Any, one_hot_encodings: Tensor) -> Tensor:
        logits = outputs['logits']
        shift_logits = logits[:, :-1, :].contiguous()
        shift_coeffs = one_hot_encodings[:, 1:, :].contiguous()
        shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
        return (shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean(-1)

    def _get_reference_model_output(self, original_text: str, encodings: BatchEncoding) -> Tensor:
        if original_text in self.reference_model_outputs:
            return self.reference_model_outputs[original_text]
        self.reference_model_outputs = {}
        with torch.no_grad():
            output = self.reference_model(encodings['input_ids']).hidden_states[-1]
        self.reference_model_outputs[original_text] = output
        return output

    def get_score_from_one_hots(self, one_hot_encodings: Tensor, original_encodings: BatchEncoding,
                                original_text: str, output_all: bool = False) -> Tensor:
        assert len(one_hot_encodings.shape) == 3
        batch_size = one_hot_encodings.shape[0]
        self.num_queries += batch_size

        batched_encodings = copy.deepcopy(original_encodings)
        del batched_encodings['input_ids']
        batched_encodings = {k: v.repeat(batch_size, 1) for k, v in batched_encodings.items()}
        model_embedding = torch.matmul(one_hot_encodings, self.bert_model_embeddings_weight)
        model_outputs = self.bert_model.forward(inputs_embeds=model_embedding, **batched_encodings)

        reference_model_embedding = torch.matmul(one_hot_encodings, self.reference_model_embeddings_weight)
        reference_model_outputs = self.reference_model(inputs_embeds=reference_model_embedding)

        adversarial_goal = self._adversarial_goal(model_outputs)
        bert_score_goal = self._bert_score_goal(reference_model_outputs, original_encodings, original_text)
        fluency_goal = self._fluency_goal(reference_model_outputs, one_hot_encodings)

        if not output_all:
            return adversarial_goal + self.bert_score_lambda * bert_score_goal + self.fluency_score_lambda * fluency_goal
        else:
            return adversarial_goal, self.bert_score_lambda * bert_score_goal, self.fluency_score_lambda * fluency_goal
