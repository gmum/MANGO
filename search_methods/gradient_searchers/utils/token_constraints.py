import abc
from typing import Optional, List

import nltk
import torch
from textattack import shared
from textattack.shared import WordEmbedding
from torch import Tensor
from transformers import BertTokenizer


class TokenConstraintBase(abc.ABC):
    def __init__(self, tokenizer: Optional[BertTokenizer]):
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def get_allowed_mask(self, token_id: int) -> Tensor:
        ...


class ComposedTokenConstraint(TokenConstraintBase):
    def __init__(self, constraints: List[TokenConstraintBase]):
        super().__init__(tokenizer=None)
        self.constraints = constraints

    def get_allowed_mask(self, token_id: int) -> Tensor:
        masks = torch.stack([c.get_allowed_mask(token_id) for c in self.constraints])
        mask = torch.prod(masks, dim=0).bool()
        return mask


class SpecialTokenTokenConstraint(TokenConstraintBase):
    def __init__(self, tokenizer: Optional[BertTokenizer]):
        super().__init__(tokenizer)
        self.mask = torch.ones(self.tokenizer.vocab_size).bool()
        special_tokens_ids = [v for k, v in tokenizer.vocab.items() if '[' in k]
        special_tokens_ids = [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]
        self.mask[special_tokens_ids] = False

    def get_allowed_mask(self, token_id: int) -> Tensor:
        return self.mask


class StopwordTokenConstraint(TokenConstraintBase):
    def __init__(self, tokenizer: Optional[BertTokenizer]):
        super().__init__(tokenizer)
        self.mask = torch.ones(self.tokenizer.vocab_size).bool()
        stopwords = set(nltk.corpus.stopwords.words("english"))
        stopwords_ids = [self.tokenizer.vocab[s] for s in stopwords if s in self.tokenizer.vocab]
        self.mask[stopwords_ids] = False

    def get_allowed_mask(self, token_id: int) -> bool:
        return self.mask


class EmbeddingSimilarityTokenConstraint(TokenConstraintBase):
    def __init__(self, tokenizer: Optional[BertTokenizer], min_cos_sim: float = 0.5,
                 embedding_path='glove_embeddings.pt', all_token_allowed_for_non_existent: bool = False):
        super().__init__(tokenizer)
        self.min_cos_sim = min_cos_sim
        embeddings = torch.load(embedding_path).to(shared.utils.device)
        embeddings = embeddings / embeddings.norm(dim=-1).view(-1, 1)
        embeddings[embeddings.isnan()] = 0.0
        self.embeddings = embeddings
        self.all_allowed = all_token_allowed_for_non_existent

    def get_allowed_mask(self, token_id: int) -> Tensor:
        mask = torch.zeros(self.tokenizer.vocab_size).bool()
        mask[token_id] = True
        token_embedding = self.embeddings[token_id]
        if torch.sum(token_embedding) == 0:
            if self.all_allowed:
                mask[:] = True
            return mask
        similarities = self.embeddings @ token_embedding
        similar_ids = torch.where(similarities >= self.min_cos_sim)[0].cpu()
        mask[similar_ids] = True
        return mask
