from typing import Any, List, Optional

import torch
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.utils import infer_torch_device
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..pipeline.ingestion import get_node_content

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512


class SentenceTransformerRerank(BaseNodePostprocessor):
    model: str = Field(description="Sentence transformer model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cpu",
        description="Device to use for sentence transformer.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    _model: Any = PrivateAttr()

    def __init__(
            self,
            top_n: int = 2,
            model: str = "cross-encoder/stsb-distilroberta-base",
            device: Optional[str] = None,
            keep_retrieval_score: Optional[bool] = False,
    ):
        try:
            from sentence_transformers import CrossEncoder  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers or torch package,",
                "please `pip install torch sentence-transformers`",
            )
        device = infer_torch_device() if device is None else device
        self._model = CrossEncoder(
            model, max_length=DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH, device=device, trust_remote_code=True,
        )
        super().__init__(
            top_n=top_n,
            model=model,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerRerank"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.NONE),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
                CBEventType.RERANKING,
                payload={
                    EventPayload.NODES: nodes,
                    EventPayload.MODEL_NAME: self.model,
                    EventPayload.QUERY_STR: query_bundle.query_str,
                    EventPayload.TOP_K: self.top_n,
                },
        ) as event:
            scores = self._model.predict(query_and_nodes)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                        : self.top_n
                        ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes


class LLMRerank(BaseNodePostprocessor):
    model: str = Field(description="Transformer model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cpu",
        description="Device to use for sentence transformer.",
    )
    keep_retrieval_score: bool = Field(
        default=True,
        description="Whether to keep the retrieval score in metadata.",
    )
    embed_bs: int = Field(
        default=64,
    )
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _yes_loc: Any = PrivateAttr()
    _layer: int = PrivateAttr()
    _embed_bs: int = PrivateAttr()
    _embed_type: int = PrivateAttr()
    _type: int = PrivateAttr()
    _compress_ratio: int = PrivateAttr()
    _compress_layer: list[int] = PrivateAttr()
    _use_efficient: int = PrivateAttr()

    def __init__(
            self,
            top_n: int = 2,
            model: str = "BAAI/bge-reranker-v2-minicpm-layerwise",
            device: Optional[str] = None,
            keep_retrieval_score: Optional[bool] = True,
            embed_bs: int = 64,
            embed_type: int = 0,
            use_efficient: int = 0
    ):
        device = infer_torch_device() if device is None else device

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._yes_loc = self._tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        self._embed_type = embed_type
        if "bge-reranker-v2-minicpm-layerwise" in model:
            self._use_efficient = use_efficient
            if self._use_efficient != 0:
                from ..utils.efficient_modeling_minicpm_reranker import LayerWiseMiniCPMForCausalLM
                self._model = LayerWiseMiniCPMForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to(device)
                self._model.efficient_type = self._use_efficient
                self._model.efficient_t = 0.4
                self._model.efficient_layers = [12]
            else:
                from ..utils.modeling_minicpm_reranker import LayerWiseMiniCPMForCausalLM
                self._model = LayerWiseMiniCPMForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to(device)
            self._model.eval()
            self._layer = 28  # 8-40
            self._type = 1
        elif "bge-reranker-v2.5-gemma2-lightweight" in model:
            from ..utils.gemma_model import CostWiseGemmaForCausalLM
            self._tokenizer.padding_side = 'right'
            self._model = CostWiseGemmaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
            self._model.eval()
            self._type = 2
            self._compress_layer = [24, 40]  # [8, 16, 24, 32, 40]
            self._compress_ratio = 2  # 1 2 4 8
            self._layer = 28  # 8-42
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
            self._model.eval()
            self._type = 0
        self._embed_bs = embed_bs
        super().__init__(
            top_n=top_n,
            model=model,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
        )

    def last_logit_pool(self, logits: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return logits[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = logits.shape[0]
            return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)

    def get_inputs_v2_5(self, pairs, tokenizer, prompt=None, max_length=1024):
        if prompt is None:
            prompt = "Predict whether passage B contains an answer to query A."
        sep = "\n"
        prompt_inputs = tokenizer(prompt,
                                  return_tensors=None,
                                  add_special_tokens=False)['input_ids']
        sep_inputs = tokenizer(sep,
                               return_tensors=None,
                               add_special_tokens=False)['input_ids']
        inputs = []
        query_lengths = []
        prompt_lengths = []
        for query, passage in pairs:
            query_inputs = tokenizer(f'A: {query}',
                                     return_tensors=None,
                                     add_special_tokens=False,
                                     max_length=max_length * 3 // 4,
                                     truncation=True)
            passage_inputs = tokenizer(f'B: {passage}',
                                       return_tensors=None,
                                       add_special_tokens=False,
                                       max_length=max_length,
                                       truncation=True)
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
            query_lengths.append(len([tokenizer.bos_token_id] + query_inputs['input_ids'] + sep_inputs))
            prompt_lengths.append(len(sep_inputs + prompt_inputs))

        return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        ), query_lengths, prompt_lengths

    def get_inputs(self, pairs, tokenizer, prompt=None, max_length=1024):
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = tokenizer(prompt,
                                  return_tensors=None,
                                  add_special_tokens=False)['input_ids']
        sep_inputs = tokenizer(sep,
                               return_tensors=None,
                               add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            query_inputs = tokenizer(f'A: {query}',
                                     return_tensors=None,
                                     add_special_tokens=False,
                                     max_length=max_length * 3 // 4,
                                     truncation=True)
            passage_inputs = tokenizer(f'B: {passage}',
                                       return_tensors=None,
                                       add_special_tokens=False,
                                       max_length=max_length,
                                       truncation=True)
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

    @classmethod
    def class_name(cls) -> str:
        return "LLMRerank"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []
        bsz = self._embed_bs
        N = len(nodes)

        for i in range(0, N, bsz):
            if self._type == 1 and i == 0 and self._use_efficient != 0:
                self._model.judge = True
                self._model.cut_layer = self._layer
            begin_idx, end_idx = i, min(i + bsz, N)
            cur_nodes = nodes[begin_idx:end_idx]
            query_and_nodes = [
                (
                    query_bundle.query_str,
                    get_node_content(node.node, self._embed_type),
                )
                for node in cur_nodes
            ]

            with self.callback_manager.event(
                    CBEventType.RERANKING,
                    payload={
                        EventPayload.NODES: cur_nodes,
                        EventPayload.MODEL_NAME: self.model,
                        EventPayload.QUERY_STR: query_bundle.query_str,
                        EventPayload.TOP_K: self.top_n,
                    },
            ) as event:
                if self._type == 2:
                    inputs, query_lengths, prompt_lengths = self.get_inputs_v2_5(query_and_nodes, self._tokenizer)
                    inputs = inputs.to(self._model.device)
                else:
                    inputs = self.get_inputs(query_and_nodes, self._tokenizer).to(self._model.device)

                with torch.no_grad():
                    if self._type == 1:
                        all_scores = self._model(**inputs, return_dict=True, cutoff_layers=[self._layer])
                        scores = [scores[:, -1].view(-1, ).float() for scores in all_scores[0]][0]
                        if i == 0 and self._use_efficient == 1:
                            self._layer = self._model.cut_layer
                            self._model.judge = False

                    elif self._type == 2:
                        outputs = self._model(**inputs,
                                              return_dict=True,
                                              cutoff_layers=[self._layer],
                                              compress_ratio=self._compress_ratio,
                                              compress_layer=self._compress_layer,
                                              query_lengths=query_lengths,
                                              prompt_lengths=prompt_lengths)
                        scores = []
                        for i in range(len(outputs.logits)):
                            logits = self.last_logit_pool(outputs.logits[i], outputs.attention_masks[i])
                            scores.append(logits.cpu().float().tolist())
                        scores = scores[0]
                    else:
                        scores = self._model(**inputs, return_dict=True).logits[:, -1, self._yes_loc].view(-1, ).float()
                assert len(scores) == len(cur_nodes)

                for node, score in zip(cur_nodes, scores):
                    if self.keep_retrieval_score:
                        # keep the retrieval score in metadata
                        node.node.metadata["retrieval_score"] = node.score
                    node.score = score
            nodes[begin_idx:end_idx] = cur_nodes

        new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                    : self.top_n
                    ]
        event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
