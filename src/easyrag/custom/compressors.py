import torch
from llmlingua import PromptCompressor
from ..pipeline.rag import cut_sent


class ContextCompressor:
    def __init__(
            self,
            method="bm25_extract",
            rate=0.5,
            bm25_retriever=None,
    ):
        self.rate = rate
        self.method = method
        if 'llmlingua' in method:
            self.prompt_compressor = PromptCompressor(
                "Qwen/Qwen2-7B-Instruct",
                model_config={
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }
            )
        elif "bm25_extract" == method:
            self.bm25_retriever = bm25_retriever

    def compress(
            self,
            query,
            context
    ):
        if self.method == 'bm25_extract':
            # 上下文切割为句子
            pre_len = len(context)
            raw_sentences = cut_sent(context)
            sentences = []
            for raw_sentence in raw_sentences:
                raw_sentence = raw_sentence.strip()
                if raw_sentence != "":
                    sentences.append(raw_sentence)
            # 获取query与每个句子的BM25分数
            scores = self.bm25_retriever.get_scores(query, sentences)
            # 按原句子相对顺序拼接分数高的句子，直到长度超过原长度的rate比例
            sorted_idx = scores.argsort()[::-1]
            i, now_len = 0, 0
            for i, idx in enumerate(sorted_idx):
                now_len += len(sentences[idx])
                if now_len >= pre_len * self.rate:
                    break
            sorted_idx = sorted_idx[:i + 1]
            sorted_idx.sort()
            new_context = ""
            for idx in sorted_idx:
                new_context += sentences[idx]
            return new_context

        else:  # llmlingua
            compressed_obj = self.prompt_compressor.compress_prompt(
                context,
                instruction="",
                question=query,
                rate=self.rate,
                rank_method=self.method,  # llmlingua longllmlingua
            )
            compressed_candidate = compressed_obj['compressed_prompt']
            return compressed_candidate
