"""
Module lõi: dense_search, bm25_search, rrf_fusion.
Import module này từ query_cli.py hoặc dùng độc lập trong notebook.
"""

import pickle
from typing import List, Tuple

import numpy as np
import torch

import config
import utils


class DenseSearcher:
    """Nhánh Dense: encode query bằng SigLIP2 text tower, search FAISS."""

    def __init__(self):
        import faiss
        from transformers import AutoModel, AutoProcessor

        print("[DenseSearcher] Loading FAISS index...")
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)

        print("[DenseSearcher] Loading id map...")
        self.id_map = list(utils.read_jsonl(config.ID_MAP_PATH))  # int_id == vị trí trong FAISS

        print("[DenseSearcher] Loading SigLIP2...")
        dtype = getattr(torch, config.DTYPE)
        self.model = AutoModel.from_pretrained(config.SIGLIP2_MODEL_ID, torch_dtype=dtype).to(config.DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(config.SIGLIP2_MODEL_ID)

    @torch.no_grad()
   def encode_query(self, query_en: str) -> np.ndarray:
    inputs = self.processor(text=[query_en], return_tensors="pt", padding=True).to(self.device)
    inputs = {k: v.to(getattr(torch, config.DTYPE)) if v.dtype == torch.float32 else v 
              for k, v in inputs.items()}
    
    # Lấy output từ model
    outputs = self.model.get_text_features(**inputs)
    
    # Trích xuất PyTorch Tensor thực sự từ Output Object
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        feats = outputs.text_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feats = outputs.pooler_output
    else:
        feats = outputs

    # Normalize L2
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()

    
    def search(self, query_en: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Trả về list (int_id, score) đã sort giảm dần theo cosine similarity."""
        top_k = top_k or config.DENSE_TOPK
        q_emb = self.encode_query(query_en)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results


class BM25Searcher:
    """Nhánh sparse: BM25 trên corpus caption+object+metadata."""

    def __init__(self):
        print("[BM25Searcher] Loading BM25 index...")
        with open(config.BM25_PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.int_ids_order = data["int_ids_order"]  # map vị trí BM25 -> int_id thật

    def search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        top_k = top_k or config.BM25_TOPK
        tokens = utils.tokenize_vi(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.int_ids_order[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def rrf_fusion(rank_lists: List[List[int]], k: int = None) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    rank_lists: list các list int_id đã sort giảm dần theo độ liên quan
                (ví dụ [dense_ids, bm25_ids]).
    Trả về list (int_id, rrf_score) sort giảm dần.
    """
    k = k or config.RRF_K
    scores = {}
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query_vi: str, query_en: str, dense: DenseSearcher, bm25: BM25Searcher,
                   top_k: int = None) -> List[Tuple[int, float]]:
    """
    Pipeline đầy đủ 1 query: search 2 nhánh -> RRF fusion.
    query_vi: dùng cho BM25 (đã tokenize tiếng Việt).
    query_en: dùng cho SigLIP2 (nên dịch VI->EN trước khi gọi, xem translate.py).
    """
    top_k = top_k or config.FINAL_TOPK

    dense_results = dense.search(query_en, top_k=config.DENSE_TOPK)
    bm25_results = bm25.search(query_vi, top_k=config.BM25_TOPK)

    dense_ids = [i for i, _ in dense_results]
    bm25_ids = [i for i, _ in bm25_results]

    fused = rrf_fusion([dense_ids, bm25_ids])
    return fused[:top_k]
