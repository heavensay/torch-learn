"""本地生成 MiniLM 稠密向量和 BM25 稀疏向量，供 Milvus Lite 混合搜索示例使用。

这里没有调用 Milvus 内置 BM25 Function，而是在 Python 进程中计算 BM25 所需的
词项权重，并将结果编码成 Milvus 可检索的稀疏向量。这样可以在不支持内置 BM25
全文搜索的 Milvus Lite 环境中，仍然把 BM25 作为一条独立的 sparse 检索通道。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


# MiniLM 是负责语义检索的 dense 模型。该模型输出 384 维向量。
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# BM25 参数：k1 控制词频增加时的收益饱和速度；b 控制文档长度归一化程度。
# 这里使用常见的默认值附近设置，后续可根据语料和检索效果进行调优。
BM25_K1 = 1.5
BM25_B = 0.75

# 插入数据时会把分词器词表、每个词的 IDF、平均文档长度等状态写到这个文件。
# Streamlit 查询时必须加载同一份状态，确保查询词被映射到与入库文本相同的维度。
BM25_STATE_PATH = Path("./milvus-data/bm25_state.pkl")


class HybridEmbedder:
    """为同一批文本生成 dense 与 BM25 sparse 两种互补向量。

    Dense 向量由 MiniLM 生成，用于匹配语义相近但措辞不同的文本。

    Sparse 向量由本类在客户端按 BM25 公式构造。为了让 Milvus 的 sparse 内积
    得分等价于 BM25 的核心计算，我们将文档侧和查询侧的权重分开存放：

    - 文档侧每个词的权重包含词频饱和与文档长度归一化部分；
    - 查询侧每个词的权重包含该词在语料中的 IDF；
    - 两者按相同的词项编号做内积，得到各个查询词贡献的分数之和。

    当前实现不额外对查询词频做 BM25 饱和加权；通常搜索查询较短，这样实现简单，
    也保留了 BM25 的主要排序因素。
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        # SentenceTransformer 会加载模型权重；第一次运行时若本地没有缓存，
        # 会从模型仓库下载。device="cpu" 适用于无需配置 GPU 的本地演示。
        self.model = SentenceTransformer(model_name, device=device)

        # 以下属性在 fit_documents() 从文档语料中学习，或由 load_bm25_state() 恢复。
        self.vectorizer: CountVectorizer | None = None
        self.idf: np.ndarray | None = None
        self.avgdl = 0.0
        self.doc_count = 0

    @property
    def dim(self) -> int:
        """返回 MiniLM dense 向量维度，用于创建 Milvus FLOAT_VECTOR 字段。"""
        return int(self.model.get_sentence_embedding_dimension())

    def fit_documents(self, texts: list[str]) -> list[dict[int, float]]:
        """从整份文档语料计算词表、IDF 和每篇文档的 BM25 sparse 向量。

        返回值是一组稀疏字典：字典键为词项在语料词表中的整数编号，值为文档侧
        BM25 权重。Milvus 的 SPARSE_FLOAT_VECTOR 可以直接接收这种 {维度: 权重} 结构。
        """
        # CountVectorizer 默认会将英文转小写并按词语切分；文档、查询都必须使用
        # 同一个 vectorizer，否则相同单词可能被映射到不同 sparse 维度。
        self.vectorizer = CountVectorizer(dtype=np.float32)
        counts = self.vectorizer.fit_transform(texts).tocsr()

        # 统计语料规模、每篇文档的词数，以及语料平均文档长度（avgdl）。
        self.doc_count = counts.shape[0]
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
        self.avgdl = float(doc_lengths.mean()) if self.doc_count else 0.0

        # df 表示包含某个词的文档数量。使用 BM25 常见的平滑 IDF 公式，
        # 让在较少文档中出现的词权重更高，常见词的权重相对较低。
        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

        vectors: list[dict[int, float]] = []
        for row_index in range(counts.shape[0]):
            row = counts.getrow(row_index)
            length = float(doc_lengths[row_index])

            # BM25 文档侧权重：tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))。
            # 这里将 IDF 留给查询向量，之后通过文档向量与查询向量内积组合起来。
            norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length / self.avgdl) if self.avgdl else BM25_K1
            vector = {
                int(term_id): float(tf * (BM25_K1 + 1.0) / (tf + norm))
                for term_id, tf in zip(row.indices, row.data)
            }

            # CountVectorizer 对空文本、纯标点或不符合默认词语规则的文本可能不产生词项。
            # Milvus 不接受空 sparse 向量，因此预留词表以外的一个维度作为兜底编号。
            # 兜底只会让这类无有效词项的文档彼此有 sparse 匹配；语义通道仍由 MiniLM 负责。
            if not vector:
                vector[len(self.vectorizer.vocabulary_)] = 1.0
            vectors.append(vector)
        return vectors

    def encode_queries(self, texts: list[str]) -> dict[str, object]:
        """将查询文本编码成 Milvus hybrid_search 所需的两组查询向量。

        查询 dense 向量使用 MiniLM；查询 sparse 向量使用入库时保存的词表和 IDF。
        返回字典的 dense/sparse 列表都按输入文本顺序排列，支持一次传入多个查询。
        """
        if self.vectorizer is None or self.idf is None:
            raise RuntimeError("Fit/load the BM25 vocabulary before encoding queries")

        # 对查询向量做 L2 归一化，使其与同样归一化的文档向量使用 IP 时等价于余弦相似度。
        dense = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        # transform 使用已学习的词表，不会为查询新建词编号；不在语料词表中的词会被忽略。
        counts = self.vectorizer.transform(texts).tocsr()
        sparse = []
        for row_index in range(counts.shape[0]):
            row = counts.getrow(row_index)

            # 查询侧只放入 IDF，Milvus 与文档侧 BM25 权重做 IP 内积后，
            # 每个共有词的贡献就是 IDF * 文档词频/长度权重。
            sparse.append({
                int(term_id): float(self.idf[term_id])
                for term_id in row.indices
            })

            # 如果查询为空或所有词都不在入库词表中，transform 会得到空向量。
            # 兜底后可避免 Milvus 报“sparse vector must contain at least one weight”。
            if not sparse[-1]:
                sparse[-1][len(self.vectorizer.vocabulary_)] = 1.0
        return {"dense": dense, "sparse": sparse}

    def save_bm25_state(self, path: Path = BM25_STATE_PATH) -> None:
        """保存查询时必须复用的 BM25 语料状态。"""
        if self.vectorizer is None or self.idf is None:
            raise RuntimeError("Fit the BM25 vocabulary before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            # pickle 同时保存 sklearn vectorizer、IDF 数组、avgdl 和文档数量。
            pickle.dump((self.vectorizer, self.idf, self.avgdl, self.doc_count), file)

    def load_bm25_state(self, path: Path = BM25_STATE_PATH) -> None:
        """从插入阶段写出的文件恢复词表和语料统计量，用于搜索查询编码。"""
        with path.open("rb") as file:
            self.vectorizer, self.idf, self.avgdl, self.doc_count = pickle.load(file)
