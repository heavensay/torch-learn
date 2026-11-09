"""准备演示语料、生成向量并将数据写入 Milvus Lite。

每条文本会同时存储两种向量：
1. MiniLM 生成的 dense 向量，用于语义相似度搜索；
2. 本地代码计算的 BM25 sparse 向量，用于关键词相关性搜索。

Milvus 的 hybrid_search 会分别执行两条检索，再由 ranker 融合结果。
运行前需安装项目依赖，并确保 MiniLM 模型权重已下载或可从模型仓库取得。
"""

from pathlib import Path

import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from embeddings import HybridEmbedder


# 1. 读取小规模演示语料。
# 文件路径相对于项目目录；TSV 每行包含两个 Quora 问题文本。
file_path = "./milvus-data/quora_duplicate_questions.tsv"
df = pd.read_csv(file_path, sep="\t")

# 原始问题对存在大量重复项。这里按首次出现顺序去重，得到稳定的语料顺序。
# 稳定顺序也让每次拟合和插入时更容易复现相同的 BM25 词表和记录顺序。
questions = []
seen_questions = set()
for _, row in df.iterrows():
    # 使用字典便于明确字段名；question1/question2 分别是问题对的两侧文本。
    obj = row.to_dict()
    # 与 Milvus text 字段 max_length=512 对齐，避免超长文本写入失败。
    for question in (obj["question1"][:512], obj["question2"][:512]):
        if question not in seen_questions:
            seen_questions.add(question)
            questions.append(question)
    # 限制演示语料大小，减少模型推理时间和 Lite 数据库占用。
    if len(questions) >= 10000:
        break

docs = questions

# 2. 为文档生成 dense 与 sparse 两类向量。
# fit_documents 会扫描完整文档集合，学习词表、文档频率、IDF 和平均文档长度，
# 并为每篇文档生成 BM25 的 sparse 权重。该过程完全在本机 Python 进程中执行。
embedder = HybridEmbedder(device="cpu")
docs_sparse = embedder.fit_documents(docs)

# MiniLM 只负责 dense 语义向量。normalize_embeddings=True 会把每个向量归一化到单位长度，
# 后续用内积 IP 搜索时，得分等价于余弦相似度。MiniLM-L6-v2 的维度为 384。
docs_dense = embedder.model.encode(
    docs,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

# 搜索端必须使用和插入端相同的 BM25 分词词表与 IDF，否则相同词语会落在不同 sparse 维度。
# 因此将拟合状态写入文件，ui.py 会读取该文件生成查询 sparse 向量。
embedder.save_bm25_state()
dense_dim = embedder.dim

# 3. 连接本地 Milvus Lite 数据库。
# URI 指向本地文件；ui.py 使用同一 URI，才能搜索到本脚本插入的 collection。
Path("./milvus-data").mkdir(parents=True, exist_ok=True)
connections.connect("default", uri="./milvus-data/milvus.db")

# 4. 定义 collection 的字段结构。
# pk 是 Milvus 自动生成的主键，auto_id=True 时主键类型应为 INT64。
# text 用于结果展示；两个向量字段分别保存词法和语义检索表示。
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields, "")
col_name = "hybrid_demo"

# 每次重新插入时重建示例 collection，防止旧 schema、旧向量维度或旧索引造成冲突。
# 这会删除本地数据库中同名 collection 的旧数据；本脚本之后会重新插入完整演示数据。
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Strong")

# 5. 为两个向量字段分别建索引。
# sparse 向量要求 SPARSE_INVERTED_INDEX，BM25 查询和文档权重通过 IP 内积打分。
# dense 使用 FLAT 精确搜索；小规模示例无需近似索引。由于 dense 向量已归一化，
# IP 排序等价于余弦相似度排序。
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)

# load 将 collection 加载到可搜索状态。插入前加载是该示例使用 Collection API 的流程。
col.load()

# 6. 分批插入原文、BM25 sparse 向量和 MiniLM dense 向量。
# 每批 50 条控制单个 RPC 请求大小。每条记录的列顺序需要与 schema 顺序对应：
# text -> sparse_vector -> dense_vector。主键由 Milvus 自动生成，因此不传 pk。
for i in range(0, len(docs), 50):
    batched_entities = [
        docs[i : i + 50],
        docs_sparse[i : i + 50],
        docs_dense[i : i + 50].tolist(),
    ]
    col.insert(batched_entities)

# flush 将缓冲中的插入数据持久化，确保之后 UI 连接数据库时可以检索到数据。
col.flush()
