"""Streamlit 查询界面：分别展示 MiniLM、BM25 及二者融合后的检索结果。"""

import re

import streamlit as st
from pymilvus import AnnSearchRequest, Collection, WeightedRanker, connections

from embeddings import HybridEmbedder


# 页面顶部展示项目图片和标题。运行 Streamlit 时请从该示例项目目录启动，
# 这样相对路径下的图片、本地数据库和 BM25 状态文件都能被正确找到。
st.image("./pic/Milvus_Logo_Official.png", width=200)
st.title("Milvus Hybrid Search Demo")


@st.cache_resource
def get_model():
    """加载 MiniLM，并恢复插入阶段保存的 BM25 词表及 IDF 统计。

    cache_resource 让 Streamlit 多次页面刷新时复用模型对象，避免重复加载模型权重。
    BM25 状态文件必须与 milvus.db 中的 collection 来自同一次插入运行。
    """
    embedder = HybridEmbedder(device="cpu")
    embedder.load_bm25_state()
    return embedder


@st.cache_resource
def get_collection():
    """连接与 insert.py 相同的本地数据库，并加载 hybrid_demo collection。"""
    connections.connect("default", uri="./milvus-data/milvus.db")
    col = Collection("hybrid_demo")
    col.load()
    return col


def hybrid_search(query_embeddings, sparse_weight=1.0, dense_weight=1.0):
    """使用两路向量搜索并融合结果。

    query_embeddings 中的 sparse 与 dense 查询向量必须分别和 collection 中对应字段匹配：
    - sparse 查询向量使用与文档相同的 CountVectorizer 词表及 IDF；
    - dense 查询向量使用与文档相同的 MiniLM 模型和归一化方式。

    sparse_weight/dense_weight 控制 WeightedRanker 融合两路检索分数时各自的权重。
    如果某一路权重为 0，该通道仍参与请求，但其结果分数不影响最终排序，因此可用于
    在同一个界面中分别观察单路检索效果。具体权重可根据实际语料和检索效果调节。
    """
    col = get_collection()

    # sparse 子请求按 BM25 权重的内积打分。AnnSearchRequest 中的向量字段名必须
    # 与 collection schema 完全一致；limit=10 表示先从该通道取最多 10 个候选。
    sparse_req = AnnSearchRequest(
        query_embeddings["sparse"],
        "sparse_vector",
        {"metric_type": "IP"},
        limit=10,
    )

    # dense 子请求使用 MiniLM 归一化向量的 IP 分数，等价于余弦相似度。
    dense_req = AnnSearchRequest(
        query_embeddings["dense"],
        "dense_vector",
        {"metric_type": "IP"},
        limit=10,
    )

    # hybrid_search 执行多个 AnnSearchRequest，并用 WeightedRanker 融合各通道分数。
    # output_fields 指定返回原文，界面无需再根据主键读取实体数据。
    results = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=WeightedRanker(sparse_weight, dense_weight),
        limit=10,
        output_fields=["text"],
    )

    # hybrid_search 的结果按查询逐条组织；当前界面只提交一条 query，取第一组命中。
    return [hit.fields["text"] for hit in results[0]] if results else []


def colorize_terms(query, text):
    """把查询中的字词在命中文本里标红，便于观察 BM25 命中的关键词。

    这里只做展示层的简单子串高亮，不参与 BM25 分词或搜索打分。正则转义可以避免
    查询里包含括号、加号等字符时被误当成正则表达式语法。
    """
    terms = sorted(
        set(re.findall(r"\b\w+\b", query, flags=re.UNICODE)),
        key=len,
        reverse=True,
    )
    if not terms:
        return text
    pattern = re.compile(
        "(" + "|".join(re.escape(term) for term in terms) + ")",
        re.IGNORECASE,
    )
    # Streamlit Markdown 的 :red[...] 语法用于显示红色文本。
    return pattern.sub(r":red[\1]", text)


# 输入框和按钮构成查询入口。只有用户点击搜索且输入非空时才进行模型推理和数据库查询。
query = st.text_input("Enter your search query:")
search_button = st.button("Search")

if search_button and query:
    # 一次生成 dense 和 sparse 查询向量，下面三个栏位共用，避免重复跑模型。
    query_embeddings = get_model().encode_queries([query])
    col1, col2, col3 = st.columns(3)

    # 第一栏只看语义通道：BM25 权重设为 0。
    with col1:
        st.header("Dense")
        for result in hybrid_search(
            query_embeddings,
            sparse_weight=0.0,
            dense_weight=1.0,
        ):
            st.markdown(result)

    # 第二栏只看词法通道：dense 权重设为 0，并标出查询词方便检查关键词召回。
    with col2:
        st.header("BM25")
        for result in hybrid_search(
            query_embeddings,
            sparse_weight=1.0,
            dense_weight=0.0,
        ):
            st.markdown(colorize_terms(query, result))

    # 第三栏融合两种检索。BM25 与 MiniLM 分数范围不同，当前 sparse 权重设得较低，
    # 这是一个演示初值，不是普适最优值；可结合标注查询集调整权重。
    with col3:
        st.header("Hybrid")
        for result in hybrid_search(
            query_embeddings,
            sparse_weight=0.1,
            dense_weight=1.0,
        ):
            st.markdown(colorize_terms(query, result))
