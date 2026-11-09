import pickle
from pathlib import Path
"""
bm25_state.pkl信息查看
"""

path = Path("milvus-data/bm25_state.pkl")
with path.open("rb") as f:
    vectorizer, idf, avgdl, doc_count = pickle.load(f)

print("文档数:", doc_count)
print("平均文档长度:", avgdl)
print("词表大小:", len(vectorizer.vocabulary_))
print("IDF 数组长度:", len(idf))
print("IDF 部分内容:", idf[:20])
print("IDF 部分内容:", idf[10193])
print("部分词表:", list(vectorizer.vocabulary_.items())[0:20])


word = "india"
term_id = vectorizer.vocabulary_.get(word)
print("词项编号:", term_id)
print("IDF:", idf[term_id] if term_id is not None else "词不在词表中")