#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
from langchain_core.documents import Document
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever,TFIDFRetriever
from langchain_core.pydantic_v1 import PrivateAttr

def str_to_Document(data):
    docs = []
    for idx, line in enumerate(data):
        line = line.strip("\n").strip()
        words = line.split("\t")
        docs.append(Document(page_content=words[0], metadata={"id": idx}))
    return docs
class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data):
        self.embeddings  = HuggingFaceEmbeddings(
                               model_name = model_path,
                               model_kwargs = {"device":"cuda"}
                               # model_kwargs = {"device":"cuda:1"}
                           )
        docs = str_to_Document(data)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        del self.embeddings
        torch.cuda.empty_cache()

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k)
       return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store
    
class BMRetriever(BaseRetriever):
    _retrievers: BM25Retriever = PrivateAttr()
    def __init__(self, documents: List[str], k):
        super().__init__()
        docs = str_to_Document(documents)
        self._retrievers = BM25Retriever.from_documents(docs, k=k)
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self._retrievers.invoke(query)
        return results

class TFIDF(BaseRetriever):
    _retrievers: TFIDFRetriever = PrivateAttr()
    def __init__(self, documents: List[str], k):
        super().__init__()
        docs = str_to_Document(documents)
        self._retrievers = TFIDFRetriever.from_documents(docs, k=k)
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self._retrievers.invoke(query)
    
class BGERetriever(BaseRetriever):
    _retrievers: BM25Retriever = PrivateAttr()
    def __init__(self, documents: List[str], k):
        super().__init__()
        docs = str_to_Document(documents)
        embedding = HuggingFaceEmbeddings(model_name="pre_train_model/bge_large_zh", model_kwargs = {"device":"cuda:0"})
        self._retrievers = FAISS.from_documents(docs, embedding).as_retriever(search_kwargs={"k": k})
        del embedding
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self._retrievers.invoke(query)
        return results

class GTERetriever(BaseRetriever):
    _retrievers: BM25Retriever = PrivateAttr()
    def __init__(self, documents: List[str], k):
        super().__init__()
        docs = str_to_Document(documents)
        embedding = HuggingFaceEmbeddings(model_name="pre_train_model/GTE_base_zh", model_kwargs = {"device":"cuda:0"})
        self._retrievers = FAISS.from_documents(docs, embedding).as_retriever(search_kwargs={"k": k})
        del embedding
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self._retrievers.invoke(query)
        return results

class BCERetriever(BaseRetriever):
    _retrievers: BM25Retriever = PrivateAttr()
    def __init__(self, documents: List[str], k):
        super().__init__()
        docs = str_to_Document(documents)
        embedding = HuggingFaceEmbeddings(model_name="pre_train_model/BCE_embedding_base_v1", model_kwargs = {"device":"cuda:0", "trust_remote_code": True})
        self._retrievers = FAISS.from_documents(docs, embedding).as_retriever(search_kwargs={"k": k})
        del embedding
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self._retrievers.invoke(query)
        return results

class BCEEmbedding_model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('pre_train_model/BCE_embedding_base_v1',trust_remote_code=True)
        self.model = AutoModel.from_pretrained('pre_train_model/BCE_embedding_base_v1', trust_remote_code=True)
        self.device = 'cuda'  # if no GPU, set "cpu"
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs_on_device, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
        return embeddings.cpu().tolist()
    def embed_query(self, query: str) -> List[float]:
        inputs = self.tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs_on_device, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
        return embeddings.cpu().tolist()[0]

class BCERetrieverv2(BaseRetriever):
    _retrievers: BM25Retriever = PrivateAttr()
    def __init__(self, documents: List[str], k):
        super().__init__()
        docs = str_to_Document(documents)
        embedding = BCEEmbedding_model()
        self._retrievers = FAISS.from_documents(docs, embedding).as_retriever(search_kwargs={"k": k})
        del embedding
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self._retrievers.invoke(query)
        return results

if __name__ == "__main__":
    base = "."
    # model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    # dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    # dp.ParseAllPage(max_seq = 256)
    # dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    # dp.ParseOnePageWithRule(max_seq = 256)
    # dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    print(type(data))

    # faissretriever = FaissRetriever(model_name, data)
    # BM25retriever = BMRetriever(data, k=8)
    # TFIDFretriever = TFIDF(data, k=8)
    BGE = BCERetrieverv2(data, k=8)
    faiss_ans = BGE.invoke("自动驾驶")
    print(len(faiss_ans))
    print(faiss_ans)

    # embedding = BCEEmbedding_model()
    # query = "如何预防新冠肺炎"
    # # query_emb = embedding.embed_query(query)
    # # print(query_emb)
    # # print(type(query_emb))
    # # print(type(query_emb[0]))
    # # print(type(query_emb[0][0]))
    # query_all = ["如何预防新冠肺炎", "如何预肺炎"]
    # query_all_emb = embedding.embed_documents(query_all)
    # print(query_all_emb)
    # print(len(query_all_emb[0]), len(query_all_emb[1]))
    # print(len(embedding.embed_query("自动驾驶")))