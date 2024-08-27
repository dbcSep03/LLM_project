from pdf_parse import DataProcess
from faiss_retriever import BMRetriever, BGERetriever
from vllm_model import agent
import json
from tqdm.auto import tqdm
import torch
from typing import TypedDict,List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from FlagEmbedding import FlagReranker
from langgraph.graph import StateGraph
from pprint import pprint
def torch_gc():
    torch.cuda.empty_cache()
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def get_documents(dir = "./data/train_a.pdf"):
    """获得文档"""
    dp =  DataProcess(pdf_path = dir)
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    return data

def get_question(state: GraphState):
    """对问题润色"""
    question = state["question"]
    results = question_chain.invoke({"question":question})
    # print(results)
    torch_gc()
    return {"question":results}

def retrieve(state: GraphState):
    """检索文档"""
    question = state["question"]
    results = BGE_Retriever.invoke(question)
    results = [doc.page_content for doc in results]
    # print(results)
    return {"documents":results, "question":question}

def rerank(state: GraphState):
    """重新排序"""
    documents = state["documents"]
    question = state["question"]
    if len(documents) == 0:
        return {"documents":[], "question":question}
    Q_D_Pairs = [[doc, question] for doc in documents]
    scores = BGE_Reranker.compute_score(Q_D_Pairs)
    # sorted_indices = [index for index, value in sorted(enumerate(scores), key=lambda x: x[1], reverse=True) if value > 0]
    # rerank有问题 因为按照道理来说相关程度是正的 但检索效果不好
    sorted_indices = [index for index, value in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    # print(sorted_indices)
    reranked_documents = [documents[index] for i, index in enumerate(sorted_indices) if i < 5]
    if len(reranked_documents) > 0:
        good_docs = []
        for doc in reranked_documents:
            good_doc = documents_chain.invoke({"documents":doc})
            good_docs.append(good_doc)
        # print(good_docs)
        return {"documents":good_docs, "question":question}
    # print(reranked_documents)
    return {"documents":reranked_documents, "question":question}
    
def generate(state: GraphState):
    """生成回答"""
    documents = state["documents"]
    question = state["question"]
    generated = "没有回复"
    for doc in documents:
        generated = generated_chain.invoke({"question":question, "documents":doc, "generation":generated})
        # print(generated)
        torch_gc()
    return {"question":question, "documents":documents, "generation":generated}

def judge_rerank(state: GraphState):
    """判断是否需要重新排序"""
    documents = state["documents"]
    if len(documents) == 0:
        return "no"
    return "yes"
def get_graph():

    workflow = StateGraph(GraphState)
    workflow.add_node("get_question", get_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("get_question")
    workflow.add_edge("get_question", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_conditional_edges(
        "rerank",
        judge_rerank,
        {
            "yes": "generate",
            "no": "get_question"
        }
    )
    
    app = workflow.compile()
    return app


if __name__ == "__main__":
    data = get_documents()
    llm = agent("Qwen")

    # 问题扩写
    question_prompt = PromptTemplate(
        template="""你是一个友善的助手，可以将输入的问题扩写为更加详细和合适的问题。问题是：{question}""",
        input_keys=["question"],
    )
    question_chain = question_prompt | llm | StrOutputParser()
    
    
    # BM25_Retriever = BMRetriever(data, 15)  # retriever 
    BGE_Retriever = BGERetriever(data, 15)  # retriever
    BGE_Reranker = FlagReranker('pre_train_model/bge-reranker-large', use_fp16=True) # reranker
    
    # 文档重新描述
    documents_prompt = PromptTemplate(
        template="""你是一个友善的助手，可以将输入的内容重写为更易理解的表述。输入是：{documents}""",
        input_keys=["documents"],
    )
    documents_chain = documents_prompt | llm | StrOutputParser()

    # 问题回答
    prompt = PromptTemplate(
        template="""
        你是一个友善的助手，对输入的问题根据输入背景，润色已回答的内容，达到更好的质量。\n
        如果背景和问题无关，请回答“无关”，不允许在回答中添加编造成分，回答请使用中文。\n
        问题是：{question},        
        背景是：{documents},
        回答是：{generation},
        润色结果为：
        """,
        input_keys=["question", "documents", "generation"],
    )
    generated_chain = prompt | llm | StrOutputParser()

    # test
    app = get_graph()
    results_all = []
    print(app.get_graph().print_ascii())
    with open("data/test_question.json", "r", encoding="utf-8") as f:
        jdata  = json.loads(f.read())
        print(len(jdata))
        for idx, line in tqdm(enumerate(jdata), total=len(jdata)):
            keep = {}
            keep["question"] = line["question"]
            question = line["question"]
            inputs = {"question":"question"}
            final_state = app.invoke(inputs)
            results = final_state["generation"]
            keep["generation"] = results
            keep["documents"] = final_state["documents"]
            results_all.append(keep)

    json.dump(results_all, open("./data/result_1.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
