"""
Graph State Schemas for RAG Pipeline
"""
from typing import List, TypedDict


class RAGState(TypedDict):
    """
    RAG 파이프라인의 전역 상태 (Global State)
    LangGraph에서 노드 간 데이터 전달에 사용
    """
    question: str
    optimized_queries: List[str]
    retrieved_docs: List[str]
    final_answer: str


class QueryOutput(TypedDict):
    """Query Rewrite 노드의 출력 스키마"""
    optimized_queries: List[str]


class RetrievalOutput(TypedDict):
    """Retriever 노드의 출력 스키마"""
    retrieved_docs: List[str]


class GenerationOutput(TypedDict):
    """Generator 노드의 출력 스키마"""
    final_answer: str
