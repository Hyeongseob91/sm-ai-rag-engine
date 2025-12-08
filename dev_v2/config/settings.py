"""
Configuration settings for RAG Server
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMSettings:
    """LLM 관련 설정"""
    rewrite_model: str = "gpt-4o-mini"
    rewrite_temperature: float = 0
    generator_model: str = "gpt-4o"
    generator_temperature: float = 0


@dataclass
class VectorStoreSettings:
    """Vector Store 관련 설정"""
    weaviate_version: str = "1.27.0"
    data_path: str = "./my_weaviate_data"
    collection_name: str = "AdvancedRAG_Chunk"
    embedding_model: str = "text-embedding-3-small"
    bm25_b: float = 0.75
    bm25_k1: float = 1.2


@dataclass
class RerankerSettings:
    """Reranker 관련 설정"""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5


@dataclass
class RetrieverSettings:
    """Retriever 관련 설정"""
    hybrid_alpha: float = 0.5
    initial_limit: int = 10


@dataclass
class Settings:
    """전체 설정 관리"""
    llm: LLMSettings = field(default_factory=LLMSettings)
    vectorstore: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    retriever: RetrieverSettings = field(default_factory=RetrieverSettings)

    def __post_init__(self):
        # 환경 변수에서 데이터 경로 오버라이드 가능
        if os.getenv("WEAVIATE_DATA_PATH"):
            self.vectorstore.data_path = os.getenv("WEAVIATE_DATA_PATH")


# 싱글톤 인스턴스
settings = Settings()
