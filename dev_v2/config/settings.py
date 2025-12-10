"""
Configuration settings for RAG Server
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMSettings:
    """LLM(API) Model Settings"""
    # Query Rewirte
    rewrite_model: str = "gpt-4o-mini"
    rewrite_temperature: float = 0.0
    # Answer Generate
    generator_model: str = "gpt-4o"
    generator_temperature: float = 0.0


@dataclass
class VectorStoreSettings:
    """VectorDB Settings"""
    # Weaviate
    weaviate_version: str = "1.27.0"
    data_path: str = "./dev_v2/my_weaviate_data"
    collection_name: str = "AdvancedRAG_Chunk"
    embedding_model: str = "text-embedding-3-small"
    bm25_b: float = 0.75        # 문서 길이 정규화
    bm25_k1: float = 1.2        # 단어 빈도 포화도


@dataclass
class RerankerSettings:
    """Reranker 관련 설정"""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5              # Lost in the Middle 방지 개수


@dataclass
class RetrieverSettings:
    """Retriever 관련 설정"""
    # Hybrid Search(Keyword + Semantic)
    hybrid_alpha: float = 0.5
    initial_limit: int = 30


@dataclass
class PreprocessingSettings:
    """전처리 관련 설정"""
    # SemanticChunker
    embedding_model: str = "text-embedding-3-small"
    breakpoint_type: str = "percentile"
    breakpoint_threshold: float = 95.0
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    # TextNormalizer
    remove_extra_whitespace: bool = True
    remove_extra_newlines: bool = True
    remove_special_chars: bool = False
    min_line_length: int = 0


@dataclass
class Settings:
    """전체 설정 관리"""
    llm: LLMSettings = field(default_factory=LLMSettings)
    vectorstore: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    retriever: RetrieverSettings = field(default_factory=RetrieverSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)

    def __post_init__(self):
        # 환경 변수에서 데이터 경로 오버라이드 가능
        if os.getenv("WEAVIATE_DATA_PATH"):
            self.vectorstore.data_path = os.getenv("WEAVIATE_DATA_PATH")


# 싱글톤 인스턴스
settings = Settings()
