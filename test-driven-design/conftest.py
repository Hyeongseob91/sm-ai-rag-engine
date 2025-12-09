"""
pytest Fixtures - 공통 Mock 및 테스트 데이터

이 파일의 fixtures는 모든 테스트에서 자동으로 사용 가능합니다.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Tuple

import pytest

# dev_v2 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dev_v2.config import Settings
from dev_v2.schemas import RAGState, RouteQuery, RewriteResult
from dev_v2.services import LLMService, VectorStoreService, RerankerService


# =============================================================================
# pytest markers 등록
# =============================================================================
def pytest_configure(config):
    """pytest 마커 등록"""
    config.addinivalue_line("markers", "unit: 단위 테스트 (Mock 기반)")
    config.addinivalue_line("markers", "integration: 통합 테스트 (실제 서비스)")
    config.addinivalue_line("markers", "e2e: End-to-End 테스트")
    config.addinivalue_line("markers", "slow: 느린 테스트 (API 호출 등)")


# =============================================================================
# Settings Fixtures
# =============================================================================
@pytest.fixture
def mock_settings() -> Settings:
    """테스트용 Settings Mock"""
    settings = Mock(spec=Settings)

    # LLM 설정
    settings.llm = Mock()
    settings.llm.rewrite_model = "gpt-4o-mini"
    settings.llm.rewrite_temperature = 0.0
    settings.llm.generator_model = "gpt-4o"
    settings.llm.generator_temperature = 0.0

    # VectorStore 설정
    settings.vectorstore = Mock()
    settings.vectorstore.collection_name = "test_collection"
    settings.vectorstore.embedding_model = "text-embedding-3-small"
    settings.vectorstore.data_path = "./test_weaviate_data"

    # Reranker 설정
    settings.reranker = Mock()
    settings.reranker.model_name = "BAAI/bge-reranker-v2-m3"
    settings.reranker.top_k = 5

    # Retriever 설정
    settings.retriever = Mock()
    settings.retriever.hybrid_alpha = 0.5
    settings.retriever.initial_limit = 30

    return settings


# =============================================================================
# Service Mock Fixtures
# =============================================================================
@pytest.fixture
def mock_llm_service(mock_settings) -> Mock:
    """LLMService Mock"""
    service = Mock(spec=LLMService)
    service.settings = mock_settings

    # get_rewrite_llm Mock
    mock_rewrite_llm = MagicMock()
    service.get_rewrite_llm.return_value = mock_rewrite_llm

    # get_generator_llm Mock
    mock_generator_llm = MagicMock()
    service.get_generator_llm.return_value = mock_generator_llm

    return service


@pytest.fixture
def mock_vectorstore_service(mock_settings) -> Mock:
    """VectorStoreService Mock"""
    service = Mock(spec=VectorStoreService)
    service.settings = mock_settings
    service.is_ready.return_value = True
    return service


@pytest.fixture
def mock_reranker_service(mock_settings) -> Mock:
    """RerankerService Mock"""
    service = Mock(spec=RerankerService)
    service.settings = mock_settings
    return service


# =============================================================================
# Sample Data Fixtures
# =============================================================================
@pytest.fixture
def sample_question() -> str:
    """테스트용 질문"""
    return "RAG 성능 고도화의 개념은 무엇인가요?"


@pytest.fixture
def sample_simple_question() -> str:
    """일반 대화용 질문 (검색 불필요)"""
    return "안녕하세요?"


@pytest.fixture
def sample_rag_state(sample_question) -> RAGState:
    """테스트용 RAGState (초기 상태)"""
    return {
        "question": sample_question,
        "optimized_queries": [],
        "retrieved_docs": [],
        "final_answer": "",
    }


@pytest.fixture
def sample_rag_state_with_queries(sample_question) -> RAGState:
    """쿼리 재작성 후의 RAGState"""
    return {
        "question": sample_question,
        "optimized_queries": [
            "RAG 성능 고도화 개념",
            "RAG 시스템 최적화 방법",
            "검색 증강 생성 성능 향상",
        ],
        "retrieved_docs": [],
        "final_answer": "",
    }


@pytest.fixture
def sample_rag_state_with_docs(sample_question) -> RAGState:
    """문서 검색 후의 RAGState"""
    return {
        "question": sample_question,
        "optimized_queries": [
            "RAG 성능 고도화 개념",
            "RAG 시스템 최적화 방법",
        ],
        "retrieved_docs": [
            "RAG 성능 고도화란 검색 증강 생성 시스템의 정확도와 효율성을 높이는 기술입니다.",
            "주요 기법으로는 Query Rewriting, Hybrid Search, Reranking 등이 있습니다.",
            "Lost in the Middle 문제를 해결하기 위해 상위 문서만 선별합니다.",
        ],
        "final_answer": "",
    }


@pytest.fixture
def sample_documents() -> List[str]:
    """테스트용 문서 리스트"""
    return [
        "RAG(Retrieval-Augmented Generation)은 검색과 생성을 결합한 AI 기술입니다.",
        "Hybrid Search는 키워드 검색(BM25)과 벡터 검색을 결합합니다.",
        "Reranking은 초기 검색 결과를 재정렬하여 품질을 향상시킵니다.",
        "Query Rewriting은 사용자 질문을 검색에 최적화된 형태로 변환합니다.",
        "Lost in the Middle 현상은 긴 컨텍스트에서 중간 정보를 놓치는 문제입니다.",
    ]


@pytest.fixture
def sample_reranked_docs() -> List[Tuple[str, float]]:
    """리랭킹된 문서 리스트 (문서, 점수)"""
    return [
        ("RAG 성능 고도화란 검색 증강 생성 시스템의 정확도를 높이는 기술입니다.", 0.95),
        ("주요 기법으로는 Query Rewriting, Hybrid Search가 있습니다.", 0.88),
        ("Reranking은 초기 검색 결과를 재정렬합니다.", 0.82),
        ("Lost in the Middle 문제 해결이 중요합니다.", 0.75),
        ("벡터 검색은 의미 기반 검색을 가능하게 합니다.", 0.70),
    ]


# =============================================================================
# Pydantic Model Fixtures
# =============================================================================
@pytest.fixture
def sample_route_query_vectorstore() -> RouteQuery:
    """VectorStore로 라우팅하는 RouteQuery"""
    return RouteQuery(datasource="vectorstore")


@pytest.fixture
def sample_route_query_llm() -> RouteQuery:
    """LLM으로 라우팅하는 RouteQuery"""
    return RouteQuery(datasource="llm")


@pytest.fixture
def sample_rewrite_result() -> RewriteResult:
    """Query Rewrite 결과"""
    return RewriteResult(queries=[
        "RAG 성능 고도화 개념 설명",
        "RAG 시스템 최적화 방법론",
        "검색 증강 생성 성능 향상 기법",
        "RAG 파이프라인 개선 전략",
        "Advanced RAG 기술 소개",
    ])


# =============================================================================
# Real Service Fixtures (Integration Tests)
# =============================================================================
@pytest.fixture
def real_settings() -> Settings:
    """실제 Settings (Integration 테스트용)"""
    from dotenv import load_dotenv
    load_dotenv()
    return Settings()


@pytest.fixture
def real_llm_service(real_settings) -> LLMService:
    """실제 LLMService (Integration 테스트용)"""
    return LLMService(real_settings)


@pytest.fixture
def real_reranker_service(real_settings) -> RerankerService:
    """실제 RerankerService (Integration 테스트용)"""
    return RerankerService(real_settings)


@pytest.fixture
def real_vectorstore_service(real_settings) -> VectorStoreService:
    """실제 VectorStoreService (Integration 테스트용)"""
    service = VectorStoreService(real_settings)
    yield service
    # 테스트 후 정리
    service.close()
