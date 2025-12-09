"""
VectorStoreService 통합 테스트

Weaviate 벡터 DB를 사용한 실제 검색 테스트입니다.
- is_ready(): 연결 상태 확인
- create_collection(): 컬렉션 생성
- add_documents(): 문서 추가
- hybrid_search(): 하이브리드 검색
- close(): 연결 종료
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.services import VectorStoreService


class TestVectorStoreServiceConnection:
    """VectorStoreService 연결 테스트"""

    @pytest.mark.integration
    def test_is_ready_returns_true_when_connected(self, real_vectorstore_service):
        """연결 성공 시 True 반환"""
        assert real_vectorstore_service.is_ready() is True

    @pytest.mark.integration
    def test_close_releases_resources(self, real_settings):
        """close() 호출 시 리소스 해제"""
        service = VectorStoreService(real_settings)

        # 연결 확인
        assert service.is_ready()

        # When
        service.close()

        # Then: 연결 종료 (재연결 시 새 인스턴스 필요)
        # close 후에도 에러 없이 완료


class TestVectorStoreServiceCollection:
    """VectorStoreService 컬렉션 관리 테스트"""

    @pytest.mark.integration
    def test_create_collection_creates_new_collection(self, real_vectorstore_service):
        """새 컬렉션 생성"""
        # When & Then: 에러 없이 생성
        try:
            real_vectorstore_service.create_collection()
        except Exception as e:
            pytest.fail(f"컬렉션 생성 실패: {e}")

    @pytest.mark.integration
    def test_collection_property_returns_collection(self, real_vectorstore_service):
        """collection 속성이 컬렉션 반환"""
        collection = real_vectorstore_service.collection
        assert collection is not None


class TestVectorStoreServiceDocuments:
    """VectorStoreService 문서 관리 테스트"""

    @pytest.mark.integration
    def test_add_documents(self, real_vectorstore_service, sample_documents):
        """문서 추가"""
        # Given: 테스트용 문서
        docs = [{"content": doc} for doc in sample_documents[:3]]

        # When & Then: 에러 없이 추가
        try:
            real_vectorstore_service.add_documents(docs)
        except Exception as e:
            # 이미 문서가 있을 수 있으므로 특정 에러만 허용
            if "already exists" not in str(e).lower():
                pytest.fail(f"문서 추가 실패: {e}")


class TestVectorStoreServiceHybridSearch:
    """VectorStoreService Hybrid Search 테스트"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_hybrid_search_returns_results(self, real_vectorstore_service):
        """hybrid_search가 결과 반환"""
        # When
        results = real_vectorstore_service.hybrid_search(
            query="RAG 성능 향상",
            limit=5,
        )

        # Then: 리스트 반환 (결과가 없을 수도 있음)
        assert isinstance(results, list)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_hybrid_search_respects_limit(self, real_vectorstore_service):
        """limit 파라미터가 적용되는지 확인"""
        # When
        results = real_vectorstore_service.hybrid_search(
            query="검색 테스트",
            limit=3,
        )

        # Then: 최대 3개
        assert len(results) <= 3

    @pytest.mark.integration
    @pytest.mark.slow
    def test_hybrid_search_with_alpha(self, real_vectorstore_service):
        """alpha 파라미터 (BM25/Vector 비중) 테스트"""
        # When: BM25 위주 (alpha=0.2)
        results_bm25 = real_vectorstore_service.hybrid_search(
            query="RAG",
            alpha=0.2,
            limit=5,
        )

        # When: Vector 위주 (alpha=0.8)
        results_vector = real_vectorstore_service.hybrid_search(
            query="RAG",
            alpha=0.8,
            limit=5,
        )

        # Then: 둘 다 결과 반환
        assert isinstance(results_bm25, list)
        assert isinstance(results_vector, list)

    @pytest.mark.integration
    def test_hybrid_search_empty_query(self, real_vectorstore_service):
        """빈 쿼리 처리"""
        # When
        results = real_vectorstore_service.hybrid_search(
            query="",
            limit=5,
        )

        # Then: 빈 결과 또는 에러 없이 처리
        assert isinstance(results, list)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_hybrid_search_korean_query(self, real_vectorstore_service):
        """한국어 쿼리 검색"""
        # When
        results = real_vectorstore_service.hybrid_search(
            query="검색 증강 생성이란 무엇인가요?",
            limit=5,
        )

        # Then: 결과 반환
        assert isinstance(results, list)


class TestVectorStoreServiceEdgeCases:
    """VectorStoreService 엣지 케이스 테스트"""

    @pytest.mark.integration
    def test_handles_special_characters_in_query(self, real_vectorstore_service):
        """특수 문자 포함 쿼리 처리"""
        # When
        results = real_vectorstore_service.hybrid_search(
            query="RAG의 <성능>은 어떻게 '측정'하나요?",
            limit=5,
        )

        # Then: 에러 없이 처리
        assert isinstance(results, list)

    @pytest.mark.integration
    def test_handles_long_query(self, real_vectorstore_service):
        """긴 쿼리 처리"""
        long_query = "RAG 시스템의 성능을 측정하고 향상시키는 방법에 대해 " * 10

        # When
        results = real_vectorstore_service.hybrid_search(
            query=long_query,
            limit=5,
        )

        # Then: 에러 없이 처리
        assert isinstance(results, list)


class TestVectorStoreServiceUnit:
    """VectorStoreService 단위 테스트 (Mock 사용)"""

    @pytest.mark.unit
    def test_lazy_loading_client(self, mock_settings):
        """client가 lazy loading되는지 확인"""
        with patch("dev_v2.services.vectorstore.weaviate") as mock_weaviate:
            service = VectorStoreService(mock_settings)

            # 아직 client에 접근하지 않음
            mock_weaviate.connect_to_embedded.assert_not_called()

    @pytest.mark.unit
    def test_uses_correct_settings(self, mock_settings):
        """설정값이 올바르게 사용되는지 확인"""
        service = VectorStoreService(mock_settings)

        assert service.settings == mock_settings
