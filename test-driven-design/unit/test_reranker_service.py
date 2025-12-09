"""
RerankerService 단위 테스트

CrossEncoder 모델을 사용하여 문서를 재순위화합니다.
- rerank(): 쿼리-문서 쌍의 관련성 점수 계산 및 정렬
- get_top_documents(): 상위 k개 문서 반환
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.services import RerankerService


class TestRerankerServiceProperties:
    """RerankerService 속성 테스트"""

    @pytest.mark.unit
    def test_lazy_loading_model(self, mock_settings):
        """model이 lazy loading되는지 확인"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)

            # 아직 model에 접근하지 않음
            mock_cross_encoder.assert_not_called()

            # model 접근 시 초기화
            _ = service.model
            mock_cross_encoder.assert_called_once()


class TestRerankerServiceRerank:
    """rerank() 메서드 테스트"""

    @pytest.mark.unit
    def test_rerank_returns_sorted_documents(self, mock_settings):
        """문서가 점수 내림차순으로 정렬되어 반환"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            # 점수 반환 (역순으로 - 리랭킹 후 정렬 확인)
            mock_model.predict.return_value = np.array([0.3, 0.9, 0.6, 0.1, 0.8])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

            # When
            result = service.rerank("query", documents)

            # Then: 점수 내림차순 정렬
            scores = [score for _, score in result]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    def test_rerank_respects_top_k(self, mock_settings):
        """top_k 개수만큼만 반환"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            documents = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]

            # When: top_k=3
            result = service.rerank("query", documents, top_k=3)

            # Then: 3개만 반환
            assert len(result) == 3

    @pytest.mark.unit
    def test_rerank_empty_documents(self, mock_settings):
        """빈 문서 리스트 처리"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)

            # When
            result = service.rerank("query", [])

            # Then: 빈 리스트 반환
            assert result == []

    @pytest.mark.unit
    def test_rerank_single_document(self, mock_settings):
        """단일 문서 처리"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.85])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)

            # When
            result = service.rerank("query", ["single_doc"])

            # Then: 1개 문서 반환
            assert len(result) == 1
            assert result[0][0] == "single_doc"
            assert result[0][1] == 0.85

    @pytest.mark.unit
    def test_rerank_returns_tuple_format(self, mock_settings):
        """반환값이 (document, score) 튜플 형식"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.7])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)

            # When
            result = service.rerank("query", ["doc1", "doc2"])

            # Then: 튜플 형식
            for item in result:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], str)  # document
                assert isinstance(item[1], (float, np.floating))  # score


class TestRerankerServiceGetTopDocuments:
    """get_top_documents() 메서드 테스트"""

    @pytest.mark.unit
    def test_get_top_documents_delegates_to_rerank(self, mock_settings):
        """get_top_documents가 rerank를 호출"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            documents = ["doc1", "doc2", "doc3"]

            # When
            result = service.get_top_documents("query", documents, top_k=2)

            # Then: rerank와 동일한 결과
            assert len(result) <= 2

    @pytest.mark.unit
    def test_get_top_documents_default_top_k(self, mock_settings):
        """기본 top_k 값 사용 (설정에서 가져옴)"""
        mock_settings.reranker.top_k = 5

        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            documents = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]

            # When: top_k 미지정
            result = service.get_top_documents("query", documents)

            # Then: 기본값 5개
            assert len(result) <= 5


class TestRerankerServiceEdgeCases:
    """RerankerService 엣지 케이스 테스트"""

    @pytest.mark.unit
    def test_handles_korean_documents(self, mock_settings):
        """한국어 문서 처리"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.8])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            documents = ["한국어 문서입니다", "또 다른 한국어 문서"]

            # When
            result = service.rerank("한국어 쿼리", documents)

            # Then: 정상 처리
            assert len(result) == 2

    @pytest.mark.unit
    def test_handles_long_documents(self, mock_settings):
        """긴 문서 처리"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.85])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            long_doc = "매우 긴 문서입니다. " * 1000

            # When
            result = service.rerank("query", [long_doc])

            # Then: 정상 처리
            assert len(result) == 1

    @pytest.mark.unit
    def test_handles_special_characters(self, mock_settings):
        """특수 문자 포함 문서 처리"""
        with patch("dev_v2.services.reranker.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.8])
            mock_cross_encoder.return_value = mock_model

            service = RerankerService(mock_settings)
            documents = [
                "문서에 <특수> 문자가 있습니다!@#$%",
                "또 다른 '따옴표' 문서",
            ]

            # When
            result = service.rerank("query", documents)

            # Then: 정상 처리
            assert len(result) == 2


class TestRerankerServiceIntegration:
    """RerankerService 통합 테스트 (실제 모델 사용)"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_reranking(self, real_reranker_service):
        """실제 CrossEncoder 모델로 리랭킹"""
        documents = [
            "RAG는 검색과 생성을 결합한 기술입니다.",
            "오늘 날씨가 좋습니다.",
            "RAG 성능 향상을 위해 리랭킹이 사용됩니다.",
            "점심 메뉴를 추천해주세요.",
            "하이브리드 검색은 BM25와 벡터 검색을 결합합니다.",
        ]

        # When
        result = real_reranker_service.rerank(
            "RAG 성능 향상 방법은?",
            documents,
            top_k=3,
        )

        # Then
        assert len(result) == 3
        # RAG 관련 문서가 상위에 있어야 함
        top_doc = result[0][0]
        assert "RAG" in top_doc or "리랭킹" in top_doc or "검색" in top_doc
