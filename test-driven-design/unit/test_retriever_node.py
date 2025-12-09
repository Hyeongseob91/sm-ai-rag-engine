"""
RetrieverNode 단위 테스트

Hybrid Search + Reranking을 통해 관련 문서를 검색합니다.
- 입력: state["question"], state["optimized_queries"]
- 출력: {"retrieved_docs": List[str]}
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.schemas import RAGState
from dev_v2.nodes import RetrieverNode


class TestRetrieverNodeProperties:
    """RetrieverNode 속성 테스트"""

    @pytest.mark.unit
    def test_node_name(self, mock_vectorstore_service, mock_reranker_service):
        """노드 이름이 'retriever'인지 확인"""
        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        assert node.name == "retriever"


class TestRetrieverNodeCall:
    """RetrieverNode __call__ 메서드 테스트"""

    @pytest.mark.unit
    def test_returns_retrieved_docs(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
        sample_documents,
        sample_reranked_docs,
    ):
        """정상적으로 retrieved_docs를 반환"""
        # Given
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(sample_rag_state_with_queries)

        # Then
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)

    @pytest.mark.unit
    def test_calls_hybrid_search_for_each_query(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
        sample_documents,
        sample_reranked_docs,
    ):
        """각 optimized_query에 대해 hybrid_search 호출"""
        # Given
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        num_queries = len(sample_rag_state_with_queries["optimized_queries"])

        # When
        node(sample_rag_state_with_queries)

        # Then: 각 쿼리마다 hybrid_search 호출
        assert mock_vectorstore_service.hybrid_search.call_count == num_queries

    @pytest.mark.unit
    def test_calls_reranker_with_original_question(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
        sample_documents,
        sample_reranked_docs,
    ):
        """원본 질문으로 리랭킹 수행"""
        # Given
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        node(sample_rag_state_with_queries)

        # Then: 원본 질문으로 리랭킹
        mock_reranker_service.get_top_documents.assert_called_once()
        call_args = mock_reranker_service.get_top_documents.call_args
        assert call_args.kwargs.get("query") == sample_rag_state_with_queries["question"] or \
               call_args.args[0] == sample_rag_state_with_queries["question"]

    @pytest.mark.unit
    def test_removes_duplicate_documents(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
    ):
        """중복 문서 제거"""
        # Given: 중복된 문서 반환
        duplicate_docs = ["doc1", "doc2", "doc1", "doc3", "doc2"]
        mock_vectorstore_service.hybrid_search.return_value = duplicate_docs
        mock_reranker_service.get_top_documents.return_value = [
            ("doc1", 0.9),
            ("doc2", 0.8),
            ("doc3", 0.7),
        ]

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(sample_rag_state_with_queries)

        # Then: 리랭커에 전달되는 문서는 중복 제거됨
        call_args = mock_reranker_service.get_top_documents.call_args
        docs_passed = call_args.kwargs.get("documents") or call_args.args[1]

        # 중복 제거 확인 (리랭커에 전달된 문서)
        assert len(docs_passed) <= len(set(duplicate_docs))

    @pytest.mark.unit
    def test_returns_top_k_documents(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
        sample_documents,
    ):
        """상위 k개 문서만 반환 (기본 5개)"""
        # Given
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        # 5개 문서 리랭킹 결과
        reranked = [(doc, 0.9 - i * 0.1) for i, doc in enumerate(sample_documents)]
        mock_reranker_service.get_top_documents.return_value = reranked[:5]

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(sample_rag_state_with_queries)

        # Then: 최대 5개 문서
        assert len(result["retrieved_docs"]) <= 5


class TestRetrieverNodeEdgeCases:
    """RetrieverNode 엣지 케이스 테스트"""

    @pytest.mark.unit
    def test_handles_empty_queries(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
    ):
        """빈 쿼리 리스트 처리"""
        # Given
        state: RAGState = {
            "question": "테스트 질문",
            "optimized_queries": [],  # 빈 쿼리
            "retrieved_docs": [],
            "final_answer": "",
        }

        mock_reranker_service.get_top_documents.return_value = []
        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(state)

        # Then: 빈 결과 반환
        assert "retrieved_docs" in result
        assert result["retrieved_docs"] == [] or len(result["retrieved_docs"]) == 0

    @pytest.mark.unit
    def test_handles_no_search_results(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
    ):
        """검색 결과 없음 처리"""
        # Given: 검색 결과 없음
        mock_vectorstore_service.hybrid_search.return_value = []
        mock_reranker_service.get_top_documents.return_value = []

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(sample_rag_state_with_queries)

        # Then: 빈 결과
        assert "retrieved_docs" in result

    @pytest.mark.unit
    def test_handles_single_query(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_documents,
        sample_reranked_docs,
    ):
        """단일 쿼리 처리"""
        # Given
        state: RAGState = {
            "question": "테스트 질문",
            "optimized_queries": ["단일 쿼리"],
            "retrieved_docs": [],
            "final_answer": "",
        }

        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(state)

        # Then: 정상 처리
        assert "retrieved_docs" in result
        mock_vectorstore_service.hybrid_search.assert_called_once()


class TestRetrieverNodeScoring:
    """RetrieverNode 점수 관련 테스트"""

    @pytest.mark.unit
    def test_documents_sorted_by_score(
        self,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_rag_state_with_queries,
        sample_documents,
    ):
        """문서가 점수 내림차순으로 정렬"""
        # Given: 리랭킹 결과 (점수 내림차순)
        reranked = [
            ("best_doc", 0.95),
            ("good_doc", 0.85),
            ("ok_doc", 0.75),
            ("fair_doc", 0.65),
            ("poor_doc", 0.55),
        ]

        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = reranked

        node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)

        # When
        result = node(sample_rag_state_with_queries)

        # Then: 첫 번째 문서가 가장 높은 점수
        if result["retrieved_docs"]:
            # 반환된 문서 중 첫 번째가 "best_doc"이어야 함
            # (튜플에서 문서만 추출하는 경우)
            first_doc = result["retrieved_docs"][0]
            if isinstance(first_doc, tuple):
                assert first_doc[0] == "best_doc"
            else:
                assert first_doc == "best_doc"
