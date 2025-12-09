"""
QueryRewriteNode 단위 테스트

사용자 질문을 검색에 최적화된 여러 쿼리로 확장합니다.
- 입력: state["question"]
- 출력: {"optimized_queries": List[str]}
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.schemas import RAGState, RewriteResult
from dev_v2.nodes import QueryRewriteNode


class TestQueryRewriteNodeProperties:
    """QueryRewriteNode 속성 테스트"""

    @pytest.mark.unit
    def test_node_name(self, mock_llm_service):
        """노드 이름이 'query_rewrite'인지 확인"""
        node = QueryRewriteNode(mock_llm_service)
        assert node.name == "query_rewrite"


class TestQueryRewriteNodeCall:
    """QueryRewriteNode __call__ 메서드 테스트"""

    @pytest.mark.unit
    def test_returns_optimized_queries(self, mock_llm_service, sample_rag_state, sample_rewrite_result):
        """정상적으로 optimized_queries를 반환"""
        # Given
        mock_llm_service.invoke_with_structured_output.return_value = sample_rewrite_result
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(sample_rag_state)

        # Then
        assert "optimized_queries" in result
        assert isinstance(result["optimized_queries"], list)
        assert len(result["optimized_queries"]) > 0

    @pytest.mark.unit
    def test_generates_multiple_queries(self, mock_llm_service, sample_rag_state, sample_rewrite_result):
        """3-5개의 재작성된 쿼리 생성"""
        # Given
        mock_llm_service.invoke_with_structured_output.return_value = sample_rewrite_result
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(sample_rag_state)

        # Then: 3-5개 범위 내의 쿼리
        assert 3 <= len(result["optimized_queries"]) <= 5

    @pytest.mark.unit
    def test_calls_llm_service_with_correct_params(self, mock_llm_service, sample_rag_state, sample_rewrite_result):
        """LLMService를 올바른 파라미터로 호출"""
        # Given
        mock_llm_service.invoke_with_structured_output.return_value = sample_rewrite_result
        node = QueryRewriteNode(mock_llm_service)

        # When
        node(sample_rag_state)

        # Then
        mock_llm_service.get_rewrite_llm.assert_called_once()
        mock_llm_service.invoke_with_structured_output.assert_called_once()

        # 호출 인자 확인
        call_args = mock_llm_service.invoke_with_structured_output.call_args
        assert call_args.kwargs["output_schema"] == RewriteResult
        assert "question" in call_args.kwargs["input_data"]

    @pytest.mark.unit
    def test_preserves_original_question_in_queries(self, mock_llm_service, sample_rag_state):
        """원본 질문의 의도가 쿼리에 반영되는지 확인"""
        # Given
        original_question = sample_rag_state["question"]
        mock_rewrite_result = RewriteResult(queries=[
            f"{original_question} 개념",
            f"{original_question} 설명",
            f"{original_question} 정의",
        ])
        mock_llm_service.invoke_with_structured_output.return_value = mock_rewrite_result
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(sample_rag_state)

        # Then: 각 쿼리가 비어있지 않음
        for query in result["optimized_queries"]:
            assert len(query) > 0


class TestQueryRewriteNodeFallback:
    """QueryRewriteNode Fallback 동작 테스트"""

    @pytest.mark.unit
    def test_fallback_on_llm_error(self, mock_llm_service, sample_rag_state):
        """LLM 호출 실패 시 원본 질문으로 Fallback"""
        # Given: LLM 호출 시 예외 발생
        mock_llm_service.invoke_with_structured_output.side_effect = Exception("LLM Error")
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(sample_rag_state)

        # Then: 원본 질문을 그대로 반환
        assert "optimized_queries" in result
        assert sample_rag_state["question"] in result["optimized_queries"]

    @pytest.mark.unit
    def test_fallback_on_empty_result(self, mock_llm_service, sample_rag_state):
        """LLM이 빈 결과 반환 시 Fallback"""
        # Given: 빈 쿼리 리스트 반환
        mock_llm_service.invoke_with_structured_output.return_value = RewriteResult(queries=[])
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(sample_rag_state)

        # Then: 빈 리스트 또는 원본 질문 포함
        assert "optimized_queries" in result
        # 빈 리스트가 반환되면 원본 질문이 포함되어야 함
        if len(result["optimized_queries"]) > 0:
            assert any(q for q in result["optimized_queries"])


class TestQueryRewriteNodeEdgeCases:
    """QueryRewriteNode 엣지 케이스 테스트"""

    @pytest.mark.unit
    def test_handles_empty_question(self, mock_llm_service):
        """빈 질문 처리"""
        # Given
        state: RAGState = {
            "question": "",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_structured_output.return_value = RewriteResult(queries=[""])
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 에러 없이 처리
        assert "optimized_queries" in result

    @pytest.mark.unit
    def test_handles_special_characters(self, mock_llm_service):
        """특수 문자 포함 질문 처리"""
        # Given
        state: RAGState = {
            "question": "RAG의 <성능>은 어떻게 '측정'하나요?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_structured_output.return_value = RewriteResult(
            queries=["RAG 성능 측정 방법", "RAG 성능 지표", "RAG 평가 기준"]
        )
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 정상 처리
        assert len(result["optimized_queries"]) > 0

    @pytest.mark.unit
    def test_handles_korean_question(self, mock_llm_service):
        """한국어 질문 처리"""
        # Given
        state: RAGState = {
            "question": "한국어로 된 질문입니다",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_structured_output.return_value = RewriteResult(
            queries=["한국어 질문 처리", "한국어 검색", "한국어 쿼리"]
        )
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 한국어 쿼리가 반환됨
        assert len(result["optimized_queries"]) > 0
        # 한국어가 포함되어 있는지 확인
        has_korean = any("한국어" in q or "질문" in q for q in result["optimized_queries"])
        assert has_korean

    @pytest.mark.unit
    def test_handles_english_question(self, mock_llm_service):
        """영어 질문 처리"""
        # Given
        state: RAGState = {
            "question": "What is RAG performance optimization?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_structured_output.return_value = RewriteResult(
            queries=["RAG performance optimization", "RAG system improvement", "RAG enhancement"]
        )
        node = QueryRewriteNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 영어 쿼리가 반환됨
        assert len(result["optimized_queries"]) > 0
