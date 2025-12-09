"""
SimpleGeneratorNode 단위 테스트

검색 없이 LLM 지식으로 바로 답변을 생성합니다.
- 입력: state["question"]
- 출력: {"final_answer": str}
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.schemas import RAGState
from dev_v2.nodes import SimpleGeneratorNode


class TestSimpleGeneratorNodeProperties:
    """SimpleGeneratorNode 속성 테스트"""

    @pytest.mark.unit
    def test_node_name(self, mock_llm_service):
        """노드 이름이 'simple_generator'인지 확인"""
        node = SimpleGeneratorNode(mock_llm_service)
        assert node.name == "simple_generator"


class TestSimpleGeneratorNodeCall:
    """SimpleGeneratorNode __call__ 메서드 테스트"""

    @pytest.mark.unit
    def test_returns_final_answer(self, mock_llm_service, sample_simple_question):
        """정상적으로 final_answer를 반환"""
        # Given
        state: RAGState = {
            "question": sample_simple_question,
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        expected_answer = "안녕하세요! 무엇을 도와드릴까요?"
        mock_llm_service.invoke_with_string_output.return_value = expected_answer

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result
        assert isinstance(result["final_answer"], str)
        assert result["final_answer"] == expected_answer

    @pytest.mark.unit
    def test_does_not_use_retrieved_docs(self, mock_llm_service):
        """검색된 문서를 사용하지 않음"""
        # Given: retrieved_docs가 있어도 무시
        state: RAGState = {
            "question": "안녕?",
            "optimized_queries": ["query1"],
            "retrieved_docs": ["doc1", "doc2"],  # 무시되어야 함
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "안녕하세요!"

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: invoke_with_string_output의 input_data에 context가 없음
        call_args = mock_llm_service.invoke_with_string_output.call_args
        input_data = call_args.kwargs.get("input_data") or call_args[1].get("input_data", {})

        # context가 없거나 빈 값
        assert "context" not in input_data or input_data.get("context") in [None, "", []]

    @pytest.mark.unit
    def test_uses_generator_llm(self, mock_llm_service, sample_simple_question):
        """Generator LLM을 사용"""
        # Given
        state: RAGState = {
            "question": sample_simple_question,
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "답변"

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        node(state)

        # Then
        mock_llm_service.get_generator_llm.assert_called_once()


class TestSimpleGeneratorNodeScenarios:
    """SimpleGeneratorNode 시나리오 테스트"""

    @pytest.mark.unit
    def test_handles_greeting(self, mock_llm_service):
        """인사말 처리"""
        # Given
        state: RAGState = {
            "question": "안녕하세요!",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "안녕하세요! 반갑습니다."

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    @pytest.mark.unit
    def test_handles_coding_question(self, mock_llm_service):
        """코딩 질문 처리"""
        # Given
        state: RAGState = {
            "question": "파이썬 리스트 정렬 방법 알려줘",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "sorted() 함수나 .sort() 메서드를 사용하세요."

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_general_knowledge(self, mock_llm_service):
        """일반 상식 질문 처리"""
        # Given
        state: RAGState = {
            "question": "1+1은 몇이야?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "1+1은 2입니다."

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result


class TestSimpleGeneratorNodeEdgeCases:
    """SimpleGeneratorNode 엣지 케이스 테스트"""

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
        mock_llm_service.invoke_with_string_output.return_value = "질문을 입력해주세요."

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 에러 없이 처리
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_long_question(self, mock_llm_service):
        """긴 질문 처리"""
        # Given
        long_question = "이것은 매우 긴 질문입니다. " * 100
        state: RAGState = {
            "question": long_question,
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "답변입니다."

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 정상 처리
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_special_characters(self, mock_llm_service):
        """특수 문자 포함 질문 처리"""
        # Given
        state: RAGState = {
            "question": "파이썬에서 @decorator와 *args, **kwargs 사용법?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        mock_llm_service.invoke_with_string_output.return_value = "데코레이터는..."

        node = SimpleGeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 정상 처리
        assert "final_answer" in result


class TestSimpleGeneratorNodeIntegration:
    """SimpleGeneratorNode 통합 테스트 (실제 LLM 사용)"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_greeting_response(self, real_llm_service):
        """실제 LLM으로 인사 응답"""
        # Given
        state: RAGState = {
            "question": "안녕하세요!",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        node = SimpleGeneratorNode(real_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_coding_response(self, real_llm_service):
        """실제 LLM으로 코딩 질문 응답"""
        # Given
        state: RAGState = {
            "question": "파이썬에서 리스트를 정렬하는 방법은?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        node = SimpleGeneratorNode(real_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result
        # 정렬 관련 내용이 포함되어야 함
        answer = result["final_answer"].lower()
        assert "sort" in answer or "정렬" in answer or "sorted" in answer
