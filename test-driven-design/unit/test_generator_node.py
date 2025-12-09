"""
GeneratorNode 단위 테스트

검색된 문서를 기반으로 답변을 생성합니다.
- 입력: state["question"], state["retrieved_docs"]
- 출력: {"final_answer": str}
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.schemas import RAGState
from dev_v2.nodes import GeneratorNode


class TestGeneratorNodeProperties:
    """GeneratorNode 속성 테스트"""

    @pytest.mark.unit
    def test_node_name(self, mock_llm_service):
        """노드 이름이 'generator'인지 확인"""
        node = GeneratorNode(mock_llm_service)
        assert node.name == "generator"


class TestGeneratorNodeCall:
    """GeneratorNode __call__ 메서드 테스트"""

    @pytest.mark.unit
    def test_returns_final_answer(self, mock_llm_service, sample_rag_state_with_docs):
        """정상적으로 final_answer를 반환"""
        # Given
        expected_answer = "RAG 성능 고도화란 검색 증강 생성 시스템의 정확도를 높이는 기술입니다."
        mock_llm_service.invoke_with_string_output.return_value = expected_answer

        node = GeneratorNode(mock_llm_service)

        # When
        result = node(sample_rag_state_with_docs)

        # Then
        assert "final_answer" in result
        assert isinstance(result["final_answer"], str)
        assert result["final_answer"] == expected_answer

    @pytest.mark.unit
    def test_calls_generator_llm(self, mock_llm_service, sample_rag_state_with_docs):
        """Generator LLM을 사용하여 답변 생성"""
        # Given
        mock_llm_service.invoke_with_string_output.return_value = "답변"
        node = GeneratorNode(mock_llm_service)

        # When
        node(sample_rag_state_with_docs)

        # Then
        mock_llm_service.get_generator_llm.assert_called_once()
        mock_llm_service.invoke_with_string_output.assert_called_once()

    @pytest.mark.unit
    def test_uses_question_and_context(self, mock_llm_service, sample_rag_state_with_docs):
        """질문과 컨텍스트를 사용하여 답변 생성"""
        # Given
        mock_llm_service.invoke_with_string_output.return_value = "답변"
        node = GeneratorNode(mock_llm_service)

        # When
        node(sample_rag_state_with_docs)

        # Then: input_data에 question과 context가 포함
        call_args = mock_llm_service.invoke_with_string_output.call_args
        input_data = call_args.kwargs.get("input_data") or call_args[1].get("input_data", {})

        assert "question" in input_data
        assert "context" in input_data


class TestGeneratorNodeFormatDocs:
    """GeneratorNode _format_docs 메서드 테스트"""

    @pytest.mark.unit
    def test_format_docs_with_indexing(self, mock_llm_service, sample_documents):
        """문서가 [1], [2] 형식으로 인덱싱"""
        node = GeneratorNode(mock_llm_service)

        # When
        formatted = node._format_docs(sample_documents)

        # Then: [1], [2] 형식 포함
        assert "[1]" in formatted
        assert "[2]" in formatted

    @pytest.mark.unit
    def test_format_docs_preserves_content(self, mock_llm_service):
        """문서 내용이 보존됨"""
        node = GeneratorNode(mock_llm_service)
        docs = ["첫 번째 문서 내용", "두 번째 문서 내용"]

        # When
        formatted = node._format_docs(docs)

        # Then: 내용 포함
        assert "첫 번째 문서 내용" in formatted
        assert "두 번째 문서 내용" in formatted

    @pytest.mark.unit
    def test_format_docs_empty_list(self, mock_llm_service):
        """빈 문서 리스트 처리"""
        node = GeneratorNode(mock_llm_service)

        # When
        formatted = node._format_docs([])

        # Then: 빈 문자열 또는 "없음" 메시지
        assert formatted == "" or "없" in formatted


class TestGeneratorNodeEdgeCases:
    """GeneratorNode 엣지 케이스 테스트"""

    @pytest.mark.unit
    def test_handles_empty_docs(self, mock_llm_service):
        """빈 문서 리스트 처리"""
        # Given
        state: RAGState = {
            "question": "테스트 질문",
            "optimized_queries": [],
            "retrieved_docs": [],  # 빈 문서
            "final_answer": "",
        }

        mock_llm_service.invoke_with_string_output.return_value = "검색된 문서가 없어 답변을 생성할 수 없습니다."
        node = GeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 에러 메시지 또는 적절한 응답
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_long_documents(self, mock_llm_service):
        """긴 문서 처리"""
        # Given
        long_doc = "긴 문서 내용입니다. " * 500
        state: RAGState = {
            "question": "테스트 질문",
            "optimized_queries": [],
            "retrieved_docs": [long_doc],
            "final_answer": "",
        }

        mock_llm_service.invoke_with_string_output.return_value = "답변"
        node = GeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 정상 처리
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_special_characters_in_docs(self, mock_llm_service):
        """특수 문자 포함 문서 처리"""
        # Given
        state: RAGState = {
            "question": "테스트 질문",
            "optimized_queries": [],
            "retrieved_docs": [
                "문서에 <특수> 문자가 있습니다!@#$%",
                "JSON: {'key': 'value'}",
            ],
            "final_answer": "",
        }

        mock_llm_service.invoke_with_string_output.return_value = "답변"
        node = GeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 정상 처리
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_tuple_docs(self, mock_llm_service, sample_reranked_docs):
        """튜플 형태의 문서 처리 (리랭킹 결과)"""
        # Given: 리랭킹 결과는 (doc, score) 튜플
        state: RAGState = {
            "question": "테스트 질문",
            "optimized_queries": [],
            "retrieved_docs": sample_reranked_docs,  # List[Tuple[str, float]]
            "final_answer": "",
        }

        mock_llm_service.invoke_with_string_output.return_value = "답변"
        node = GeneratorNode(mock_llm_service)

        # When
        result = node(state)

        # Then: 정상 처리 (튜플에서 문서만 추출)
        assert "final_answer" in result


class TestGeneratorNodeIntegration:
    """GeneratorNode 통합 테스트 (실제 LLM 사용)"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_generation(self, real_llm_service):
        """실제 LLM으로 답변 생성"""
        # Given
        state: RAGState = {
            "question": "RAG란 무엇인가요?",
            "optimized_queries": [],
            "retrieved_docs": [
                "RAG(Retrieval-Augmented Generation)은 검색과 생성을 결합한 AI 기술입니다.",
                "RAG는 외부 데이터베이스에서 관련 정보를 검색하여 LLM의 답변을 보강합니다.",
            ],
            "final_answer": "",
        }

        node = GeneratorNode(real_llm_service)

        # When
        result = node(state)

        # Then
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0
        # RAG 관련 내용이 포함되어야 함
        assert "RAG" in result["final_answer"] or "검색" in result["final_answer"]
