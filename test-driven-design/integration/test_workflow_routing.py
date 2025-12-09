"""
Router 테스트 - route_question() 메서드 검증

데이터플로우 진입점으로, 질문을 분류하여 적절한 경로로 라우팅합니다.
- "vectorstore": RAG 파이프라인 (QueryRewrite → Retriever → Generator)
- "llm": 단순 LLM 응답 (SimpleGenerator)
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.schemas import RAGState, RouteQuery
from dev_v2.graph import RAGWorkflow
from dev_v2.nodes import QueryRewriteNode, RetrieverNode, GeneratorNode, SimpleGeneratorNode


# =============================================================================
# Unit Tests (Mock 기반)
# =============================================================================
class TestRouteQuestionUnit:
    """route_question() 단위 테스트 - Mock 사용"""

    @pytest.mark.unit
    def test_route_to_vectorstore_for_domain_question(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_vectorstore,
    ):
        """도메인 질문은 vectorstore로 라우팅"""
        # Given: RAG 관련 질문
        state: RAGState = {
            "question": "환불 규정을 알려주세요",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        # Mock 설정: invoke_with_structured_output이 vectorstore 반환
        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_vectorstore

        # 노드 생성 (Mock 서비스 주입)
        query_rewrite_node = QueryRewriteNode(mock_llm_service)
        retriever_node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        generator_node = GeneratorNode(mock_llm_service)
        simple_generator_node = SimpleGeneratorNode(mock_llm_service)

        # Workflow 생성
        workflow = RAGWorkflow(
            mock_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When: route_question 호출
        result = workflow.route_question(state)

        # Then: query_rewrite 노드로 라우팅
        assert result == "query_rewrite"
        mock_llm_service.invoke_with_structured_output.assert_called_once()

    @pytest.mark.unit
    def test_route_to_llm_for_simple_question(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_llm,
    ):
        """일반 대화 질문은 llm으로 라우팅"""
        # Given: 인사 질문
        state: RAGState = {
            "question": "안녕하세요",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        # Mock 설정: invoke_with_structured_output이 llm 반환
        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_llm

        # 노드 생성
        query_rewrite_node = QueryRewriteNode(mock_llm_service)
        retriever_node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        generator_node = GeneratorNode(mock_llm_service)
        simple_generator_node = SimpleGeneratorNode(mock_llm_service)

        workflow = RAGWorkflow(
            mock_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then: simple_generator 노드로 라우팅
        assert result == "simple_generator"

    @pytest.mark.unit
    def test_route_to_llm_for_coding_question(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_llm,
    ):
        """코딩 질문은 llm으로 라우팅"""
        # Given: 코딩 관련 질문
        state: RAGState = {
            "question": "파이썬 리스트 정렬 방법 알려줘",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_llm

        query_rewrite_node = QueryRewriteNode(mock_llm_service)
        retriever_node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        generator_node = GeneratorNode(mock_llm_service)
        simple_generator_node = SimpleGeneratorNode(mock_llm_service)

        workflow = RAGWorkflow(
            mock_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then
        assert result == "simple_generator"


# =============================================================================
# Integration Tests (실제 LLM 호출)
# =============================================================================
class TestRouteQuestionIntegration:
    """route_question() 통합 테스트 - 실제 LLM 사용"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_route_greeting_to_llm(self, real_llm_service, real_reranker_service, real_vectorstore_service):
        """인사말은 LLM 경로로 라우팅 (실제 API 호출)"""
        # Given
        state: RAGState = {
            "question": "안녕?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        query_rewrite_node = QueryRewriteNode(real_llm_service)
        retriever_node = RetrieverNode(real_vectorstore_service, real_reranker_service)
        generator_node = GeneratorNode(real_llm_service)
        simple_generator_node = SimpleGeneratorNode(real_llm_service)

        workflow = RAGWorkflow(
            real_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then: 인사말은 llm으로 라우팅
        assert result == "simple_generator"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_route_domain_question_to_vectorstore(
        self, real_llm_service, real_reranker_service, real_vectorstore_service
    ):
        """도메인 질문은 vectorstore 경로로 라우팅 (실제 API 호출)"""
        # Given
        state: RAGState = {
            "question": "회사의 환불 규정이 어떻게 되나요?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        query_rewrite_node = QueryRewriteNode(real_llm_service)
        retriever_node = RetrieverNode(real_vectorstore_service, real_reranker_service)
        generator_node = GeneratorNode(real_llm_service)
        simple_generator_node = SimpleGeneratorNode(real_llm_service)

        workflow = RAGWorkflow(
            real_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then: 도메인 질문은 vectorstore로 라우팅
        assert result == "query_rewrite"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_route_general_knowledge_to_llm(
        self, real_llm_service, real_reranker_service, real_vectorstore_service
    ):
        """일반 상식 질문은 LLM 경로로 라우팅"""
        # Given
        state: RAGState = {
            "question": "1+1은 몇이야?",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        query_rewrite_node = QueryRewriteNode(real_llm_service)
        retriever_node = RetrieverNode(real_vectorstore_service, real_reranker_service)
        generator_node = GeneratorNode(real_llm_service)
        simple_generator_node = SimpleGeneratorNode(real_llm_service)

        workflow = RAGWorkflow(
            real_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then
        assert result == "simple_generator"


# =============================================================================
# Edge Cases
# =============================================================================
class TestRouteQuestionEdgeCases:
    """route_question() 엣지 케이스 테스트"""

    @pytest.mark.unit
    def test_route_empty_question(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_llm,
    ):
        """빈 질문 처리"""
        state: RAGState = {
            "question": "",
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_llm

        query_rewrite_node = QueryRewriteNode(mock_llm_service)
        retriever_node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        generator_node = GeneratorNode(mock_llm_service)
        simple_generator_node = SimpleGeneratorNode(mock_llm_service)

        workflow = RAGWorkflow(
            mock_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then: 빈 질문은 LLM으로 처리
        assert result in ["simple_generator", "query_rewrite"]

    @pytest.mark.unit
    def test_route_long_question(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_vectorstore,
    ):
        """긴 질문 처리"""
        long_question = "RAG 시스템의 성능을 최적화하기 위한 " * 20 + "방법은?"

        state: RAGState = {
            "question": long_question,
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }

        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_vectorstore

        query_rewrite_node = QueryRewriteNode(mock_llm_service)
        retriever_node = RetrieverNode(mock_vectorstore_service, mock_reranker_service)
        generator_node = GeneratorNode(mock_llm_service)
        simple_generator_node = SimpleGeneratorNode(mock_llm_service)

        workflow = RAGWorkflow(
            mock_llm_service,
            query_rewrite_node,
            retriever_node,
            generator_node,
            simple_generator_node,
        )

        # When
        result = workflow.route_question(state)

        # Then: 정상 처리
        assert result in ["simple_generator", "query_rewrite"]
