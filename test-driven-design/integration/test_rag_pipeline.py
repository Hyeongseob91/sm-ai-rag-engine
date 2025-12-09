"""
RAG Pipeline 통합 테스트

전체 파이프라인 흐름을 테스트합니다.
- RAG 경로: Router → QueryRewrite → Retriever → Generator
- LLM 경로: Router → SimpleGenerator
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.schemas import RAGState, RouteQuery, RewriteResult
from dev_v2.graph import RAGWorkflow
from dev_v2.nodes import QueryRewriteNode, RetrieverNode, GeneratorNode, SimpleGeneratorNode


class TestRAGPipelineUnit:
    """RAG Pipeline 단위 테스트 (Mock 사용)"""

    @pytest.mark.unit
    def test_full_rag_path_mock(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_vectorstore,
        sample_rewrite_result,
        sample_documents,
        sample_reranked_docs,
    ):
        """RAG 전체 경로 테스트 (Mock)"""
        # Given: 모든 서비스 Mock 설정
        # Router: vectorstore로 라우팅
        mock_llm_service.invoke_with_structured_output.side_effect = [
            sample_route_query_vectorstore,  # Router
            sample_rewrite_result,  # QueryRewrite
        ]

        # Retriever
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs

        # Generator
        mock_llm_service.invoke_with_string_output.return_value = "RAG 기반 답변입니다."

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
        workflow.build()

        # When
        result = workflow.invoke("RAG 성능 고도화란?")

        # Then: 최종 답변 확인
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    @pytest.mark.unit
    def test_llm_path_mock(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_llm,
    ):
        """LLM 직접 응답 경로 테스트 (Mock)"""
        # Given: llm으로 라우팅
        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_llm
        mock_llm_service.invoke_with_string_output.return_value = "안녕하세요! 반갑습니다."

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
        workflow.build()

        # When
        result = workflow.invoke("안녕?")

        # Then: SimpleGenerator 경로로 답변
        assert "final_answer" in result
        # RAG 노드들이 호출되지 않아야 함
        mock_vectorstore_service.hybrid_search.assert_not_called()


class TestRAGPipelineIntegration:
    """RAG Pipeline 통합 테스트 (실제 서비스 사용)"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_greeting_uses_llm_path(
        self,
        real_llm_service,
        real_reranker_service,
        real_vectorstore_service,
    ):
        """인사말은 LLM 경로 사용"""
        # Given
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
        workflow.build()

        # When
        result = workflow.invoke("안녕하세요!")

        # Then
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0
        # 검색이 실행되지 않았으므로 retrieved_docs가 비어있어야 함
        assert result["retrieved_docs"] == [] or len(result["retrieved_docs"]) == 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_domain_question_uses_rag_path(
        self,
        real_llm_service,
        real_reranker_service,
        real_vectorstore_service,
    ):
        """도메인 질문은 RAG 경로 사용"""
        # Given
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
        workflow.build()

        # When
        result = workflow.invoke("회사의 환불 규정이 어떻게 되나요?")

        # Then
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0
        # QueryRewrite가 실행되었으므로 optimized_queries가 있어야 함
        assert len(result["optimized_queries"]) > 0


class TestRAGPipelineDataFlow:
    """RAG Pipeline 데이터 플로우 테스트"""

    @pytest.mark.unit
    def test_state_transitions_rag_path(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_vectorstore,
        sample_rewrite_result,
        sample_documents,
        sample_reranked_docs,
    ):
        """RAG 경로의 상태 전이 확인"""
        # Given
        mock_llm_service.invoke_with_structured_output.side_effect = [
            sample_route_query_vectorstore,
            sample_rewrite_result,
        ]
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs
        mock_llm_service.invoke_with_string_output.return_value = "최종 답변"

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
        workflow.build()

        # When
        result = workflow.invoke("테스트 질문")

        # Then: 모든 상태가 채워짐
        assert result["question"] == "테스트 질문"
        assert len(result["optimized_queries"]) > 0
        assert len(result["retrieved_docs"]) > 0
        assert len(result["final_answer"]) > 0

    @pytest.mark.unit
    def test_state_transitions_llm_path(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_llm,
    ):
        """LLM 경로의 상태 전이 확인"""
        # Given
        mock_llm_service.invoke_with_structured_output.return_value = sample_route_query_llm
        mock_llm_service.invoke_with_string_output.return_value = "간단한 답변"

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
        workflow.build()

        # When
        result = workflow.invoke("안녕")

        # Then: question과 final_answer만 채워짐 (검색 생략)
        assert result["question"] == "안녕"
        assert result["optimized_queries"] == []
        assert result["retrieved_docs"] == []
        assert len(result["final_answer"]) > 0


class TestRAGPipelineEdgeCases:
    """RAG Pipeline 엣지 케이스 테스트"""

    @pytest.mark.unit
    def test_handles_empty_search_results(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_vectorstore,
        sample_rewrite_result,
    ):
        """검색 결과 없음 처리"""
        # Given: 검색 결과 없음
        mock_llm_service.invoke_with_structured_output.side_effect = [
            sample_route_query_vectorstore,
            sample_rewrite_result,
        ]
        mock_vectorstore_service.hybrid_search.return_value = []
        mock_reranker_service.get_top_documents.return_value = []
        mock_llm_service.invoke_with_string_output.return_value = "검색된 문서가 없습니다."

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
        workflow.build()

        # When
        result = workflow.invoke("없는 내용에 대한 질문")

        # Then: 에러 없이 처리
        assert "final_answer" in result

    @pytest.mark.unit
    def test_handles_query_rewrite_failure(
        self,
        mock_llm_service,
        mock_vectorstore_service,
        mock_reranker_service,
        sample_route_query_vectorstore,
        sample_documents,
        sample_reranked_docs,
    ):
        """QueryRewrite 실패 시 Fallback"""
        # Given: QueryRewrite 실패
        mock_llm_service.invoke_with_structured_output.side_effect = [
            sample_route_query_vectorstore,
            Exception("LLM Error"),  # QueryRewrite 실패
        ]
        mock_vectorstore_service.hybrid_search.return_value = sample_documents
        mock_reranker_service.get_top_documents.return_value = sample_reranked_docs
        mock_llm_service.invoke_with_string_output.return_value = "답변"

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
        workflow.build()

        # When
        result = workflow.invoke("테스트 질문")

        # Then: Fallback으로 원본 질문 사용, 최종 답변 생성
        assert "final_answer" in result
