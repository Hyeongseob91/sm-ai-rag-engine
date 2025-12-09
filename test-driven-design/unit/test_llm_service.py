"""
LLMService 단위 테스트

LLM 인스턴스 생성 및 호출을 추상화하는 서비스 클래스입니다.
- get_rewrite_llm(): Query Rewrite용 LLM (gpt-4o-mini)
- get_generator_llm(): Generator용 LLM (gpt-4o)
- invoke_with_structured_output(): 구조화된 출력
- invoke_with_string_output(): 문자열 출력
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.services import LLMService
from dev_v2.schemas import RewriteResult, RouteQuery


class TestLLMServiceFactoryMethods:
    """LLM Factory 메서드 테스트"""

    @pytest.mark.unit
    def test_get_rewrite_llm_returns_chatopeanai(self, mock_settings):
        """get_rewrite_llm()이 ChatOpenAI 인스턴스 반환"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)
            llm = service.get_rewrite_llm()

            assert isinstance(llm, ChatOpenAI)

    @pytest.mark.unit
    def test_get_rewrite_llm_uses_correct_model(self, mock_settings):
        """get_rewrite_llm()이 올바른 모델 사용"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)
            llm = service.get_rewrite_llm()

            assert llm.model_name == mock_settings.llm.rewrite_model

    @pytest.mark.unit
    def test_get_rewrite_llm_uses_correct_temperature(self, mock_settings):
        """get_rewrite_llm()이 올바른 temperature 사용"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)
            llm = service.get_rewrite_llm()

            assert llm.temperature == mock_settings.llm.rewrite_temperature

    @pytest.mark.unit
    def test_get_generator_llm_returns_chatopeanai(self, mock_settings):
        """get_generator_llm()이 ChatOpenAI 인스턴스 반환"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)
            llm = service.get_generator_llm()

            assert isinstance(llm, ChatOpenAI)

    @pytest.mark.unit
    def test_get_generator_llm_uses_correct_model(self, mock_settings):
        """get_generator_llm()이 올바른 모델 사용"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)
            llm = service.get_generator_llm()

            assert llm.model_name == mock_settings.llm.generator_model

    @pytest.mark.unit
    def test_different_llms_for_different_purposes(self, mock_settings):
        """rewrite와 generator가 다른 LLM 인스턴스 사용"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            # 다른 모델명 설정
            mock_settings.llm.rewrite_model = "gpt-4o-mini"
            mock_settings.llm.generator_model = "gpt-4o"

            service = LLMService(mock_settings)
            rewrite_llm = service.get_rewrite_llm()
            generator_llm = service.get_generator_llm()

            assert rewrite_llm.model_name != generator_llm.model_name


class TestLLMServiceInvokeStructuredOutput:
    """invoke_with_structured_output() 테스트"""

    @pytest.mark.unit
    @patch("dev_v2.services.llm.ChatOpenAI")
    def test_returns_pydantic_model(self, mock_chat_openai, mock_settings):
        """Pydantic 모델 반환"""
        # Mock 설정
        mock_llm_instance = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        expected_result = RewriteResult(queries=["query1", "query2", "query3"])
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_result
        mock_structured_llm.__or__ = MagicMock(return_value=mock_chain)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)

            # Mock prompt
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            # When
            result = service.invoke_with_structured_output(
                llm=mock_llm_instance,
                prompt=mock_prompt,
                output_schema=RewriteResult,
                input_data={"question": "test"},
            )

            # Then
            assert isinstance(result, RewriteResult)

    @pytest.mark.unit
    def test_structured_output_schema_validation(self, mock_settings):
        """출력 스키마 검증"""
        # RewriteResult 스키마 검증
        result = RewriteResult(queries=["q1", "q2"])
        assert hasattr(result, "queries")
        assert isinstance(result.queries, list)

        # RouteQuery 스키마 검증
        route = RouteQuery(datasource="vectorstore")
        assert hasattr(route, "datasource")
        assert route.datasource in ["vectorstore", "llm"]


class TestLLMServiceInvokeStringOutput:
    """invoke_with_string_output() 테스트"""

    @pytest.mark.unit
    def test_invoke_with_string_output_calls_chain(self, mock_settings):
        """invoke_with_string_output이 체인을 호출하는지 확인"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = LLMService(mock_settings)

            # Mock 설정: LCEL 체인 전체를 패치
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "테스트 답변입니다."

            with patch.object(service, 'invoke_with_string_output', return_value="테스트 답변입니다.") as mock_method:
                # When
                result = service.invoke_with_string_output(
                    llm=MagicMock(),
                    prompt=MagicMock(),
                    input_data={"question": "test"},
                )

                # Then
                assert result == "테스트 답변입니다."
                mock_method.assert_called_once()


class TestLLMServiceIntegration:
    """LLMService 통합 테스트 (실제 API 호출)"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_structured_output(self, real_llm_service):
        """실제 LLM으로 구조화된 출력 테스트"""
        from langchain_core.prompts import ChatPromptTemplate

        llm = real_llm_service.get_rewrite_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 질문을 검색에 최적화된 3개의 쿼리로 변환하세요."),
            ("human", "{question}"),
        ])

        result = real_llm_service.invoke_with_structured_output(
            llm=llm,
            prompt=prompt,
            output_schema=RewriteResult,
            input_data={"question": "RAG 성능 고도화란?"},
        )

        assert isinstance(result, RewriteResult)
        assert len(result.queries) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_string_output(self, real_llm_service):
        """실제 LLM으로 문자열 출력 테스트"""
        from langchain_core.prompts import ChatPromptTemplate

        llm = real_llm_service.get_generator_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "간단히 인사해주세요."),
            ("human", "{question}"),
        ])

        result = real_llm_service.invoke_with_string_output(
            llm=llm,
            prompt=prompt,
            input_data={"question": "안녕하세요"},
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_route_query(self, real_llm_service):
        """실제 LLM으로 라우팅 테스트"""
        from langchain_core.prompts import ChatPromptTemplate
        from dev_v2.prompts import ROUTER_SYSTEM_PROMPT

        llm = real_llm_service.get_rewrite_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # 일반 대화 질문
        result = real_llm_service.invoke_with_structured_output(
            llm=llm,
            prompt=prompt,
            output_schema=RouteQuery,
            input_data={"question": "안녕하세요"},
        )

        assert isinstance(result, RouteQuery)
        assert result.datasource == "llm"
