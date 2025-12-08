"""
LLM Service - Language Model 관련 기능 제공
"""
from typing import Type, TypeVar
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from ..config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """LLM 호출 관련 서비스 클래스"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def get_rewrite_llm(self) -> ChatOpenAI:
        """Query Rewrite용 LLM 인스턴스 반환"""
        return ChatOpenAI(
            model=self.settings.llm.rewrite_model,
            temperature=self.settings.llm.rewrite_temperature,
        )

    def get_generator_llm(self) -> ChatOpenAI:
        """Generator용 LLM 인스턴스 반환"""
        return ChatOpenAI(
            model=self.settings.llm.generator_model,
            temperature=self.settings.llm.generator_temperature,
        )

    def invoke_with_structured_output(
        self,
        llm: ChatOpenAI,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        input_data: dict,
    ) -> T:
        """Structured Output으로 LLM 호출"""
        structured_llm = llm.with_structured_output(output_schema)
        chain = prompt | structured_llm
        return chain.invoke(input_data)

    def invoke_with_string_output(
        self,
        llm: ChatOpenAI,
        prompt: ChatPromptTemplate,
        input_data: dict,
    ) -> str:
        """String Output으로 LLM 호출"""
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(input_data)
