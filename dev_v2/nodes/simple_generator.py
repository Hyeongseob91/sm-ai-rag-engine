"""
Simple Generator Node - 검색 없이 바로 답변 생성
"""
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .base import BaseNode
from ..schemas import RAGState
from ..services import LLMService


class SimpleGeneratorNode(BaseNode):
    """검색 없이 LLM 지식으로 바로 답변하는 노드"""

    def __init__(self, llm_service: LLMService):
        self._llm_service = llm_service
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 친절하고 정확하게 답변하세요."),
            ("human", "{question}"),
        ])

    @property
    def name(self) -> str:
        return "simple_generator"

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        print("--- [Route: LLM] 검색 없이 즉시 답변 생성 ---")

        llm = self._llm_service.get_generator_llm()
        answer = self._llm_service.invoke_with_string_output(
            llm=llm,
            prompt=self._prompt,
            input_data={"question": state["question"]},
        )

        return {"final_answer": answer}
