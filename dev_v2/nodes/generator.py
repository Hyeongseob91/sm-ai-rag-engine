"""
Generator Node - 검색된 문서 기반 답변 생성
"""
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from .base import BaseNode
from ..schemas import RAGState
from ..services import LLMService
from ..prompts import GENERATOR_PROMPT_TEMPLATE


class GeneratorNode(BaseNode):
    """Generator 노드 클래스"""

    def __init__(self, llm_service: LLMService):
        self._llm_service = llm_service
        self._prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT_TEMPLATE)

    @property
    def name(self) -> str:
        return "generator"

    def _format_docs(self, docs: List[str]) -> str:
        """문서 리스트를 번호가 매겨진 텍스트로 변환"""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"[{i + 1}] {doc}")
        return "\n\n".join(formatted)

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """
        검색된 문서를 기반으로 답변 생성
        """
        print("--- [Step 3] Generator 시작 ---")

        question = state["question"]
        docs = state.get("retrieved_docs", [])

        if not docs:
            return {"final_answer": "검색된 문서가 없어 답변을 생성할 수 없습니다."}

        # 문서 포맷팅
        context_str = self._format_docs(docs)

        # LLM 호출
        llm = self._llm_service.get_generator_llm()
        response = self._llm_service.invoke_with_string_output(
            llm=llm,
            prompt=self._prompt,
            input_data={
                "question": question,
                "context": context_str,
            },
        )

        print(f"--- 생성 완료: {response[:50]}... ---")

        return {"final_answer": response}
