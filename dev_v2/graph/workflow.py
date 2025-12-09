"""
RAG Workflow - LangGraph 기반 워크플로우 구성 (Router Pattern 적용)
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ..schemas import RAGState, RouteQuery
from ..nodes import QueryRewriteNode, RetrieverNode, GeneratorNode, SimpleGeneratorNode
from ..services import LLMService
from ..prompts import ROUTER_SYSTEM_PROMPT


class RAGWorkflow:
    """RAG 워크플로우 클래스 (Router Pattern 적용)"""

    def __init__(
        self,
        llm_service: LLMService,
        query_rewrite_node: QueryRewriteNode,
        retriever_node: RetrieverNode,
        generator_node: GeneratorNode,
        simple_generator_node: SimpleGeneratorNode,
    ):
        """RAG 워크플로우 초기화 (라우팅 기능 포함)"""
        self._llm_service = llm_service
        self._query_rewrite_node = query_rewrite_node
        self._retriever_node = retriever_node
        self._generator_node = generator_node
        self._simple_generator_node = simple_generator_node
        self._app = None

    def route_question(self, state: RAGState) -> Literal["query_rewrite", "simple_generator"]:
        """
        질문을 분석하여 다음 경로를 결정하는 조건부 함수 (Router)

        Returns:
            - "query_rewrite": RAG 검색이 필요한 경우
            - "simple_generator": 일반 대화로 바로 응답하는 경우
        """
        print(f"--- [Router] 질문 분석 중: {state['question']} ---")

        llm = self._llm_service.get_rewrite_llm()  # 가벼운 모델 사용

        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        decision = self._llm_service.invoke_with_structured_output(
            llm=llm,
            prompt=prompt,
            output_schema=RouteQuery,
            input_data={"question": state["question"]}
        )

        if decision.datasource == "vectorstore":
            print("--- [Decision] RAG 검색 진행 (VectorStore) ---")
            return self._query_rewrite_node.name
        else:
            print("--- [Decision] 일반 대화 진행 (LLM) ---")
            return self._simple_generator_node.name

    def build(self) -> "RAGWorkflow":
        """StateGraph를 생성하고 조건부 라우팅을 적용하여 워크플로우를 빌드합니다."""
        workflow = StateGraph(RAGState)

        # 1. 노드 등록
        workflow.add_node(
            self._query_rewrite_node.name,
            self._query_rewrite_node,
        )
        workflow.add_node(
            self._retriever_node.name,
            self._retriever_node,
        )
        workflow.add_node(
            self._generator_node.name,
            self._generator_node,
        )
        workflow.add_node(
            self._simple_generator_node.name,
            self._simple_generator_node,
        )

        # 2. 조건부 시작점 설정 (Router Pattern)
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                self._query_rewrite_node.name: self._query_rewrite_node.name,
                self._simple_generator_node.name: self._simple_generator_node.name,
            }
        )

        # 3. RAG 파이프라인 연결
        workflow.add_edge(
            self._query_rewrite_node.name,
            self._retriever_node.name,
        )
        workflow.add_edge(
            self._retriever_node.name,
            self._generator_node.name,
        )
        workflow.add_edge(self._generator_node.name, END)

        # 4. 단순 대화 파이프라인 연결
        workflow.add_edge(self._simple_generator_node.name, END)

        # 컴파일
        self._app = workflow.compile()

        return self

    @property
    def app(self):
        """빌드된 LangGraph 워크플로우 앱을 반환하며, 아직 빌드되지 않았으면 RuntimeError를 발생시킵니다."""
        if self._app is None:
            raise RuntimeError("Workflow가 빌드되지 않았습니다. build()를 먼저 호출하세요.")
        return self._app

    def invoke(self, question: str) -> dict:
        """사용자 질문을 받아서 초기 상태를 설정하고 워크플로우를 실행하여 질문 재작성, 문서 검색, 답변 생성 과정을 거쳐 최종 답변을 포함한 상태 딕셔너리를 반환합니다."""
        initial_state = {
            "question": question,
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        return self.app.invoke(initial_state)
