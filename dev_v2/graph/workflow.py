"""
RAG Workflow - LangGraph 기반 워크플로우 구성
"""
from langgraph.graph import StateGraph, END

from ..schemas import RAGState
from ..nodes import QueryRewriteNode, RetrieverNode, GeneratorNode


class RAGWorkflow:
    """RAG 워크플로우 클래스"""

    def __init__(
        self,
        query_rewrite_node: QueryRewriteNode,
        retriever_node: RetrieverNode,
        generator_node: GeneratorNode,
    ):
        self._query_rewrite_node = query_rewrite_node
        self._retriever_node = retriever_node
        self._generator_node = generator_node
        self._app = None

    def build(self) -> "RAGWorkflow":
        """워크플로우 빌드"""
        workflow = StateGraph(RAGState)

        # 노드 추가
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

        # 엣지 연결 (순서 정의)
        workflow.set_entry_point(self._query_rewrite_node.name)
        workflow.add_edge(
            self._query_rewrite_node.name,
            self._retriever_node.name,
        )
        workflow.add_edge(
            self._retriever_node.name,
            self._generator_node.name,
        )
        workflow.add_edge(self._generator_node.name, END)

        # 컴파일
        self._app = workflow.compile()

        return self

    @property
    def app(self):
        """컴파일된 앱 반환"""
        if self._app is None:
            raise RuntimeError("Workflow가 빌드되지 않았습니다. build()를 먼저 호출하세요.")
        return self._app

    def invoke(self, question: str) -> dict:
        """워크플로우 실행"""
        initial_state = {
            "question": question,
            "optimized_queries": [],
            "retrieved_docs": [],
            "final_answer": "",
        }
        return self.app.invoke(initial_state)
