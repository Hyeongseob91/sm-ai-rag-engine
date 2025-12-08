"""
Retriever Node - 문서 검색 및 Reranking
"""
from typing import Dict, Any, List

from .base import BaseNode
from ..schemas import RAGState
from ..services import VectorStoreService, RerankerService


class RetrieverNode(BaseNode):
    """Retriever 노드 클래스"""

    def __init__(
        self,
        vectorstore_service: VectorStoreService,
        reranker_service: RerankerService,
    ):
        self._vectorstore = vectorstore_service
        self._reranker = reranker_service

    @property
    def name(self) -> str:
        return "retriever"

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """
        최적화된 쿼리들로 문서를 검색하고 Reranking 수행
        """
        print(f"--- [Step 2] Retriever 시작 ---")

        optimized_queries = state.get("optimized_queries", [state["question"]])

        # 모든 쿼리로 검색하여 결과 수집
        all_results: List[str] = []
        seen_contents = set()

        for query in optimized_queries:
            print(f"    검색 쿼리: {query}")
            results = self._vectorstore.hybrid_search(query)

            # 중복 제거하며 추가
            for content in results:
                if content not in seen_contents:
                    seen_contents.add(content)
                    all_results.append(content)

        print(f"--- 1차 검색 결과: {len(all_results)}개 문서 ---")

        # Reranking (원본 질문 기준)
        original_question = state["question"]
        final_docs = self._reranker.get_top_documents(
            query=original_question,
            documents=all_results,
        )

        print(f"--- Reranking 후: {len(final_docs)}개 문서 선별 ---")

        return {"retrieved_docs": final_docs}
