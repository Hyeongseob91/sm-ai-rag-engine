"""
Reranker Service - Cross-Encoder Reranking 기능 제공
"""
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from ..config import Settings


class RerankerService:
    """CrossEncoder Reranker 서비스 클래스"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: CrossEncoder = None

    @property
    def model(self) -> CrossEncoder:
        """Reranker 모델 (Lazy Loading)"""
        if self._model is None:
            self._model = CrossEncoder(self.settings.reranker.model_name)
        return self._model

    def rerank(self, 
        query: str, 
        documents: List[str], 
        top_k: int = None,
        ) -> List[Tuple[str, float]]:
        """
        문서들을 Query와의 관련성으로 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            (문서 내용, 점수) 튜플 리스트 (점수 내림차순)
        """
        if top_k is None:
            top_k = self.settings.reranker.top_k

        # Query-Document 쌍 생성
        pairs = [[query, doc] for doc in documents]

        # 점수 예측
        scores = self.model.predict(pairs)

        # 점수 기준 정렬
        ranked_results = sorted(
            list(zip(documents, scores)),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked_results[:top_k]

    def get_top_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> List[str]:
        """상위 문서 내용만 반환"""
        ranked = self.rerank(query, documents, top_k)

        return ranked
