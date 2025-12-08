"""
Base Node - 모든 노드의 기본 인터페이스
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..schemas import RAGState


class BaseNode(ABC):
    """노드 기본 클래스 (추상 클래스)"""

    @property
    @abstractmethod
    def name(self) -> str:
        """노드 이름"""
        pass

    @abstractmethod
    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """
        노드 실행

        Args:
            state: 현재 RAG 상태

        Returns:
            업데이트할 상태 딕셔너리
        """
        pass
