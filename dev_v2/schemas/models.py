"""
Pydantic Models for Structured Output
"""
from typing import List
from pydantic import BaseModel, Field


class RewriteResult(BaseModel):
    """Query Rewrite 결과를 위한 Pydantic 모델"""
    queries: List[str] = Field(
        description="검색 엔진에 최적화된, 3개 내외의 재작성된 쿼리 리스트"
    )
