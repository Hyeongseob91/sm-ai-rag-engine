"""
Preprocessing Schemas - 전처리 관련 데이터 모델
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
import uuid


@dataclass
class RawDocument:
    """파서에서 추출된 원본 문서 데이터"""

    content: str  # 추출된 전체 텍스트
    source: str  # 파일 경로
    file_type: str  # 파일 확장자 (pdf, docx, xlsx, txt, json)
    file_name: str  # 파일 이름
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: Optional[List[str]] = None  # 페이지별 텍스트 (PDF, DOCX)
    sheets: Optional[Dict[str, str]] = None  # 시트별 텍스트 (XLSX)

    def __post_init__(self):
        if self.pages is None:
            self.pages = []
        if self.sheets is None:
            self.sheets = {}

    def __repr__(self):
        return (
            f"RawDocument(\n"
            f"  file_name='{self.file_name}',\n"
            f"  file_type='{self.file_type}',\n"
            f"  content_length={len(self.content)},\n"
            f"  pages={len(self.pages)},\n"
            f"  sheets={list(self.sheets.keys()) if self.sheets else []},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )


@dataclass
class Chunk:
    """청킹된 텍스트 단위"""

    content: str  # 청크 텍스트
    chunk_id: str  # 고유 ID (UUID)
    chunk_index: int  # 문서 내 순서
    doc_id: str  # 원본 문서 ID
    source: str  # 원본 파일 경로
    file_name: str  # 파일 이름
    file_type: str  # 파일 형식
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """일반 딕셔너리 변환"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "source": self.source,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "char_count": self.char_count,
            **self.metadata,
        }

    def to_weaviate_object(self) -> Dict[str, Any]:
        """Weaviate 저장용 딕셔너리 변환"""
        # Weaviate는 RFC3339 형식 필요 (타임존 포함)
        created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.metadata.get("total_chunks", 1),
            "source": self.source,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "char_count": self.char_count,
            "page_number": self.metadata.get("page_number"),
            "sheet_name": self.metadata.get("sheet_name"),
            "created_at": created_at,
        }


class DocumentMetadata(BaseModel):
    """문서 메타데이터 스키마 (Pydantic v2)"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    doc_id: str = Field(description="문서 고유 ID")
    source: str = Field(description="파일 경로")
    file_name: str = Field(description="파일 이름")
    file_type: str = Field(description="파일 형식")
    file_size: int = Field(default=0, description="파일 크기 (bytes)")
    created_at: datetime = Field(default_factory=datetime.now)
    page_count: Optional[int] = Field(default=None, description="페이지 수 (PDF, DOCX)")
    sheet_count: Optional[int] = Field(default=None, description="시트 수 (XLSX)")


class ChunkMetadata(BaseModel):
    """청크 메타데이터 스키마 (Pydantic v2)"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    chunk_id: str = Field(description="청크 고유 ID")
    doc_id: str = Field(description="원본 문서 ID")
    chunk_index: int = Field(description="문서 내 순서")
    total_chunks: int = Field(description="문서의 전체 청크 수")
    source: str = Field(description="원본 파일 경로")
    file_name: str = Field(description="파일 이름")
    file_type: str = Field(description="파일 형식")
    char_count: int = Field(description="청크 글자 수")
    page_number: Optional[int] = Field(default=None, description="페이지 번호 (PDF)")
    sheet_name: Optional[str] = Field(default=None, description="시트 이름 (XLSX)")
    created_at: datetime = Field(default_factory=datetime.now)


@dataclass
class PreprocessingResult:
    """전처리 결과"""

    document: Optional[RawDocument]
    chunks: List[Chunk]
    metadata: Optional[DocumentMetadata]
    stats: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
