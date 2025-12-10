"""
Preprocessing Pipeline - 전체 처리 흐름 통합.

파일 파싱, 텍스트 정규화, 시맨틱 청킹을 순차적으로 수행하여 RAG 시스템에 적합한 청크 단위로 변환합니다.
"""
import uuid
from pathlib import Path
from typing import List

from ...config import Settings
from ...schemas.preprocessing import (
    Chunk,
    DocumentMetadata,
    PreprocessingResult,
    RawDocument,
)
from .chunking import ChunkingService
from .normalizer import TextNormalizer
from .parsers import UnifiedFileParser


class PreprocessingPipeline:
    """전처리 파이프라인 - 파일 -> 청크 -> 메타데이터 전체 처리.

    UnifiedFileParser, TextNormalizer, ChunkingService를 조합하여 단일 파일 또는 디렉토리 전체를 처리합니다.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._parser = UnifiedFileParser()
        self._normalizer = TextNormalizer(settings)
        self._chunking_service = ChunkingService(settings)

    def process_file(self, file_path: str) -> PreprocessingResult:
        """단일 파일 전처리"""
        try:
            # 1. 파싱
            raw_doc = self._parser.parse(file_path)

            # 2. 정규화
            normalized_doc = self._normalizer.normalize_document(raw_doc)

            # 3. 청킹
            chunks = self._chunking_service.chunk_document(normalized_doc)

            # 4. 청크 메타데이터 보강
            total_chunks = len(chunks)
            for chunk in chunks:
                chunk.metadata["total_chunks"] = total_chunks

            # 5. 문서 메타데이터 생성
            doc_metadata = DocumentMetadata(
                doc_id=chunks[0].doc_id if chunks else str(uuid.uuid4()),
                source=normalized_doc.source,
                file_name=normalized_doc.file_name,
                file_type=normalized_doc.file_type,
                file_size=normalized_doc.metadata.get("file_size", 0),
                page_count=normalized_doc.metadata.get("page_count"),
                sheet_count=normalized_doc.metadata.get("sheet_count"),
            )

            # 6. 통계
            stats = self._chunking_service.get_chunk_stats(chunks)
            stats["original_length"] = len(raw_doc.content)
            stats["normalized_length"] = len(normalized_doc.content)

            return PreprocessingResult(
                document=normalized_doc,
                chunks=chunks,
                metadata=doc_metadata,
                stats=stats,
                success=True,
            )

        except Exception as e:
            return PreprocessingResult(
                document=None,
                chunks=[],
                metadata=None,
                stats={},
                success=False,
                error=str(e),
            )

    def process_directory(
        self, dir_path: str, recursive: bool = False
    ) -> List[PreprocessingResult]:
        """디렉토리 내 모든 파일 전처리"""
        results = []
        path = Path(dir_path)
        pattern = "**/*" if recursive else "*"

        supported_exts = self._parser.get_supported_extensions()

        for file_path in path.glob(pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower().lstrip(".")
                if ext in supported_exts:
                    result = self.process_file(str(file_path))
                    results.append(result)

                    if result.success:
                        print(f"✓ {file_path.name}: {result.stats['count']}개 청크")
                    else:
                        print(f"✗ {file_path.name}: {result.error}")

        return results

    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록 반환"""
        return self._parser.get_supported_extensions()
