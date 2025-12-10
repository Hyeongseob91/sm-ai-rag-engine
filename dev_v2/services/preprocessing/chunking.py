# Chunking Service - SemanticChunker 기반 시맨틱 청킹.
# 의미적 유사도를 기반으로 텍스트를 분할하여 문맥을 보존한 청크를 생성합니다.
import uuid
from typing import Any, Dict, List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from ...config import Settings
from ...schemas.preprocessing import Chunk, RawDocument


class ChunkingService:
    """시맨틱 청킹 서비스.

    OpenAI 임베딩 모델과 SemanticChunker를 사용하여 텍스트를 의미 단위로 분할하고 청크 통계를 제공합니다.
    """

    def __init__(self, settings: Settings):
        self.settings = settings  # 외부에서 주입된 설정 객체
        self._config = settings.preprocessing  # 전처리 관련 설정 섹션
        self._embeddings = None  # Lazy init: OpenAIEmbeddings 인스턴스
        self._chunker = None  # Lazy init: SemanticChunker 인스턴스

    def _initialize(self):
        """Lazy initialization"""
        if self._embeddings is None:
            # 설정된 임베딩 모델로 OpenAI 임베딩 생성
            self._embeddings = OpenAIEmbeddings(model=self._config.embedding_model)
            # 의미적 단절점을 찾는 청킹기 구성
            self._chunker = SemanticChunker(
                embeddings=self._embeddings,
                breakpoint_threshold_type=self._config.breakpoint_type,
                breakpoint_threshold_amount=self._config.breakpoint_threshold,
            )

    def chunk_text(self, text: str, doc_id: str, source: str, file_name: str, file_type: str) -> List[Chunk]:
        """텍스트를 시맨틱 청킹"""
        self._initialize()  # 필요 시 청커 초기화

        # SemanticChunker로 분할
        langchain_docs = self._chunker.create_documents([text])  # LangChain Document 리스트 생성

        chunks = []
        for i, lc_doc in enumerate(langchain_docs):
            content = lc_doc.page_content  # 실제 청크 텍스트

            chunk = Chunk(
                content=content,
                chunk_id=str(uuid.uuid4()),  # 청크 고유 ID
                chunk_index=i,  # 청크 순번
                doc_id=doc_id,  # 원본 문서 ID
                source=source,  # 문서 출처
                file_name=file_name,  # 파일명
                file_type=file_type,  # 파일 타입
                metadata={
                    "chunking_method": "semantic",  # 사용한 청킹 방식
                    "breakpoint_type": self._config.breakpoint_type,  # 단절점 유형
                    "breakpoint_threshold": self._config.breakpoint_threshold,  # 단절점 임계값
                },
            )
            chunks.append(chunk)

        # Safety Guard: 청크 크기 검증 및 보정
        chunks = self._post_process_chunks(chunks, doc_id, source, file_name, file_type)

        return chunks

    # Factory Method
    def _split_large_chunk(
        self, chunk: Chunk, doc_id: str, source: str, file_name: str, file_type: str
    ) -> List[Chunk]:
        """max_chunk_size 초과 청크를 RecursiveCharacterTextSplitter로 재분할.

        Args:
            chunk: 분할할 청크
            doc_id, source, file_name, file_type: 새 청크 생성 시 필요한 메타데이터

        Returns:
            분할된 Chunk 리스트 (원본 크기가 max 이하면 원본 그대로 반환)
        """
        max_size = self._config.max_chunk_size

        if chunk.char_count <= max_size:
            return [chunk]

        # RecursiveCharacterTextSplitter로 재분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size,
            chunk_overlap=100,  # 문맥 보존을 위한 오버랩
            separators=["\n\n", "\n", ". ", ", ", " "],
        )

        split_texts = splitter.split_text(chunk.content)

        new_chunks = []
        for split_text in split_texts:
            new_chunk = Chunk(
                content=split_text,
                chunk_id=str(uuid.uuid4()),
                chunk_index=0,  # 나중에 재정렬됨
                doc_id=doc_id,
                source=source,
                file_name=file_name,
                file_type=file_type,
                metadata={
                    **chunk.metadata,
                    "was_split": True,  # 분할 표시
                    "original_chunk_id": chunk.chunk_id,
                },
            )
            new_chunks.append(new_chunk)

        return new_chunks

    def _merge_small_chunks(
        self, chunks: List[Chunk], doc_id: str, source: str, file_name: str, file_type: str
    ) -> List[Chunk]:
        """min_chunk_size 미달 청크를 인접 청크와 병합.

        Args:
            chunks: 청크 리스트
            doc_id, source, file_name, file_type: 새 청크 생성 시 필요한 메타데이터

        Returns:
            병합된 Chunk 리스트
        """
        if not chunks:
            return chunks

        min_size = self._config.min_chunk_size
        max_size = self._config.max_chunk_size
        merged_chunks = []
        buffer_content = ""
        buffer_metadata = {}
        merged_ids = []

        for chunk in chunks:
            # 버퍼 + 현재 청크 합친 크기 계산
            combined = buffer_content + ("\n\n" if buffer_content else "") + chunk.content
            combined_size = len(combined)

            if chunk.char_count >= min_size and not buffer_content:
                # 정상 크기이고 버퍼가 비어있으면 그대로 추가
                merged_chunks.append(chunk)
            elif combined_size <= max_size:
                # 합쳐도 max 이하면 버퍼에 누적
                buffer_content = combined
                if not buffer_metadata:
                    buffer_metadata = chunk.metadata.copy()
                merged_ids.append(chunk.chunk_id)
            else:
                # 합치면 max 초과 → 버퍼 내용을 새 청크로 생성
                if buffer_content:
                    new_chunk = Chunk(
                        content=buffer_content,
                        chunk_id=str(uuid.uuid4()),
                        chunk_index=0,  # 나중에 재정렬
                        doc_id=doc_id,
                        source=source,
                        file_name=file_name,
                        file_type=file_type,
                        metadata={
                            **buffer_metadata,
                            "was_merged": True,
                            "merged_from": merged_ids.copy(),
                        },
                    )
                    merged_chunks.append(new_chunk)

                # 버퍼 리셋하고 현재 청크로 시작
                buffer_content = chunk.content
                buffer_metadata = chunk.metadata.copy()
                merged_ids = [chunk.chunk_id]

        # 남은 버퍼 처리
        if buffer_content:
            new_chunk = Chunk(
                content=buffer_content,
                chunk_id=str(uuid.uuid4()),
                chunk_index=0,
                doc_id=doc_id,
                source=source,
                file_name=file_name,
                file_type=file_type,
                metadata={
                    **buffer_metadata,
                    "was_merged": len(merged_ids) > 1,
                    "merged_from": merged_ids if len(merged_ids) > 1 else [],
                },
            )
            merged_chunks.append(new_chunk)

        return merged_chunks

    def _post_process_chunks(
        self, chunks: List[Chunk], doc_id: str, source: str, file_name: str, file_type: str
    ) -> List[Chunk]:
        """청크 크기 검증 및 보정 (Safety Guard).

        1. 큰 청크(max_chunk_size 초과) 분할
        2. 작은 청크(min_chunk_size 미달) 병합
        3. chunk_index 재정렬

        Args:
            chunks: 원본 청크 리스트
            doc_id, source, file_name, file_type: 메타데이터

        Returns:
            보정된 Chunk 리스트
        """
        if not chunks:
            return chunks

        # 1. 큰 청크 분할
        split_chunks = []
        for chunk in chunks:
            split_chunks.extend(
                self._split_large_chunk(chunk, doc_id, source, file_name, file_type)
            )

        # 2. 작은 청크 병합
        merged_chunks = self._merge_small_chunks(
            split_chunks, doc_id, source, file_name, file_type
        )

        # 3. chunk_index 재정렬
        for i, chunk in enumerate(merged_chunks):
            chunk.chunk_index = i

        return merged_chunks

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        """RawDocument를 시맨틱 청킹"""
        doc_id = str(uuid.uuid4())  # 문서별 고유 ID 생성
        return self.chunk_text(
            text=doc.content,
            doc_id=doc_id,
            source=doc.source,
            file_name=doc.file_name,
            file_type=doc.file_type,
        )

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """청킹 통계 반환"""
        if not chunks:
            return {"count": 0}  # 빈 입력 보호

        sizes = [c.char_count for c in chunks]  # 각 청크 길이 집계
        return {
            "count": len(chunks),  # 총 청크 개수
            "total_chars": sum(sizes),  # 전체 문자 수
            "avg_size": sum(sizes) / len(sizes),  # 평균 청크 길이
            "min_size": min(sizes),  # 최소 청크 길이
            "max_size": max(sizes),  # 최대 청크 길이
        }
