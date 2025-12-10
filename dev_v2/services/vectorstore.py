"""
VectorStore Service - Weaviate 벡터 DB 관련 기능 제공
"""
import os
from typing import TYPE_CHECKING, List, Optional

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

from ..config import Settings

if TYPE_CHECKING:
    from ..schemas.preprocessing import Chunk


class VectorStoreService:
    """Weaviate Vector Store 서비스 클래스"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[weaviate.WeaviateClient] = None
        self._collection = None

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Weaviate 클라이언트 (Lazy Loading)"""
        if self._client is None:
            self._connect()
        return self._client

    @property
    def collection(self):
        """현재 컬렉션 반환"""
        if self._collection is None:
            self._collection = self.client.collections.get(
                self.settings.vectorstore.collection_name
            )
        return self._collection

    def _connect(self) -> None:
        """Weaviate 연결"""
        data_path = self.settings.vectorstore.data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self._client = weaviate.connect_to_embedded(
            version=self.settings.vectorstore.weaviate_version,
            persistence_data_path=data_path,
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
            environment_variables={
                "ENABLE_MODULES": "text2vec-openai",
                "DEFAULT_VECTORIZER_MODULE": "text2vec-openai",
            },
        )

    def is_ready(self) -> bool:
        """연결 상태 확인"""
        return self.client.is_ready()

    def close(self) -> None:
        """연결 종료"""
        if self._client is not None and self._client.is_connected():
            self._client.close()
            self._client = None
            self._collection = None

    def create_collection(self, extended_schema: bool = True) -> None:
        """컬렉션 생성

        Args:
            extended_schema: True면 확장된 메타데이터 스키마 사용
        """
        name = self.settings.vectorstore.collection_name

        if self.client.collections.exists(name):
            self.client.collections.delete(name)

        # 기본 속성
        properties = [
            Property(name="content", data_type=DataType.TEXT),
        ]

        if extended_schema:
            # 확장된 메타데이터 속성
            properties.extend(
                [
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="total_chunks", data_type=DataType.INT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="file_name", data_type=DataType.TEXT),
                    Property(name="file_type", data_type=DataType.TEXT),
                    Property(name="char_count", data_type=DataType.INT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="sheet_name", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                ]
            )
        else:
            # 기존 호환성을 위한 간단한 스키마
            properties.append(Property(name="doc_id", data_type=DataType.INT))

        self._collection = self.client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model=self.settings.vectorstore.embedding_model
            ),
            inverted_index_config=Configure.inverted_index(
                bm25_b=self.settings.vectorstore.bm25_b,
                bm25_k1=self.settings.vectorstore.bm25_k1,
            ),
            properties=properties,
        )


    def add_documents(self, documents: List[dict]) -> None:
        """문서 추가 (기존 호환성)"""
        with self.collection.batch.dynamic() as batch:
            for doc in documents:
                batch.add_object(properties=doc)


    def add_chunks(self, chunks: List["Chunk"]) -> int:
        """청크 배치 추가

        Args:
            chunks: Chunk 객체 리스트

        Returns:
            추가된 청크 수
        """
        added_count = 0
        with self.collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(properties=chunk.to_weaviate_object())
                added_count += 1
        return added_count


    def get_document_count(self) -> int:
        """저장된 문서(청크) 수 반환"""
        result = self.collection.aggregate.over_all(total_count=True)
        return result.total_count or 0


    def hybrid_search(
        self,
        query: str,
        alpha: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Hybrid Search 실행"""
        if alpha is None:
            alpha = self.settings.retriever.hybrid_alpha
        if limit is None:
            limit = self.settings.retriever.initial_limit

        response = self.collection.query.hybrid(
            query=query,
            alpha=alpha,
            limit=limit,
            return_metadata=MetadataQuery(score=True),
        )

        results = []
        for obj in response.objects:
            results.append(obj.properties["content"])

        return results
