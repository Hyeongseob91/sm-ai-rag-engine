"""
RAG Server v2 - Entry Point
"""
from typing import List

from dotenv import load_dotenv

from .config import Settings
from .graph import RAGWorkflow
from .nodes import GeneratorNode, QueryRewriteNode, RetrieverNode, SimpleGeneratorNode
from .schemas.preprocessing import PreprocessingResult
from .services import LLMService, RerankerService, VectorStoreService
from .services.preprocessing import PreprocessingPipeline


class RAGApplication:
    """RAG 애플리케이션 (DI Container 역할)"""

    def __init__(self, settings: Settings = None):
        # 설정 초기화
        self.settings = settings or Settings()

        # 서비스 초기화 (DI)
        self._llm_service = LLMService(self.settings)
        self._vectorstore_service = VectorStoreService(self.settings)
        self._reranker_service = RerankerService(self.settings)
        self._preprocessing_pipeline = PreprocessingPipeline(self.settings)

        # 노드 초기화 (서비스 주입)
        self._query_rewrite_node = QueryRewriteNode(self._llm_service)
        self._retriever_node = RetrieverNode(
            self._vectorstore_service,
            self._reranker_service,
        )
        self._generator_node = GeneratorNode(self._llm_service)
        self._simple_generator_node = SimpleGeneratorNode(self._llm_service)

        # 워크플로우 초기화 (노드 주입 + Router Pattern)
        self._workflow = RAGWorkflow(
            self._llm_service,
            self._query_rewrite_node,
            self._retriever_node,
            self._generator_node,
            self._simple_generator_node,
        )

    @property
    def vectorstore(self) -> VectorStoreService:
        """VectorStore 서비스 접근"""
        return self._vectorstore_service

    @property
    def preprocessing(self) -> PreprocessingPipeline:
        """전처리 파이프라인 접근"""
        return self._preprocessing_pipeline

    def initialize(self, create_collection: bool = False) -> "RAGApplication":
        """애플리케이션 초기화 (연결 및 빌드)

        Args:
            create_collection: True면 새 컬렉션 생성 (기존 데이터 삭제됨)
        """
        # VectorStore 연결 확인
        if self._vectorstore_service.is_ready():
            print("✅ VectorStore 연결 완료")
        else:
            raise RuntimeError("VectorStore 연결 실패")

        # 컬렉션 생성 (선택적)
        if create_collection:
            self._vectorstore_service.create_collection(extended_schema=True)
            print("✅ 컬렉션 생성 완료 (확장 스키마)")

        # 워크플로우 빌드
        self._workflow.build()
        print("✅ RAG 워크플로우 빌드 완료")

        return self

    def run(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        result = self._workflow.invoke(question)
        return result["final_answer"]

    def ingest_file(self, file_path: str) -> PreprocessingResult:
        """파일 수집 및 Weaviate 저장

        Args:
            file_path: 파일 경로

        Returns:
            PreprocessingResult 객체
        """
        result = self._preprocessing_pipeline.process_file(file_path)
        if result.success and result.chunks:
            added = self._vectorstore_service.add_chunks(result.chunks)
            print(f"✅ {result.metadata.file_name}: {added}개 청크 저장 완료")
        return result

    def ingest_directory(
        self, dir_path: str, recursive: bool = False
    ) -> List[PreprocessingResult]:
        """디렉토리 내 파일 수집 및 Weaviate 저장

        Args:
            dir_path: 디렉토리 경로
            recursive: 하위 디렉토리 포함 여부

        Returns:
            PreprocessingResult 리스트
        """
        results = self._preprocessing_pipeline.process_directory(dir_path, recursive)
        total_chunks = 0
        for result in results:
            if result.success and result.chunks:
                added = self._vectorstore_service.add_chunks(result.chunks)
                total_chunks += added

        print(f"✅ 총 {total_chunks}개 청크 저장 완료")
        return results

    def close(self) -> None:
        """리소스 정리"""
        self._vectorstore_service.close()
        print("✅ 리소스 정리 완료")


def create_app(settings: Settings = None) -> RAGApplication:
    """팩토리 함수: RAG 애플리케이션 생성"""
    load_dotenv()
    return RAGApplication(settings)


# CLI 실행용
if __name__ == "__main__":
    load_dotenv()

    # 애플리케이션 생성 및 초기화
    app = create_app().initialize()

    # 테스트 질문
    test_question = "RAG 성능 고도화의 개념은?"
    print(f"\n질문: {test_question}\n")

    # 실행
    answer = app.run(test_question)
    print(f"\n답변:\n{answer}")

    # 정리
    app.close()
