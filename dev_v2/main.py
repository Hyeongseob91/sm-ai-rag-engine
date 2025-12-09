"""
RAG Server v2 - Entry Point
"""
from dotenv import load_dotenv

from .config import Settings
from .services import LLMService, VectorStoreService, RerankerService
from .nodes import QueryRewriteNode, RetrieverNode, GeneratorNode, SimpleGeneratorNode
from .graph import RAGWorkflow


class RAGApplication:
    """RAG 애플리케이션 (DI Container 역할)"""

    def __init__(self, settings: Settings = None):
        # 설정 초기화
        self.settings = settings or Settings()

        # 서비스 초기화 (DI)
        self._llm_service = LLMService(self.settings)
        self._vectorstore_service = VectorStoreService(self.settings)
        self._reranker_service = RerankerService(self.settings)

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

    def initialize(self) -> "RAGApplication":
        """애플리케이션 초기화 (연결 및 빌드)"""
        # VectorStore 연결 확인
        if self._vectorstore_service.is_ready():
            print("✅ VectorStore 연결 완료")
        else:
            raise RuntimeError("VectorStore 연결 실패")

        # 워크플로우 빌드
        self._workflow.build()
        print("✅ RAG 워크플로우 빌드 완료")

        return self

    def run(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        result = self._workflow.invoke(question)
        return result["final_answer"]

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
