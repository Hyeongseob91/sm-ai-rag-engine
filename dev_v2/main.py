"""
RAG Server v2 - Entry Point

DI(Dependency Injection) 패턴으로 구성된 RAG 파이프라인

# dev_v2 RAG Architecture 구조
  dev_v2/
  ├── __init__.py                    # 패키지 초기화
  ├── main.py                        # 엔트리포인트 + DI Container
  │
  ├── config/
  │   ├── __init__.py
  │   └── settings.py                # 설정 관리 (LLM, VectorStore, Reranker)
  │
  ├── schemas/
  │   ├── __init__.py
  │   ├── state.py                   # RAGState, 부분 스키마 (TypedDict)
  │   └── models.py                  # Pydantic 모델 (RewriteResult)
  │
  ├── prompts/
  │   ├── __init__.py
  │   └── templates.py               # 프롬프트 템플릿 (Rewrite, Generator)
  │
  ├── services/
  │   ├── __init__.py
  │   ├── llm.py                     # LLMService 클래스
  │   ├── vectorstore.py             # VectorStoreService 클래스 (Weaviate)
  │   └── reranker.py                # RerankerService 클래스 (CrossEncoder)
  │
  ├── nodes/
  │   ├── __init__.py
  │   ├── base.py                    # BaseNode 추상 클래스
  │   ├── query_rewrite.py           # QueryRewriteNode
  │   ├── retriever.py               # RetrieverNode
  │   └── generator.py               # GeneratorNode
  │
  └── graph/
      ├── __init__.py
      └── workflow.py                # RAGWorkflow (LangGraph)
"""
from dotenv import load_dotenv

from .config import Settings
from .services import LLMService, VectorStoreService, RerankerService
from .nodes import QueryRewriteNode, RetrieverNode, GeneratorNode
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

        # 워크플로우 초기화 (노드 주입)
        self._workflow = RAGWorkflow(
            self._query_rewrite_node,
            self._retriever_node,
            self._generator_node,
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
