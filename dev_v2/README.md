# dev_v2: Prototype Phase

> dev_v1에서 검증된 기능을 모듈화하여 프로덕션 수준의 RAG 파이프라인 프로토타입 구축

---

## 목차

1. [개요](#1-개요)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [폴더 구조](#3-폴더-구조)
4. [모듈별 상세 설명](#4-모듈별-상세-설명)
5. [핵심 클래스 API](#5-핵심-클래스-api)
6. [설계 패턴](#6-설계-패턴)
7. [환경 설정](#7-환경-설정)
8. [실행 방법](#8-실행-방법)
9. [테스트](#9-테스트)
10. [MVP 단계 준비사항](#10-mvp-단계-준비사항)

---

## 1. 개요

### 1.1 프로젝트 개발 로드맵

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Research & Design (dev_v1)                                │
│  - Jupyter Notebook으로 개별 기능 테스트                              │
│  - 설계 구조 검토 및 검증                                             │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ★ Phase 2: Prototype (dev_v2) ← 현재 단계                          │
│  - 검토된 기능을 Scripts 단위로 모듈화                                │
│  - Prototype 구축                                                   │
│  - TDD 테스트 구축 (test-driven-design 폴더)                         │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: MVP (예정)                                                 │
│  - Docker 컨테이너화                                                 │
│  - 보안 강화                                                         │
│  - 베타 서비스 배포                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 목적

| 목적 | 설명 |
|------|------|
| **모듈화** | Jupyter 노트북 → 재사용 가능한 Python 모듈 |
| **프로토타입 구축** | 완전한 RAG 파이프라인 동작 확인 |
| **테스트 기반 개발** | TDD로 코드 품질 보장 |
| **MVP 준비** | 프로덕션 배포를 위한 기반 마련 |

### 1.3 v1 대비 개선사항

| 항목 | v1 | v2 |
|------|----|----|
| 코드 형식 | Jupyter Notebook | Python 모듈 |
| 상태 관리 | 개념 설명 | LangGraph 구현 |
| 에러 처리 | 없음 | Fallback 패턴 |
| 설정 관리 | 하드코딩 | Settings 클래스 |
| 테스트 | 수동 | pytest TDD (116개) |
| 라우팅 | 없음 | Router Pattern |

---

## 2. 아키텍처 개요

### 2.1 데이터 플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Question                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Router (route_question)                          │
│                                                                     │
│   "환불 규정이 뭐야?" ──────────────► vectorstore (RAG 경로)         │
│   "안녕하세요?"      ──────────────► llm (직접 응답 경로)            │
│   "파이썬 정렬 방법?" ──────────────► llm (직접 응답 경로)            │
└─────────────────────────────────────────────────────────────────────┘
          │                                       │
          │ (vectorstore)                         │ (llm)
          ▼                                       ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│    QueryRewriteNode      │          │   SimpleGeneratorNode    │
│                          │          │                          │
│  "환불 규정"             │          │  LLM 지식으로 직접 응답   │
│      ↓                   │          │                          │
│  ["환불 정책 안내",       │          └──────────────────────────┘
│   "환불 절차 방법",       │                      │
│   "환불 조건 규정"]       │                      │
└──────────────────────────┘                      │
          │                                       │
          ▼                                       │
┌──────────────────────────┐                      │
│     RetrieverNode        │                      │
│                          │                      │
│  Hybrid Search (BM25 +   │                      │
│  Dense Vector)           │                      │
│      ↓                   │                      │
│  CrossEncoder Reranking  │                      │
│      ↓                   │                      │
│  Top-5 Documents         │                      │
└──────────────────────────┘                      │
          │                                       │
          ▼                                       │
┌──────────────────────────┐                      │
│     GeneratorNode        │                      │
│                          │                      │
│  문서 기반 답변 생성      │                      │
│  출처 표기 ([1], [2])     │                      │
│  할루시네이션 방지        │                      │
└──────────────────────────┘                      │
          │                                       │
          └───────────────────┬───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Final Answer                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Router Pattern

Router는 질문을 분석하여 적절한 경로로 분기합니다:

| 경로 | 조건 | 처리 |
|------|------|------|
| **vectorstore** | 기업 규정, 도메인 지식, 특정 사실 | RAG 파이프라인 (검색 → 생성) |
| **llm** | 인사, 일반 상식, 코딩 질문, 요약 | 직접 LLM 응답 |

**Router 판단 로직:**
```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "llm"] = Field(
        description="'vectorstore': 외부 정보 검색 필요, 'llm': 단순 대화/상식"
    )

def route_question(state: RAGState) -> str:
    decision = llm.invoke_with_structured_output(
        prompt=ROUTER_PROMPT,
        output_schema=RouteQuery,
        input_data={"question": state["question"]}
    )
    if decision.datasource == "vectorstore":
        return "query_rewrite"  # RAG 경로
    else:
        return "simple_generator"  # 직접 응답 경로
```

---

## 3. 폴더 구조

```
dev_v2/
├── __init__.py                 # 패키지 초기화
├── main.py                     # 애플리케이션 엔트리 포인트
│
├── config/                     # 설정 관리
│   ├── __init__.py
│   └── settings.py             # Settings, LLMSettings, VectorStoreSettings...
│
├── schemas/                    # 데이터 모델 정의
│   ├── __init__.py
│   ├── state.py                # RAGState (LangGraph 상태)
│   └── models.py               # RouteQuery, RewriteResult (Pydantic)
│
├── services/                   # 비즈니스 로직 서비스
│   ├── __init__.py
│   ├── llm.py                  # LLMService (OpenAI 호출)
│   ├── vectorstore.py          # VectorStoreService (Weaviate)
│   └── reranker.py             # RerankerService (CrossEncoder)
│
├── nodes/                      # LangGraph 노드들
│   ├── __init__.py
│   ├── base.py                 # BaseNode (추상 클래스)
│   ├── query_rewrite.py        # QueryRewriteNode
│   ├── retriever.py            # RetrieverNode
│   ├── generator.py            # GeneratorNode (RAG)
│   └── simple_generator.py     # SimpleGeneratorNode (직접 응답)
│
├── graph/                      # 워크플로우 정의
│   ├── __init__.py
│   └── workflow.py             # RAGWorkflow (LangGraph StateGraph)
│
└── prompts/                    # 프롬프트 템플릿
    ├── __init__.py
    └── templates.py            # 시스템/휴먼 프롬프트
```

---

## 4. 모듈별 상세 설명

### 4.1 config/ - 설정 관리

**settings.py**

계층적 설정 관리를 위한 dataclass 구조:

```python
@dataclass
class LLMSettings:
    rewrite_model: str = "gpt-4o-mini"
    rewrite_temperature: float = 0.0
    generator_model: str = "gpt-4o"
    generator_temperature: float = 0.0

@dataclass
class VectorStoreSettings:
    weaviate_version: str = "1.27.0"
    data_path: str = "./my_weaviate_data"
    collection_name: str = "AdvancedRAG_Chunk"
    embedding_model: str = "text-embedding-3-small"
    bm25_b: float = 0.75
    bm25_k1: float = 1.2

@dataclass
class RerankerSettings:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5

@dataclass
class RetrieverSettings:
    hybrid_alpha: float = 0.5
    initial_limit: int = 30

@dataclass
class Settings:
    llm: LLMSettings
    vectorstore: VectorStoreSettings
    reranker: RerankerSettings
    retriever: RetrieverSettings
```

### 4.2 schemas/ - 데이터 모델

**state.py - RAGState**

LangGraph 노드 간 데이터 전달을 위한 상태 스키마:

```python
class RAGState(TypedDict):
    question: str              # 원본 질문
    optimized_queries: List[str]  # 최적화된 쿼리 리스트
    retrieved_docs: List[str]  # 검색된 문서들
    final_answer: str          # 최종 답변
```

**models.py - Pydantic 모델**

LLM 구조화 출력을 위한 모델:

```python
class RouteQuery(BaseModel):
    """라우팅 결정"""
    datasource: Literal["vectorstore", "llm"]

class RewriteResult(BaseModel):
    """쿼리 재작성 결과"""
    queries: List[str] = Field(
        description="최적화된 검색 쿼리 리스트 (3~5개)"
    )
```

### 4.3 services/ - 비즈니스 로직

**llm.py - LLMService**

OpenAI API 호출 추상화:

```python
class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_rewrite_llm(self) -> ChatOpenAI:
        """Query Rewrite용 LLM (gpt-4o-mini)"""

    def get_generator_llm(self) -> ChatOpenAI:
        """답변 생성용 LLM (gpt-4o)"""

    def invoke_with_structured_output(
        self, llm, prompt, output_schema, input_data
    ) -> BaseModel:
        """구조화된 출력 (JSON/Pydantic)"""

    def invoke_with_string_output(
        self, llm, prompt, input_data
    ) -> str:
        """문자열 출력"""
```

**vectorstore.py - VectorStoreService**

Weaviate 벡터 DB 관리:

```python
class VectorStoreService:
    def is_ready(self) -> bool:
        """연결 상태 확인"""

    def create_collection(self) -> None:
        """컬렉션 생성 (벡터화, BM25 설정)"""

    def add_documents(self, documents: List[Dict]) -> None:
        """문서 배치 추가"""

    def hybrid_search(
        self, query: str, alpha: float, limit: int
    ) -> List[Dict]:
        """하이브리드 검색 (Dense + BM25)"""

    def close(self) -> None:
        """연결 종료"""
```

**reranker.py - RerankerService**

CrossEncoder 기반 리랭킹:

```python
class RerankerService:
    def rerank(
        self, query: str, documents: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        """Query-Document 쌍 재채점"""

    def get_top_documents(
        self, query: str, documents: List[str], top_k: int
    ) -> List[str]:
        """상위 k개 문서만 반환"""
```

### 4.4 nodes/ - LangGraph 노드

**base.py - BaseNode**

모든 노드의 추상 인터페이스:

```python
class BaseNode(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """노드 이름 (고유 식별자)"""

    @abstractmethod
    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """노드 실행 (상태 업데이트)"""
```

**query_rewrite.py - QueryRewriteNode**

```python
class QueryRewriteNode(BaseNode):
    """질문 → 검색 최적화 쿼리 변환"""

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        # 1. 프롬프트로 쿼리 확장
        # 2. RewriteResult 구조로 출력
        # 3. 실패 시 원본 질문 반환 (Fallback)
        return {"optimized_queries": queries}
```

**retriever.py - RetrieverNode**

```python
class RetrieverNode(BaseNode):
    """검색 + 리랭킹"""

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        # 1. 모든 최적화 쿼리로 하이브리드 검색
        # 2. 중복 제거
        # 3. CrossEncoder 리랭킹
        # 4. Top-k 선별
        return {"retrieved_docs": documents}
```

**generator.py - GeneratorNode**

```python
class GeneratorNode(BaseNode):
    """문서 기반 답변 생성"""

    def _format_docs(self, docs: List[str]) -> str:
        """[1], [2] 형식으로 포맷팅"""

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        # 1. 문서 포맷팅
        # 2. 프롬프트 + LLM 호출
        # 3. 할루시네이션 방지 지침 적용
        return {"final_answer": answer}
```

**simple_generator.py - SimpleGeneratorNode**

```python
class SimpleGeneratorNode(BaseNode):
    """검색 없이 직접 응답"""

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        # LLM 지식으로 직접 응답
        # 인사, 코딩 질문, 일반 상식 처리
        return {"final_answer": answer}
```

### 4.5 graph/ - 워크플로우

**workflow.py - RAGWorkflow**

LangGraph StateGraph 기반 워크플로우:

```python
class RAGWorkflow:
    def __init__(
        self,
        llm_service: LLMService,
        query_rewrite_node: QueryRewriteNode,
        retriever_node: RetrieverNode,
        generator_node: GeneratorNode,
        simple_generator_node: SimpleGeneratorNode,
    ):
        """Dependency Injection으로 노드 주입"""

    def route_question(self, state: RAGState) -> str:
        """조건부 라우팅 (vectorstore/llm)"""

    def build(self) -> "RAGWorkflow":
        """StateGraph 구성 및 컴파일"""
        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("query_rewrite", self._query_rewrite_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("simple_generator", self._simple_generator_node)

        # 조건부 진입점
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "query_rewrite": "query_rewrite",
                "simple_generator": "simple_generator",
            }
        )

        # 엣지 연결
        workflow.add_edge("query_rewrite", "retriever")
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)
        workflow.add_edge("simple_generator", END)

        self._app = workflow.compile()
        return self

    def invoke(self, question: str) -> str:
        """질문 실행 → 최종 답변"""
```

### 4.6 prompts/ - 프롬프트 템플릿

**templates.py**

| 프롬프트 | 용도 |
|---------|------|
| `ROUTER_SYSTEM_PROMPT` | 질문 분류 (vectorstore/llm) |
| `QUERY_REWRITE_SYSTEM_PROMPT` | 쿼리 최적화 |
| `GENERATOR_SYSTEM_PROMPT` | 문서 기반 답변 생성 |
| `GENERATOR_HUMAN_PROMPT` | 컨텍스트 + 질문 형식 |

---

## 5. 핵심 클래스 API

### 5.1 LLMService

```python
class LLMService:
    # LLM 인스턴스 생성
    get_rewrite_llm() -> ChatOpenAI
    get_generator_llm() -> ChatOpenAI

    # LLM 호출
    invoke_with_structured_output(llm, prompt, output_schema, input_data) -> BaseModel
    invoke_with_string_output(llm, prompt, input_data) -> str
```

### 5.2 VectorStoreService

```python
class VectorStoreService:
    # 연결 관리
    is_ready() -> bool
    close() -> None

    # 컬렉션 관리
    create_collection() -> None
    add_documents(documents: List[Dict]) -> None

    # 검색
    hybrid_search(query: str, alpha: float, limit: int) -> List[Dict]
```

### 5.3 RerankerService

```python
class RerankerService:
    # 리랭킹
    rerank(query: str, documents: List[str], top_k: int) -> List[Tuple[str, float]]
    get_top_documents(query: str, documents: List[str], top_k: int) -> List[str]
```

### 5.4 RAGWorkflow

```python
class RAGWorkflow:
    # 워크플로우 구성
    build() -> RAGWorkflow

    # 라우팅
    route_question(state: RAGState) -> str

    # 실행
    invoke(question: str) -> str
```

---

## 6. 설계 패턴

### 6.1 Dependency Injection (DI)

```python
# main.py에서 서비스 → 노드 주입
llm_service = LLMService(settings)
vectorstore_service = VectorStoreService(settings)
reranker_service = RerankerService(settings)

query_rewrite_node = QueryRewriteNode(llm_service)
retriever_node = RetrieverNode(vectorstore_service, reranker_service)
generator_node = GeneratorNode(llm_service)

workflow = RAGWorkflow(
    llm_service,
    query_rewrite_node,
    retriever_node,
    generator_node,
    simple_generator_node,
)
```

**장점:**
- 테스트 시 Mock 주입 용이
- 컴포넌트 교체 유연성
- 의존성 명시적 관리

### 6.2 Router Pattern

질문 유형에 따른 조건부 분기:

```python
def route_question(state: RAGState) -> str:
    decision = llm.invoke_with_structured_output(...)
    if decision.datasource == "vectorstore":
        return "query_rewrite"
    else:
        return "simple_generator"
```

**장점:**
- 불필요한 검색 비용 절감
- 응답 속도 향상 (단순 질문)
- 리소스 효율적 사용

### 6.3 Lazy Loading

```python
class VectorStoreService:
    def __init__(self, settings):
        self._client = None  # 지연 초기화

    @property
    def client(self):
        if self._client is None:
            self._client = weaviate.connect_to_embedded(...)
        return self._client
```

**장점:**
- 메모리 효율성
- 시작 시간 단축
- 필요할 때만 리소스 할당

### 6.4 Factory Pattern

```python
class LLMService:
    def get_rewrite_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm.rewrite_model,
            temperature=self.settings.llm.rewrite_temperature,
        )

    def get_generator_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm.generator_model,
            temperature=self.settings.llm.generator_temperature,
        )
```

**장점:**
- 객체 생성 로직 캡슐화
- 설정 기반 인스턴스 생성
- 일관된 객체 구성

### 6.5 Template Method

```python
class BaseNode(ABC):
    @abstractmethod
    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """모든 노드가 구현해야 하는 메서드"""
        pass
```

**장점:**
- 일관된 노드 인터페이스
- 다형성 활용
- LangGraph 통합 용이

---

## 7. 환경 설정

### 7.1 환경 변수

```bash
# .env 파일
OPENAI_API_KEY=sk-...
WEAVIATE_DATA_PATH=./my_weaviate_data  # 선택적
```

### 7.2 설정 계층 구조

```
Settings (최상위)
├── LLMSettings
│   ├── rewrite_model: "gpt-4o-mini"
│   ├── rewrite_temperature: 0.0
│   ├── generator_model: "gpt-4o"
│   └── generator_temperature: 0.0
│
├── VectorStoreSettings
│   ├── weaviate_version: "1.27.0"
│   ├── data_path: "./my_weaviate_data"
│   ├── collection_name: "AdvancedRAG_Chunk"
│   ├── embedding_model: "text-embedding-3-small"
│   ├── bm25_b: 0.75
│   └── bm25_k1: 1.2
│
├── RerankerSettings
│   ├── model_name: "BAAI/bge-reranker-v2-m3"
│   └── top_k: 5
│
└── RetrieverSettings
    ├── hybrid_alpha: 0.5
    └── initial_limit: 30
```

### 7.3 설정 오버라이드

```python
# 커스텀 설정으로 앱 생성
custom_settings = Settings(
    llm=LLMSettings(generator_model="gpt-4o-2024-08-06"),
    reranker=RerankerSettings(top_k=10),
)
app = create_app(settings=custom_settings)
```

---

## 8. 실행 방법

### 8.1 환경 설정

```bash
# 의존성 설치 (uv 사용)
uv sync

# 환경 변수 설정
export OPENAI_API_KEY=sk-...
```

### 8.2 CLI 실행

```bash
# main.py 직접 실행
uv run python -m dev_v2.main

# 출력 예시:
# === RAG Application 시작 ===
# --- [Router] 질문 분류 중... ---
# --- [Route: vectorstore] RAG 파이프라인 시작 ---
# --- [Step 1] Query Rewrite 시작 ---
# --- [Step 2] Retriever 시작 ---
# --- [Step 3] Generator 시작 ---
# === 최종 답변 ===
# RAG 성능 고도화란...
```

### 8.3 프로그래밍 방식

```python
from dev_v2.main import create_app

# 앱 생성 및 초기화
app = create_app()
app.initialize()

# 질문 실행
answer = app.run("RAG 성능 고도화의 개념은?")
print(answer)

# 리소스 정리
app.close()
```

### 8.4 테스트 질문 예시

```python
# RAG 경로 (vectorstore)
app.run("회사의 환불 규정이 어떻게 되나요?")
app.run("프로젝트 담당자가 누구인가요?")

# LLM 경로 (직접 응답)
app.run("안녕하세요?")
app.run("파이썬에서 리스트 정렬 방법은?")
app.run("1+1은 몇이야?")
```

---

## 9. 테스트

### 9.1 테스트 폴더 구조

```
test-driven-design/
├── conftest.py                 # 공통 fixtures
├── unit/                       # 단위 테스트 (64개)
│   ├── test_llm_service.py
│   ├── test_reranker_service.py
│   ├── test_query_rewrite_node.py
│   ├── test_retriever_node.py
│   ├── test_generator_node.py
│   └── test_simple_generator_node.py
├── integration/                # 통합 테스트 (28개)
│   ├── test_workflow_routing.py
│   ├── test_vectorstore_service.py
│   └── test_rag_pipeline.py
└── e2e/                        # E2E 테스트 (17개)
    └── test_rag_application.py
```

### 9.2 테스트 실행

```bash
# 단위 테스트 (Mock 기반, 빠름)
uv run pytest test-driven-design/ -v -m unit

# 통합 테스트 (실제 API 필요)
uv run pytest test-driven-design/ -v -m integration

# E2E 테스트 (전체 환경 필요)
uv run pytest test-driven-design/ -v -m e2e

# 전체 테스트
uv run pytest test-driven-design/ -v

# 커버리지 포함
uv run pytest test-driven-design/ -v --cov=dev_v2 --cov-report=html
```

### 9.3 테스트 통계

```
총 테스트 수: 116개
├── unit/     : 64개 ✅ 전체 통과
├── integration/: 28개 (API 키 필요)
└── e2e/      : 17개 (전체 환경 필요)
```

### 9.4 TDD로 발견된 버그

테스트 작성 과정에서 3개의 프로덕션 버그를 발견하고 수정했습니다:

| # | 파일 | 버그 | 수정 |
|---|------|------|------|
| 1 | `generator.py` | `from_template()` 오용 | `from_messages()` 사용 |
| 2 | `simple_generator.py` | LCEL 체인 Mock 불가 | 서비스 메서드 사용 |
| 3 | `prompts/__init__.py` | 잘못된 변수명 export | 올바른 변수명으로 수정 |

---

## 10. MVP 단계 준비사항

### 10.1 예정된 개선사항

| 영역 | 현재 (Prototype) | MVP 목표 |
|------|-----------------|---------|
| **배포** | 로컬 실행 | Docker 컨테이너화 |
| **보안** | API 키 환경변수 | Secret Manager 연동 |
| **인증** | 없음 | JWT/OAuth 2.0 |
| **API** | CLI | FastAPI REST 엔드포인트 |
| **로깅** | print 문 | 구조화된 로깅 (JSON) |
| **모니터링** | 없음 | Prometheus + Grafana |
| **캐싱** | 없음 | Redis 쿼리 캐싱 |

### 10.2 Docker 컨테이너화 계획

```dockerfile
# Dockerfile (예정)
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

EXPOSE 8000
CMD ["uvicorn", "dev_v3.api:app", "--host", "0.0.0.0"]
```

### 10.3 보안 강화 계획

- API 키 Secret Manager 저장
- 입력 검증 및 Sanitization
- Rate Limiting
- CORS 설정
- HTTPS 강제

### 10.4 베타 서비스 체크리스트

- [ ] Docker 이미지 빌드
- [ ] 환경별 설정 분리 (dev/staging/prod)
- [ ] 헬스체크 엔드포인트
- [ ] 에러 알림 (Slack/Email)
- [ ] 성능 모니터링 대시보드
- [ ] 백업 및 복구 전략
- [ ] 사용량 제한 정책

---

## 부록: 기술 스택 요약

```python
# 핵심 프레임워크
langchain-openai>=1.1.0
langgraph>=1.0.4
weaviate-client>=4.0.0
sentence-transformers>=5.1.2

# 데이터 검증
pydantic

# 테스트
pytest>=9.0.2
pytest-cov>=7.0.0

# 개발 환경
python>=3.13
uv  # 패키지 관리자
```

---

**작성일:** 2025-12-09
**버전:** 1.0.0
**다음 단계:** Phase 3 - MVP (Docker, 보안, 베타 서비스)
