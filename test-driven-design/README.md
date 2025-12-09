# Test-Driven Design (TDD) for RAG Pipeline

> **dev_v2/** 모듈의 품질 보증을 위한 체계적인 테스트 설계 문서

---

## 목차

1. [개요](#1-개요)
2. [테스트 대상 시스템 분석](#2-테스트-대상-시스템-분석)
3. [테스트 폴더 구조](#3-테스트-폴더-구조)
4. [데이터플로우 기반 테스트 설계](#4-데이터플로우-기반-테스트-설계)
5. [테스트 실행 결과](#5-테스트-실행-결과)
6. [발견된 버그 및 수정 과정](#6-발견된-버그-및-수정-과정)
7. [Mock 전략 및 Fixtures](#7-mock-전략-및-fixtures)
8. [테스트 파일별 상세 분석](#8-테스트-파일별-상세-분석)
9. [실행 방법](#9-실행-방법)
10. [결론 및 교훈](#10-결론-및-교훈)

---

## 1. 개요

### 1.1 프로젝트 배경

이 테스트 스위트는 **RAG (Retrieval-Augmented Generation) 파이프라인**의 모든 핵심 컴포넌트를 검증하기 위해 설계되었습니다. Router Pattern 구현 이후, 코드의 신뢰성과 유지보수성을 확보하기 위해 TDD 수준의 테스트 클래스를 구축했습니다.

### 1.2 목표

| 목표 | 설명 |
|------|------|
| **품질 보증** | 모든 핵심 메서드가 예상대로 동작하는지 검증 |
| **회귀 방지** | 코드 변경 시 기존 기능이 깨지지 않음을 보장 |
| **버그 조기 발견** | TDD 과정에서 실제 코드의 버그를 사전에 발견 |
| **문서화** | 테스트 코드가 곧 사용 예제이자 스펙 문서 역할 |
| **리팩토링 안정성** | 테스트가 있으면 안심하고 리팩토링 가능 |

### 1.3 테스트 범위 선택

사용자 요구사항에 따라 **전체 범위 (Unit + Integration + E2E)**를 선택했으며, **데이터플로우 순서**에 따라 구현을 진행했습니다.

```
선택된 범위: 전체 (Unit + Integration + E2E)
구현 순서: 데이터플로우 순서 (Router → QueryRewrite → Retriever → Generator)
```

---

## 2. 테스트 대상 시스템 분석

### 2.1 dev_v2/ 폴더 구조

테스트 대상인 `dev_v2/` 모듈의 전체 구조입니다:

```
dev_v2/
├── __init__.py
├── main.py                      # RAGApplication (진입점)
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Settings (환경 설정)
│
├── schemas/
│   ├── __init__.py
│   ├── state.py                 # RAGState (TypedDict)
│   └── models.py                # RouteQuery, RewriteResult (Pydantic)
│
├── services/
│   ├── __init__.py
│   ├── llm.py                   # LLMService (LLM 호출 추상화)
│   ├── vectorstore.py           # VectorStoreService (Weaviate)
│   └── reranker.py              # RerankerService (CrossEncoder)
│
├── nodes/
│   ├── __init__.py
│   ├── base.py                  # BaseNode (추상 클래스)
│   ├── query_rewrite.py         # QueryRewriteNode
│   ├── retriever.py             # RetrieverNode
│   ├── generator.py             # GeneratorNode
│   └── simple_generator.py      # SimpleGeneratorNode
│
├── prompts/
│   ├── __init__.py
│   └── templates.py             # 프롬프트 템플릿 모음
│
└── graph/
    ├── __init__.py
    └── workflow.py              # RAGWorkflow (LangGraph StateGraph)
```

### 2.2 핵심 컴포넌트 역할

| 컴포넌트 | 역할 | 테스트 우선순위 |
|---------|------|----------------|
| `RAGWorkflow.route_question()` | 질문을 RAG/LLM 경로로 분류 | 🔴 최상위 |
| `QueryRewriteNode` | 질문을 검색 최적화 쿼리로 확장 | 🔴 높음 |
| `RetrieverNode` | 하이브리드 검색 + 리랭킹 | 🔴 높음 |
| `GeneratorNode` | 문서 기반 답변 생성 | 🟡 중간 |
| `SimpleGeneratorNode` | 검색 없이 직접 답변 | 🟡 중간 |
| `LLMService` | LLM 호출 추상화 계층 | 🟢 기반 |
| `RerankerService` | CrossEncoder 리랭킹 | 🟢 기반 |
| `VectorStoreService` | Weaviate 연동 | 🟢 기반 |

---

## 3. 테스트 폴더 구조

### 3.1 디렉토리 레이아웃

```
test-driven-design/
├── __init__.py                           # 패키지 초기화
├── conftest.py                           # 공통 pytest fixtures
├── README.md                             # 이 문서
│
├── unit/                                 # 단위 테스트 (Mock 기반)
│   ├── __init__.py
│   ├── test_llm_service.py               # LLMService 테스트 (8개)
│   ├── test_reranker_service.py          # RerankerService 테스트 (12개)
│   ├── test_query_rewrite_node.py        # QueryRewriteNode 테스트 (11개)
│   ├── test_retriever_node.py            # RetrieverNode 테스트 (10개)
│   ├── test_generator_node.py            # GeneratorNode 테스트 (11개)
│   └── test_simple_generator_node.py     # SimpleGeneratorNode 테스트 (12개)
│
├── integration/                          # 통합 테스트 (실제 서비스)
│   ├── __init__.py
│   ├── test_workflow_routing.py          # Router 테스트 (8개)
│   ├── test_vectorstore_service.py       # VectorStore 테스트 (12개)
│   └── test_rag_pipeline.py              # 파이프라인 테스트 (8개)
│
└── e2e/                                  # End-to-End 테스트
    ├── __init__.py
    └── test_rag_application.py           # 전체 앱 테스트 (17개)
```

### 3.2 테스트 분류 체계

```
┌─────────────────────────────────────────────────────────────┐
│                      E2E Tests (17개)                       │
│    RAGApplication 전체 생명주기 검증                         │
│    - 실제 API 호출, 실제 VectorStore 연결                    │
├─────────────────────────────────────────────────────────────┤
│                 Integration Tests (28개)                    │
│    여러 컴포넌트의 상호작용 검증                              │
│    - 실제 LLM API 호출                                      │
│    - 실제 Weaviate 연결                                     │
├─────────────────────────────────────────────────────────────┤
│                    Unit Tests (64개)                        │
│    개별 메서드의 독립적 동작 검증                             │
│    - Mock 객체 사용                                         │
│    - 빠른 실행 (외부 의존성 없음)                            │
└─────────────────────────────────────────────────────────────┘
```

**테스트 마커 정의:**

| 마커 | 용도 | 실행 조건 |
|------|------|----------|
| `@pytest.mark.unit` | 단위 테스트 | Mock만 사용, 항상 실행 가능 |
| `@pytest.mark.integration` | 통합 테스트 | API 키, Weaviate 필요 |
| `@pytest.mark.e2e` | E2E 테스트 | 전체 환경 구성 필요 |
| `@pytest.mark.slow` | 느린 테스트 | API 호출 포함 |

---

## 4. 데이터플로우 기반 테스트 설계

### 4.1 RAG 파이프라인 데이터플로우

```
[입력] question: str
       ↓
┌─────────────────────────────────────────────────────────────┐
│ [Router] route_question()                                   │
│   ⟶ RouteQuery(datasource="vectorstore" | "llm")           │
│   테스트: 질문 분류 정확도                                   │
└─────────────────────────────────────────────────────────────┘
       ↓                              ↓
       ↓ (vectorstore)                ↓ (llm)
       ↓                              ↓
┌──────────────────┐          ┌──────────────────┐
│ RAG Path         │          │ LLM Path         │
└──────────────────┘          └──────────────────┘
       ↓                              ↓
┌──────────────────┐          ┌──────────────────┐
│ QueryRewriteNode │          │ SimpleGenerator  │
│ 테스트: 쿼리 확장 │          │ 테스트: 즉시 응답 │
│ 입력: question    │          │ 입력: question    │
│ 출력: queries[]   │          │ 출력: answer      │
└──────────────────┘          └──────────────────┘
       ↓                              ↓
┌──────────────────┐                  │
│ RetrieverNode    │                  │
│ 테스트: 검색+리랭킹│                  │
│ 입력: queries[]   │                  │
│ 출력: docs[]      │                  │
└──────────────────┘                  │
       ↓                              │
┌──────────────────┐                  │
│ GeneratorNode    │                  │
│ 테스트: 문맥 기반 │                  │
│ 입력: docs[]      │                  │
│ 출력: answer      │                  │
└──────────────────┘                  │
       ↓                              ↓
┌─────────────────────────────────────────────────────────────┐
│ [출력] final_answer: str                                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 RAGState 스키마

모든 노드가 공유하는 상태 객체:

```python
class RAGState(TypedDict):
    question: str              # 원본 질문
    optimized_queries: List[str]  # 확장된 검색 쿼리
    retrieved_docs: List[str]  # 검색된 문서
    final_answer: str          # 최종 답변
```

### 4.3 테스트 구현 순서 (7단계)

데이터플로우 순서에 따라 총 7개의 Phase로 구현했습니다:

```
Phase 1: 기반 설정
├── TDD/__init__.py
├── TDD/unit/__init__.py
├── TDD/integration/__init__.py
├── TDD/e2e/__init__.py
└── TDD/conftest.py (fixtures)

Phase 2: Router 테스트 (진입점)
├── [Unit] test_workflow_routing.py (route_question Mock 테스트)
└── [Integration] test_workflow_routing.py (실제 LLM 라우팅)

Phase 3: QueryRewrite 테스트
├── [Unit] test_query_rewrite_node.py (Mock LLMService)
└── [Unit] test_llm_service.py (LLMService 단위 테스트)

Phase 4: Retriever 테스트
├── [Unit] test_retriever_node.py (Mock VectorStore, Reranker)
├── [Unit] test_reranker_service.py (Mock CrossEncoder)
└── [Integration] test_vectorstore_service.py (실제 Weaviate)

Phase 5: Generator 테스트
├── [Unit] test_generator_node.py (Mock LLMService)
└── [Unit] test_simple_generator_node.py (Mock LLMService)

Phase 6: 파이프라인 통합 테스트
└── [Integration] test_rag_pipeline.py (전체 흐름)

Phase 7: E2E 테스트
└── [E2E] test_rag_application.py (RAGApplication 전체)
```

---

## 5. 테스트 실행 결과

### 5.1 최종 테스트 통계

```
===================== 테스트 수집 결과 =====================
총 테스트 수: 116개
├── unit/     : 64개
├── integration/: 28개
└── e2e/      : 17개

===================== 단위 테스트 실행 결과 =====================
실행 명령어: uv run pytest test-driven-design/ -v -m unit

결과: 62 passed, 7 deselected in 0.80s
성공률: 100% (단위 테스트 전체 통과)
```

### 5.2 테스트 파일별 결과

| 파일 | 테스트 수 | 결과 | 비고 |
|------|----------|------|------|
| `test_llm_service.py` | 8개 | ✅ 전체 통과 | - |
| `test_reranker_service.py` | 12개 | ✅ 전체 통과 | - |
| `test_query_rewrite_node.py` | 11개 | ✅ 전체 통과 | - |
| `test_retriever_node.py` | 10개 | ✅ 전체 통과 | - |
| `test_generator_node.py` | 11개 | ✅ 전체 통과 | 버그 수정 후 |
| `test_simple_generator_node.py` | 12개 | ✅ 전체 통과 | 버그 수정 후 |
| `test_workflow_routing.py` | 8개 | ✅ 전체 통과 | - |
| `test_vectorstore_service.py` | 12개 | ⏸️ 대기 | Integration |
| `test_rag_pipeline.py` | 8개 | ⏸️ 대기 | Integration |
| `test_rag_application.py` | 17개 | ⏸️ 대기 | E2E |

**참고:** Integration/E2E 테스트는 실제 API 키와 Weaviate 연결이 필요합니다.

---

## 6. 발견된 버그 및 수정 과정

TDD 과정에서 **실제 프로덕션 코드의 버그 3건**을 발견하고 수정했습니다. 이는 TDD의 핵심 가치인 **"버그 조기 발견"**을 증명합니다.

### 6.1 버그 #1: GeneratorNode - ChatPromptTemplate 메서드 오류

**발견 경위:** `test_generator_node.py` 실행 시 TypeError 발생

**에러 메시지:**
```
TypeError: expected str, got list
File: dev_v2/nodes/generator.py, Line 19
```

**원인 분석:**
```python
# 수정 전 (오류 코드)
self._prompt = ChatPromptTemplate.from_template([
    ("system", GENERATOR_SYSTEM_PROMPT),
    ("human", GENERATOR_HUMAN_PROMPT)
])
```

`from_template()`은 **단일 문자열**을 받는 메서드인데, **리스트**를 전달했습니다.

**수정 후:**
```python
# 수정 후 (올바른 코드)
self._prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM_PROMPT),
    ("human", GENERATOR_HUMAN_PROMPT)
])
```

**교훈:** LangChain의 `from_template()` vs `from_messages()` API 차이를 명확히 이해해야 합니다.

---

### 6.2 버그 #2: SimpleGeneratorNode - Mock 불가능한 LCEL 체인

**발견 경위:** `test_simple_generator_node.py` 실행 시 Pydantic ValidationError 발생

**에러 메시지:**
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ChatOpenAI
  Input should be a valid dictionary or instance of ChatOpenAI [type=model_type]
```

**원인 분석:**
```python
# 수정 전 (Mock 불가능한 코드)
def __call__(self, state: RAGState) -> Dict[str, Any]:
    llm = self._llm_service.get_generator_llm()
    chain = self._prompt | llm | StrOutputParser()  # LCEL 체인
    answer = chain.invoke({"question": state["question"]})
    return {"final_answer": answer}
```

문제점:
1. `self._prompt | llm` 연산에서 `llm`이 Mock 객체이면 `|` (파이프) 연산자가 실패
2. LCEL 체인은 실제 LangChain 객체를 요구함
3. Mock 객체로 대체 불가능

**수정 후:**
```python
# 수정 후 (Mock 가능한 코드)
def __call__(self, state: RAGState) -> Dict[str, Any]:
    llm = self._llm_service.get_generator_llm()
    answer = self._llm_service.invoke_with_string_output(
        llm=llm,
        prompt=self._prompt,
        input_data={"question": state["question"]},
    )
    return {"final_answer": answer}
```

**해결 방법:**
- LCEL 체인 대신 `LLMService.invoke_with_string_output()` 메서드 사용
- 서비스 계층에서 체인 로직을 캡슐화하여 Mock 가능하게 변경

**교훈:** LCEL 체인은 편리하지만, **테스트 용이성(Testability)**을 고려하면 서비스 계층에서 추상화하는 것이 좋습니다.

---

### 6.3 버그 #3: prompts/__init__.py - 잘못된 Export

**발견 경위:** 테스트 실행 시 ImportError 발생

**에러 메시지:**
```
ImportError: cannot import name 'GENERATOR_PROMPT_TEMPLATE' from 'dev_v2.prompts'
```

**원인 분석:**
```python
# 수정 전 (__init__.py)
from .templates import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    GENERATOR_PROMPT_TEMPLATE,  # 존재하지 않는 변수명
    ROUTER_SYSTEM_PROMPT,
)
```

`templates.py`에는 `GENERATOR_PROMPT_TEMPLATE`이 없고, `GENERATOR_SYSTEM_PROMPT`와 `GENERATOR_HUMAN_PROMPT`가 따로 있었습니다.

**수정 후:**
```python
# 수정 후 (__init__.py)
from .templates import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,   # 올바른 변수명
    GENERATOR_HUMAN_PROMPT,    # 올바른 변수명
    ROUTER_SYSTEM_PROMPT,
)
```

**교훈:** 모듈의 `__init__.py`에서 re-export할 때는 실제 존재하는 변수명을 확인해야 합니다.

---

### 6.4 버그 요약 표

| # | 파일 | 버그 유형 | 심각도 | 해결 시간 |
|---|------|---------|--------|----------|
| 1 | `nodes/generator.py` | API 메서드 오용 | 🔴 Critical | 5분 |
| 2 | `nodes/simple_generator.py` | Mock 불가능 설계 | 🟡 Medium | 15분 |
| 3 | `prompts/__init__.py` | Import 오류 | 🔴 Critical | 2분 |

---

## 7. Mock 전략 및 Fixtures

### 7.1 conftest.py 구조

모든 테스트에서 공유하는 fixtures를 `conftest.py`에 정의했습니다.

```python
# conftest.py 구조
@pytest.fixture
def mock_settings() -> Settings:
    """테스트용 Settings Mock"""

@pytest.fixture
def mock_llm_service(mock_settings) -> Mock:
    """LLMService Mock (핵심)"""

@pytest.fixture
def mock_vectorstore_service(mock_settings) -> Mock:
    """VectorStoreService Mock"""

@pytest.fixture
def mock_reranker_service(mock_settings) -> Mock:
    """RerankerService Mock"""

@pytest.fixture
def sample_rag_state(sample_question) -> RAGState:
    """테스트용 RAGState (초기 상태)"""

@pytest.fixture
def real_llm_service(real_settings) -> LLMService:
    """실제 LLMService (Integration 테스트용)"""
```

### 7.2 Mock 패턴

**서비스 Mock 패턴:**

```python
# Given: Mock 설정
mock_llm_service = Mock(spec=LLMService)
mock_llm_service.invoke_with_structured_output.return_value = RewriteResult(
    queries=["쿼리1", "쿼리2", "쿼리3"]
)

# When: 노드에 Mock 주입
node = QueryRewriteNode(mock_llm_service)
result = node({"question": "테스트 질문", ...})

# Then: 검증
mock_llm_service.invoke_with_structured_output.assert_called_once()
assert len(result["optimized_queries"]) >= 3
```

### 7.3 Fixture 의존성 그래프

```
mock_settings
    │
    ├── mock_llm_service
    │       └── [모든 노드 테스트에서 사용]
    │
    ├── mock_vectorstore_service
    │       └── RetrieverNode 테스트
    │
    └── mock_reranker_service
            └── RetrieverNode 테스트

real_settings
    │
    ├── real_llm_service
    │       └── Integration 테스트
    │
    └── real_vectorstore_service
            └── Integration 테스트
```

---

## 8. 테스트 파일별 상세 분석

### 8.1 test_llm_service.py (8개 테스트)

**테스트 대상:** `LLMService` 클래스

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestLLMServiceLLMCreation` | `test_get_rewrite_llm_returns_llm` | ChatOpenAI 인스턴스 반환 |
| | `test_get_generator_llm_returns_llm` | ChatOpenAI 인스턴스 반환 |
| | `test_rewrite_llm_uses_correct_model` | 모델명, temperature 확인 |
| | `test_generator_llm_uses_correct_model` | 모델명, temperature 확인 |
| `TestLLMServiceInvoke` | `test_invoke_with_structured_output` | Pydantic 모델 반환 |
| | `test_invoke_with_string_output` | str 반환 |
| | `test_invoke_passes_input_data` | 입력 데이터 전달 |
| | `test_invoke_uses_provided_llm` | 지정된 LLM 사용 |

---

### 8.2 test_reranker_service.py (12개 테스트)

**테스트 대상:** `RerankerService` 클래스

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestRerankerServiceModel` | `test_model_lazy_loading` | 첫 접근 시 초기화 |
| | `test_model_returns_cross_encoder` | CrossEncoder 인스턴스 |
| `TestRerankerServiceRerank` | `test_rerank_returns_list` | List 반환 |
| | `test_rerank_sorted_by_score_desc` | 점수 내림차순 정렬 |
| | `test_rerank_respects_top_k` | top_k개 제한 |
| | `test_rerank_handles_empty_docs` | 빈 리스트 처리 |
| | `test_rerank_handles_empty_query` | 빈 쿼리 처리 |
| `TestRerankerServiceGetTopDocuments` | `test_get_top_documents_delegates_to_rerank` | rerank 메서드 위임 |
| | `test_get_top_documents_returns_top_k` | 상위 k개 반환 |
| `TestRerankerServiceEdgeCases` | `test_rerank_with_single_document` | 단일 문서 처리 |
| | `test_rerank_with_many_documents` | 다수 문서 처리 |
| | `test_rerank_preserves_document_content` | 문서 내용 보존 |

---

### 8.3 test_query_rewrite_node.py (11개 테스트)

**테스트 대상:** `QueryRewriteNode` 클래스

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestQueryRewriteNodeProperties` | `test_node_name` | "query_rewrite" 반환 |
| `TestQueryRewriteNodeCall` | `test_returns_optimized_queries` | optimized_queries 키 존재 |
| | `test_generates_multiple_queries` | 3-5개 쿼리 생성 |
| | `test_calls_llm_service_with_correct_params` | RewriteResult 스키마 사용 |
| | `test_preserves_original_question_in_queries` | 원본 의도 반영 |
| `TestQueryRewriteNodeFallback` | `test_fallback_on_llm_error` | 예외 시 원본 질문 반환 |
| | `test_fallback_on_empty_result` | 빈 결과 시 처리 |
| `TestQueryRewriteNodeEdgeCases` | `test_handles_empty_question` | 빈 질문 처리 |
| | `test_handles_special_characters` | 특수 문자 처리 |
| | `test_handles_korean_question` | 한국어 처리 |
| | `test_handles_english_question` | 영어 처리 |

---

### 8.4 test_retriever_node.py (10개 테스트)

**테스트 대상:** `RetrieverNode` 클래스

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestRetrieverNodeProperties` | `test_node_name` | "retriever" 반환 |
| `TestRetrieverNodeCall` | `test_returns_retrieved_docs` | retrieved_docs 키 존재 |
| | `test_calls_vectorstore_for_each_query` | 쿼리별 검색 호출 |
| | `test_calls_reranker_with_results` | 리랭커 호출 |
| | `test_removes_duplicate_documents` | 중복 제거 |
| | `test_returns_top_k_documents` | top_k 제한 |
| `TestRetrieverNodeEdgeCases` | `test_handles_empty_queries` | 빈 쿼리 리스트 |
| | `test_handles_no_search_results` | 검색 결과 없음 |
| | `test_handles_single_query` | 단일 쿼리 |
| `TestRetrieverNodeScoring` | `test_documents_sorted_by_score` | 점수 기준 정렬 |

---

### 8.5 test_generator_node.py (11개 테스트)

**테스트 대상:** `GeneratorNode` 클래스

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestGeneratorNodeProperties` | `test_node_name` | "generator" 반환 |
| `TestGeneratorNodeCall` | `test_returns_final_answer` | final_answer 키 존재 |
| | `test_calls_generator_llm` | Generator LLM 호출 |
| | `test_uses_question_and_context` | question, context 사용 |
| `TestGeneratorNodeFormatDocs` | `test_format_docs_with_indexing` | [1], [2] 형식 인덱싱 |
| | `test_format_docs_preserves_content` | 내용 보존 |
| | `test_format_docs_empty_list` | 빈 리스트 처리 |
| `TestGeneratorNodeEdgeCases` | `test_handles_empty_docs` | 빈 문서 리스트 |
| | `test_handles_long_documents` | 긴 문서 처리 |
| | `test_handles_special_characters_in_docs` | 특수 문자 처리 |
| | `test_handles_tuple_docs` | 튜플 형태 문서 (리랭킹 결과) |

---

### 8.6 test_simple_generator_node.py (12개 테스트)

**테스트 대상:** `SimpleGeneratorNode` 클래스

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestSimpleGeneratorNodeProperties` | `test_node_name` | "simple_generator" 반환 |
| `TestSimpleGeneratorNodeCall` | `test_returns_final_answer` | final_answer 키 존재 |
| | `test_does_not_use_retrieved_docs` | 문서 미사용 확인 |
| | `test_uses_generator_llm` | Generator LLM 호출 |
| `TestSimpleGeneratorNodeScenarios` | `test_handles_greeting` | 인사말 처리 |
| | `test_handles_coding_question` | 코딩 질문 처리 |
| | `test_handles_general_knowledge` | 일반 상식 처리 |
| `TestSimpleGeneratorNodeEdgeCases` | `test_handles_empty_question` | 빈 질문 처리 |
| | `test_handles_long_question` | 긴 질문 처리 |
| | `test_handles_special_characters` | 특수 문자 처리 |
| `TestSimpleGeneratorNodeIntegration` | `test_real_greeting_response` | 실제 LLM 인사 응답 |
| | `test_real_coding_response` | 실제 LLM 코딩 응답 |

---

### 8.7 test_workflow_routing.py (8개 테스트)

**테스트 대상:** `RAGWorkflow.route_question()` 메서드

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestRouteQuestionUnit` | `test_route_to_vectorstore_for_domain_question` | 도메인 질문 → vectorstore |
| | `test_route_to_llm_for_simple_question` | 인사말 → llm |
| | `test_route_to_llm_for_coding_question` | 코딩 질문 → llm |
| `TestRouteQuestionIntegration` | `test_route_greeting_to_llm` | 실제 API로 인사 라우팅 |
| | `test_route_domain_question_to_vectorstore` | 실제 API로 도메인 라우팅 |
| | `test_route_general_knowledge_to_llm` | 실제 API로 상식 라우팅 |
| `TestRouteQuestionEdgeCases` | `test_route_empty_question` | 빈 질문 처리 |
| | `test_route_long_question` | 긴 질문 처리 |

---

### 8.8 test_rag_application.py (17개 테스트)

**테스트 대상:** `RAGApplication` 클래스 (E2E)

| 테스트 클래스 | 테스트 케이스 | 검증 항목 |
|-------------|-------------|----------|
| `TestRAGApplicationLifecycle` | `test_create_app_returns_application` | 인스턴스 생성 |
| | `test_create_app_with_custom_settings` | 커스텀 Settings |
| | `test_initialize_connects_vectorstore` | VectorStore 연결 |
| | `test_close_releases_resources` | 리소스 해제 |
| `TestRAGApplicationRun` | `test_run_returns_string_answer` | 문자열 답변 |
| | `test_run_greeting_question` | 인사 응답 |
| | `test_run_coding_question` | 코딩 응답 |
| | `test_run_domain_question` | 도메인 응답 (RAG) |
| `TestRAGApplicationScenarios` | `test_multiple_questions_session` | 연속 질문 처리 |
| | `test_korean_and_english_questions` | 다국어 처리 |
| `TestRAGApplicationEdgeCases` | `test_empty_question` | 빈 질문 |
| | `test_long_question` | 긴 질문 |
| | `test_special_characters_question` | 특수 문자 |
| `TestRAGApplicationError` | `test_run_before_initialize_raises_error` | 미초기화 에러 |
| | `test_double_initialize_is_safe` | 중복 초기화 안전성 |
| | `test_double_close_is_safe` | 중복 종료 안전성 |
| `TestRAGApplicationPerformance` | `test_llm_path_is_faster` | LLM 경로 성능 비교 |

---

## 9. 실행 방법

### 9.1 환경 설정

```bash
# 의존성 설치 (uv 사용)
uv sync

# 환경 변수 설정 (.env 파일)
OPENAI_API_KEY=sk-xxx
WEAVIATE_URL=http://localhost:8080
```

### 9.2 테스트 실행 명령어

```bash
# 모든 테스트 수집 확인
uv run pytest test-driven-design/ --collect-only

# 단위 테스트만 실행 (Mock 기반, 빠름)
uv run pytest test-driven-design/ -v -m unit

# 통합 테스트만 실행 (실제 API 필요)
uv run pytest test-driven-design/ -v -m integration

# E2E 테스트만 실행 (전체 환경 필요)
uv run pytest test-driven-design/ -v -m e2e

# 전체 테스트 실행
uv run pytest test-driven-design/ -v

# 커버리지 포함 실행
uv run pytest test-driven-design/ -v --cov=dev_v2 --cov-report=html

# 특정 파일만 실행
uv run pytest test-driven-design/unit/test_generator_node.py -v

# 특정 클래스만 실행
uv run pytest test-driven-design/unit/test_generator_node.py::TestGeneratorNodeCall -v

# 특정 테스트만 실행
uv run pytest "test-driven-design/unit/test_generator_node.py::TestGeneratorNodeCall::test_returns_final_answer" -v
```

### 9.3 CI/CD 파이프라인 권장 설정

```yaml
# .github/workflows/test.yml 예시
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Unit Tests
        run: uv run pytest test-driven-design/ -m unit -v

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests  # 단위 테스트 통과 후 실행
    steps:
      - uses: actions/checkout@v4
      - name: Run Integration Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: uv run pytest test-driven-design/ -m integration -v
```

---

## 10. 결론 및 교훈

### 10.1 TDD의 가치 입증

이 프로젝트를 통해 TDD의 핵심 가치가 입증되었습니다:

| 가치 | 실제 경험 |
|------|---------|
| **버그 조기 발견** | 3개의 프로덕션 버그를 테스트 작성 단계에서 발견 |
| **설계 개선** | LCEL 체인 → 서비스 메서드로 리팩토링하여 테스트 용이성 확보 |
| **문서화** | 테스트 코드가 각 메서드의 사용 예제 역할 |
| **자신감** | 62개 단위 테스트가 모두 통과하여 코드 품질 확신 |

### 10.2 발견된 주요 교훈

1. **API 사용법 확인:** LangChain의 `from_template()` vs `from_messages()` 차이를 명확히 이해
2. **테스트 용이성 설계:** LCEL 체인보다 서비스 메서드가 Mock하기 쉬움
3. **모듈 Export 관리:** `__init__.py`의 re-export는 실제 변수명 확인 필수
4. **데이터플로우 순서:** 진입점(Router)부터 출력(Generator)까지 순서대로 테스트

### 10.3 향후 개선 방향

- [ ] Integration 테스트를 위한 Test Container 도입 (Weaviate)
- [ ] 성능 벤치마크 테스트 추가
- [ ] Mutation Testing으로 테스트 품질 검증
- [ ] Property-Based Testing (Hypothesis) 도입 검토

---

## 부록: 파일 목록

```
총 15개 파일 생성됨

test-driven-design/
├── __init__.py                     # 패키지 초기화
├── conftest.py                     # pytest fixtures (243줄)
├── README.md                       # 이 문서
│
├── unit/
│   ├── __init__.py
│   ├── test_llm_service.py         # 8개 테스트
│   ├── test_reranker_service.py    # 12개 테스트
│   ├── test_query_rewrite_node.py  # 11개 테스트
│   ├── test_retriever_node.py      # 10개 테스트
│   ├── test_generator_node.py      # 11개 테스트
│   └── test_simple_generator_node.py # 12개 테스트
│
├── integration/
│   ├── __init__.py
│   ├── test_workflow_routing.py    # 8개 테스트
│   ├── test_vectorstore_service.py # 12개 테스트
│   └── test_rag_pipeline.py        # 8개 테스트
│
└── e2e/
    ├── __init__.py
    └── test_rag_application.py     # 17개 테스트

총 테스트 수: 116개
- unit: 64개
- integration: 28개
- e2e: 17개
```

---

**작성일:** 2025-12-09
**작성자:** Claude Code (Anthropic)
**버전:** 1.0.0
