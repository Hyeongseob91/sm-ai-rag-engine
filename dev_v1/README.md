# dev_v1: Research & Design Phase

> RAG 파이프라인의 핵심 컴포넌트를 Jupyter Notebook으로 개별 설계하고 검증하는 연구 단계

---

## 목차

1. [개요](#1-개요)
2. [폴더 구조](#2-폴더-구조)
3. [노트북별 상세 분석](#3-노트북별-상세-분석)
4. [검증된 기술 스택](#4-검증된-기술-스택)
5. [v2로의 발전 방향](#5-v2로의-발전-방향)
6. [실행 방법](#6-실행-방법)

---

## 1. 개요

### 1.1 프로젝트 개발 로드맵

```
┌─────────────────────────────────────────────────────────────────────┐
│  ★ Phase 1: Research & Design (dev_v1) ← 현재 단계                  │
│  - Jupyter Notebook으로 개별 기능 테스트                              │
│  - 설계 구조 검토 및 검증                                             │
│  - 기술 스택 선정                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: Prototype (dev_v2)                                        │
│  - 검토된 기능을 Scripts 단위로 모듈화                                │
│  - Prototype 구축 및 TDD 테스트                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: MVP (예정)                                                 │
│  - Docker 컨테이너화                                                 │
│  - 보안 강화 및 베타 서비스 배포                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 목적

| 목적 | 설명 |
|------|------|
| **기술 검증** | RAG 파이프라인의 핵심 기술 선정 및 PoC |
| **설계 검토** | LangGraph 상태 관리, 노드 설계 패턴 학습 |
| **프로토타이핑** | 각 단계별 동작 확인 및 파라미터 튜닝 |
| **문서화** | 개념 정리 및 의사결정 근거 기록 |

### 1.3 연구 결과 요약

| 단계 | 검증 내용 | 상태 |
|------|----------|------|
| Query Rewrite | 다중 쿼리 생성, 모호성 제거 | ✅ 완료 |
| Retriever | Hybrid Search + Reranking | ✅ 완료 |
| Generator | 할루시네이션 방지 프롬프트 | ✅ 설계 완료 |
| 통합 테스트 | LangGraph 전체 파이프라인 | ⏳ v2에서 진행 |

---

## 2. 폴더 구조

```
dev_v1/
├── step1_query_rewrite_design.ipynb   # Query Rewrite 설계 (13.6 KB)
├── step2_retriever_node_design.ipynb  # Retriever 설계 (29.5 KB)
├── step3_generator_node_design.ipynb  # Generator 설계 (4.2 KB)
└── my_weaviate_data/                  # Weaviate 테스트 데이터 (736 KB)
    ├── modules.db
    ├── schema.db
    ├── classifications.db
    ├── raft/raft.db
    └── advancedrag_chunk/             # 테스트 컬렉션 데이터
        └── EvZHKgfAUqpw/
            ├── main.hnsw.commitlog.d/  # HNSW 인덱스
            ├── lsm/                     # LSM Tree
            │   ├── property__id/
            │   ├── property_doc_id/
            │   └── property_content/
            └── ...
```

---

## 3. 노트북별 상세 분석

### 3.1 Step 1: Query Rewrite Design

**파일:** `step1_query_rewrite_design.ipynb`

#### 3.1.1 목적

사용자의 자연어 질문을 벡터 데이터베이스 검색에 최적화된 **다중 쿼리**로 변환합니다.

#### 3.1.2 LangGraph 상태 관리 개념

이 노트북에서는 LangGraph의 핵심 개념을 상세히 설명합니다:

**전역 상태 정의 (RAGState):**
```python
class RAGState(TypedDict):
    question: str              # 원본 질문
    optimized_queries: List[str]  # 최적화된 쿼리 리스트
    retrieved_docs: List[str]  # 검색된 문서
    final_answer: str          # 최종 답변
```

**Partial Schema 패턴:**
- 각 노드는 자신이 담당할 필드만 업데이트
- 스키마에 정의되지 않은 필드는 자동 필터링
- 예: QueryRewriteNode는 `{"optimized_queries": [...]}` 만 반환

#### 3.1.3 구현 코드 핵심

**프롬프트 전략:**
```python
QUERY_REWRITE_SYSTEM_PROMPT = """
당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 RAG 시스템에서 더 효과적인 검색 결과를 얻을 수 있도록
3~5개의 다양한 버전으로 재작성하세요.

지침:
1. 모호한 표현이나 대명사(그, 그것, 이것 등)를 명확한 명사로 교체하세요.
2. 원본 질문의 핵심 의도를 유지하되, 다양한 키워드와 구문을 사용하세요.
3. 검색에 불필요한 경어나 조사는 간결하게 줄이세요.
4. 원본 질문이 한국어면 한국어로, 영어면 영어로 유지하세요.
"""
```

**Pydantic 구조화 출력:**
```python
class RewriteResult(BaseModel):
    queries: List[str] = Field(
        description="최적화된 검색 쿼리 리스트 (3~5개)"
    )

# LLM 호출
structured_llm = llm.with_structured_output(RewriteResult)
chain = prompt | structured_llm
result = chain.invoke({"question": user_question})
```

#### 3.1.4 테스트 결과

**입력:**
```
"그거 환불 규정이 어떻게 돼? 아이폰 산거 말이야"
```

**출력 (5개 쿼리):**
```python
[
    '아이폰 환불 규정은 어떻게 되나요?',
    '아이폰 구매 후 환불 절차는 무엇인가요?',
    '아이폰 환불 정책에 대해 알고 싶어요.',
    '아이폰을 환불하려면 어떤 조건이 필요한가요?',
    '아이폰 환불 규정에 대한 자세한 정보가 필요합니다.'
]
```

**분석:**
- 대명사 "그거" → "아이폰"으로 명확화
- 다양한 표현으로 검색 Recall 향상
- 검색 의도 유지

---

### 3.2 Step 2: Retriever Node Design

**파일:** `step2_retriever_node_design.ipynb`

#### 3.2.1 목적

최적화된 쿼리를 사용하여 **하이브리드 검색** 후 **Reranking**으로 정밀도를 높입니다.

#### 3.2.2 Weaviate 설정

**벡터 DB 구성:**
```python
# Embedded Weaviate (로컬 실행)
client = weaviate.connect_to_embedded(
    version="1.27.0",
    persistence_data_path="./my_weaviate_data",
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    environment_variables={
        "ENABLE_MODULES": "text2vec-openai",
        "DEFAULT_VECTORIZER_MODULE": "text2vec-openai",
    },
)
```

**컬렉션 스키마:**
```python
collection = client.collections.create(
    name="AdvancedRAG_Chunk",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="doc_id", data_type=DataType.INT),
    ],
    inverted_index_config=Configure.inverted_index(
        bm25_b=0.75,   # 문서 길이 정규화
        bm25_k1=1.2,   # 키워드 빈도 포화
    ),
)
```

#### 3.2.3 테스트 데이터 구성

총 **50개** 문서로 다양성을 확보:

| 카테고리 | 문서 수 | 예시 |
|---------|--------|------|
| AI & LLM Core | 10개 | RAG, 프롬프트 엔지니어링, Fine-tuning |
| Vector DB | 10개 | Weaviate, ChromaDB, HNSW, Faiss |
| Python & Engineering | 10개 | GIL, Docker, FastAPI, Kubernetes |
| Finance (노이즈) | 6개 | ETF, ISA, Fed, 비트코인 |
| Food & Lifestyle (노이즈) | 6개 | 스테이크, 파스타, 와인 |
| General Knowledge (노이즈) | 8개 | 광합성, 양자역학, 임진왜란 |

**노이즈 데이터의 목적:**
- 의도하지 않은 관련성 없는 문서 검색 방지 테스트
- Reranking의 효과 측정

#### 3.2.4 Hybrid Search 구현

```python
def hybrid_search(query: str, alpha: float = 0.5, limit: int = 10):
    """
    하이브리드 검색 (의미론적 + 키워드)

    Args:
        query: 검색 쿼리
        alpha: 검색 비율 (0=BM25, 1=Vector, 0.5=균형)
        limit: 반환할 문서 수
    """
    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        return_metadata=MetadataQuery(score=True)
    )
    return response.objects
```

**Alpha 파라미터 의미:**
| alpha 값 | 의미 | 적합한 상황 |
|---------|------|-----------|
| 0.0 | 순수 BM25 (키워드) | 정확한 용어 매칭 필요 |
| 0.5 | 균형 (기본값) | 일반적인 검색 |
| 1.0 | 순수 Vector (의미론적) | 유사 의미 검색 |

#### 3.2.5 Reranking 파이프라인

```python
from sentence_transformers import CrossEncoder

# CrossEncoder 모델 로드
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank_documents(query: str, documents: List[str], top_k: int = 5):
    """
    CrossEncoder 기반 리랭킹

    검색 결과를 Query-Document 쌍으로 재채점하여
    가장 관련성 높은 문서를 선별
    """
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    # 점수 기준 정렬
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_k]
```

**Reranking의 필요성 (Lost in the Middle 문제):**
- 초기 검색(Recall) → 많은 문서 확보 (10~30개)
- Reranking(Precision) → 상위 문서 선별 (5개)
- LLM 컨텍스트 윈도우 내에서 중요 정보가 중간에 묻히는 문제 방지

#### 3.2.6 테스트 결과

**입력:**
```
"RAG성능 고도화의 개념은?"
```

**Reranking 후 결과:**
```
1등 (0.0919점): RAG는 외부 데이터베이스에서 지식을 가져와...
2등 (0.0090점): Fine-tuning은 모델의 파라미터를 업데이트하지만...
3등 (0.0001점): Semantic Chunking은 문장 간의 의미적 유사도를...
```

**분석:**
- 1등 문서가 2등보다 10배 높은 점수
- CrossEncoder가 RAG 관련 문서를 정확하게 식별
- 노이즈 데이터(금융, 음식 등)는 상위에 노출되지 않음

---

### 3.3 Step 3: Generator Node Design

**파일:** `step3_generator_node_design.ipynb`

#### 3.3.1 목적

검색된 문서를 기반으로 **할루시네이션 없이** 정확한 답변을 생성합니다.

#### 3.3.2 프롬프트 전략

**시스템 프롬프트:**
```python
GENERATOR_SYSTEM_PROMPT = """
당신은 기업 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

핵심 지침:
1. 오직 제공된 문서의 정보만을 사용하여 답변하세요.
2. 절대로 외부 지식이나 일반 상식을 섞지 마세요.
3. 각 문장의 출처를 대괄호 번호로 표기하세요. 예: [1], [2]
4. 문서에서 답을 찾을 수 없으면 "제공된 문서에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
5. 명확하고 전문적인 용어를 사용하되, 간결하게 작성하세요.
"""
```

**휴먼 프롬프트:**
```python
GENERATOR_HUMAN_PROMPT = """
[참고 문서]
{context}

[질문]
{question}

위 문서를 참고하여 질문에 답변해주세요.
"""
```

#### 3.3.3 할루시네이션 방지 기법

| 기법 | 설명 |
|------|------|
| **문서 한정** | "오직 제공된 문서만 사용" 명시 |
| **출처 표기** | [1], [2] 형식으로 근거 명시 |
| **모름 인정** | 문서에 없으면 솔직히 인정 |
| **Temperature 0** | 창의성 억제, 사실 기반 응답 |

#### 3.3.4 모델 선택 전략

| 단계 | 모델 | 이유 |
|------|------|------|
| Query Rewrite | GPT-4o-mini | 단순 변환 작업, 비용 효율 |
| Generator | GPT-4o | 복잡한 추론, 높은 정확도 필요 |

**비용 최적화:**
- 전처리(Query Rewrite)는 저렴한 모델
- 핵심 답변 생성만 고성능 모델 사용

---

## 4. 검증된 기술 스택

### 4.1 핵심 프레임워크

```python
# LLM 관련
langchain-openai>=1.1.0      # ChatOpenAI, Embeddings
langchain-core               # ChatPromptTemplate, StrOutputParser
pydantic                     # 구조화 출력 (BaseModel)

# 벡터 DB
weaviate-client>=4.0.0       # Embedded Weaviate
  - text2vec-openai          # OpenAI 임베딩 모듈
  - Hybrid Search            # BM25 + Dense Vector

# 리랭킹
sentence-transformers>=5.1.2  # CrossEncoder
  - BAAI/bge-reranker-v2-m3  # 다국어 리랭커

# 그래프 오케스트레이션 (계획)
langgraph>=1.0.4             # StateGraph, END
```

### 4.2 선정 근거

| 기술 | 선정 이유 |
|------|---------|
| **Weaviate** | 하이브리드 검색 네이티브 지원, 로컬 임베디드 모드 |
| **text-embedding-3-small** | 비용 효율, 충분한 성능 |
| **bge-reranker-v2-m3** | 한국어 지원, 다국어 리랭킹 |
| **GPT-4o** | 복잡한 추론, 지시 따르기 능력 |
| **LangGraph** | 상태 기반 워크플로우, 조건부 분기 |

### 4.3 검증된 파라미터

```python
# Weaviate BM25 설정
bm25_b = 0.75    # 문서 길이 정규화 (0~1)
bm25_k1 = 1.2    # 키워드 빈도 포화 (1.2~2.0)

# Hybrid Search
alpha = 0.5      # Dense/Sparse 균형

# Retriever
initial_limit = 30  # 초기 검색 문서 수
top_k = 5           # 리랭킹 후 선별 수

# LLM
temperature = 0     # 사실 기반, 창의성 억제
```

---

## 5. v2로의 발전 방향

### 5.1 통합 필요 사항

| 항목 | v1 상태 | v2 요구사항 |
|------|--------|-----------|
| 노드 통합 | 개별 노트북 | LangGraph StateGraph |
| 상태 관리 | 개념 설명 | 실제 구현 |
| 에러 처리 | 기초적 | Fallback, Retry |
| 설정 관리 | 하드코딩 | Settings 클래스 |
| 테스트 | 수동 확인 | pytest TDD |

### 5.2 개선 포인트

**Query Rewrite:**
- 동적 쿼리 개수 (질문 복잡도 기반)
- HyDE (Hypothetical Document Embeddings) 추가 검토

**Retriever:**
- Multi-turn 검색 (관련성에 따라 재검색)
- Metadata 필터링 (카테고리, 날짜)

**Generator:**
- RAG-Fusion (여러 쿼리 결과 통합)
- Chain-of-Thought 적용

**평가 시스템:**
- RAGAS 메트릭 도입
- Faithfulness, Relevance, Coherence 측정

### 5.3 v2에서 구현된 추가 기능

v2에서는 다음 기능이 추가되었습니다:

| 기능 | 설명 |
|------|------|
| **Router Pattern** | 질문 유형에 따라 RAG/LLM 경로 분기 |
| **SimpleGeneratorNode** | 검색 없이 직접 답변 (인사, 코딩 질문) |
| **Dependency Injection** | 서비스 → 노드 주입으로 테스트 용이성 확보 |
| **TDD 테스트** | 116개 테스트 케이스 |

---

## 6. 실행 방법

### 6.1 환경 설정

```bash
# 가상환경 생성 (uv 사용)
uv sync

# 환경 변수 설정
export OPENAI_API_KEY=sk-...
```

### 6.2 노트북 실행

```bash
# Jupyter 시작
jupyter notebook

# 순서대로 실행
# 1. step1_query_rewrite_design.ipynb
# 2. step2_retriever_node_design.ipynb
# 3. step3_generator_node_design.ipynb
```

### 6.3 주의사항

- Step 2 실행 시 Weaviate가 `my_weaviate_data/` 폴더에 데이터를 저장합니다
- OpenAI API 키가 필요합니다
- CrossEncoder 모델 다운로드에 시간이 소요될 수 있습니다

---

## 부록: 주요 테스트 데이터 예시

**AI & LLM 카테고리:**
```
- RAG는 외부 데이터베이스에서 지식을 가져와 LLM의 답변을 보강합니다.
- 프롬프트 엔지니어링은 Few-shot, Chain-of-Thought 등의 기법을 포함합니다.
- Fine-tuning은 모델의 파라미터를 업데이트하지만, RAG는 파라미터를 바꾸지 않습니다.
```

**Vector DB 카테고리:**
```
- Weaviate는 GraphQL 기반 벡터 검색 엔진으로 Hybrid Search를 지원합니다.
- HNSW는 고차원 벡터 검색을 위한 근사 최근접 이웃 알고리즘입니다.
- Faiss는 메타에서 개발한 효율적인 유사도 검색 라이브러리입니다.
```

---

**작성일:** 2025-12-09
**버전:** 1.0.0
**다음 단계:** dev_v2 Prototype 구축
