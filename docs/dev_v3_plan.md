# SM-AI RAG Engine: dev_v3 Development Plan

## Subject: Phase 3 - Production MVP & Advanced RAG Capabilities

| **작성일** | 2025. 12. 11 | **작성자** | AI Software Engineer |
| :--- | :--- | :--- | :--- |
| **버전** | dev_v3 | **상태** | Planning |

---

## 1. Executive Summary (개요)

### 1.1 Vision
**dev_v3**는 SM-AI RAG Engine의 세 번째 개발 단계로, dev_v2에서 구현된 프로토타입을 **프로덕션 수준의 MVP(Minimum Viable Product)**로 격상시키는 것을 목표로 한다.

### 1.2 Evolution Path
```
dev_v1 (Research)     →  dev_v2 (Prototype)      →  dev_v3 (MVP)
─────────────────────────────────────────────────────────────────
• Jupyter Notebook       • Python 모듈화            • Production Infrastructure
• 기술 검증               • TDD 테스트 (116개)        • Docker 컨테이너화
• 설계 문서화             • LangGraph 워크플로우       • FastAPI REST API
                        • Semantic Chunking        • SPLADE + Self-Correction
                        • Hybrid Search + Reranking • Multi-Modal 지원
```

### 1.3 Key Objectives
1. **정밀도 강화**: SPLADE, Self-Querying Retriever 도입
2. **추론 능력**: Self-Reflective RAG (자기 교정 루프)
3. **멀티모달**: 도면/이미지/표 처리 능력
4. **프로덕션 인프라**: Docker, FastAPI, 보안, 모니터링

---

## 2. Current State Analysis (현재 상태 분석)

### 2.1 dev_v2 Implementation Status

| 구분 | 현황 | 코드량 |
|------|------|--------|
| **Services** | LLM, VectorStore, Reranker, Preprocessing | 약 400줄 |
| **Nodes** | QueryRewrite, Retriever, Generator, SimpleGenerator | 약 250줄 |
| **Graph** | RAGWorkflow (Router Pattern) | 약 130줄 |
| **Config** | Settings (Pydantic/dataclass) | 약 80줄 |
| **Schemas** | State, Models, Preprocessing | 약 200줄 |
| **Tests** | Unit(64) + Integration(28) + E2E(17) | 116개 |

### 2.2 Implemented Technologies

```
✅ 구현 완료                    ⏳ dev_v3 예정
─────────────────────────────────────────────────
✅ Semantic Chunking           ⏳ SPLADE (Sparse Vector)
✅ Hybrid Search (BM25+Dense)  ⏳ Self-Querying Retriever
✅ CrossEncoder Reranking      ⏳ Self-Correction Loop
✅ Query Rewrite (Multi-Query) ⏳ Image Captioning
✅ LangGraph Router Pattern    ⏳ Table → Markdown
✅ TDD (pytest)                ⏳ Docker + FastAPI
✅ Weaviate Embedded           ⏳ Prometheus + Grafana
```

---

## 3. Gap Analysis (기술 갭 분석)

### 3.1 Industry Pain Points vs SM-AI Solutions

| 업계 한계 | dev_v2 해결 | dev_v3 목표 |
|-----------|-------------|-------------|
| Context Fragmentation | ✅ Semantic Chunking | 유지 |
| Static Pipeline | ✅ LangGraph Router | Self-Correction 추가 |
| BM25 한계 | ✅ Hybrid Search | **SPLADE 도입** |
| 단순 Prompt Tuning | ✅ Query Rewrite | **Self-Querying** |
| 정성 평가 의존 | ⚠️ TDD 부분 해결 | **RAGAS 정량 평가** |
| 도면/이미지 처리 불가 | ❌ 미구현 | **Multi-Modal** |

### 3.2 Blue Ocean Opportunities
컨퍼런스 Q&A에서 확인된 업계 미착수 영역:
- **SPLADE**: 현업 엔지니어 인지도 낮음 → 선점 기회
- **Self-Correction**: "환각 = 전처리 실패"라는 인식 → 런타임 검증으로 차별화
- **Multi-Modal**: UI/UX 우회가 아닌 기술적 해결책 제시

---

## 4. dev_v3 Core Features (핵심 기능)

### Phase 3-1: Precision Enhancement (정밀도 강화)

#### 4.1.1 SPLADE (Sparse Lexical and Expansion)

**개념**: 학습된 희소 벡터(Sparse Vector)로 키워드 검색의 한계 극복

**현재 (BM25)**:
```
Query: "건축 허가 절차"
→ 정확히 "건축", "허가", "절차" 포함 문서만 매칭
```

**SPLADE 적용 후**:
```
Query: "건축 허가 절차"
→ 학습된 확장: "인허가", "건축법", "승인", "신청서" 등 연관 용어까지 매칭
```

**구현 계획**:
```
dev_v3/
├── services/
│   └── sparse_encoder.py      # SPLADE 인코더 서비스
│       ├── SPLADEEncoder
│       ├── encode_query()
│       └── encode_document()
│
├── nodes/
│   └── retriever.py           # 기존 확장
│       └── hybrid_search()    # Dense + Sparse(SPLADE) + BM25
```

**기술 스택**:
- 모델: `naver/splade-cocondenser-ensembledistil`
- 통합: Weaviate SPLADE module 또는 별도 인덱스

#### 4.1.2 Self-Querying Retriever

**개념**: LLM이 메타데이터 필터를 자동 생성하여 검색 범위 축소

**예시**:
```
사용자: "2024년에 작성된 계약서에서 위약금 조항을 찾아줘"

Self-Querying 분석:
├── Query: "위약금 조항"
└── Filter: { "file_type": "docx", "created_at": {"$gte": "2024-01-01"} }
```

**구현 계획**:
```python
# dev_v3/services/self_query.py
class SelfQueryService:
    def parse_query(self, question: str) -> tuple[str, dict]:
        """질문 → (검색 쿼리, 메타데이터 필터) 분리"""

# dev_v3/nodes/self_query_node.py
class SelfQueryNode(BaseNode):
    def __call__(self, state: RAGState) -> dict:
        query, filters = self.service.parse_query(state["question"])
        return {"optimized_query": query, "metadata_filters": filters}
```

---

### Phase 3-2: Agentic Capability (추론 능력 강화)

#### 4.2.1 Self-Correction / Self-Reflective RAG

**개념**: Generator가 생성한 답변을 스스로 검증하고, 부족하면 재검색

**워크플로우**:
```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
[Query] → [Retrieve] → [Generate] → [Grade] ──(불충분)──┘
                                       │
                                   (충분)
                                       ▼
                                  [Final Answer]
```

**Grading 기준**:
1. **Relevance**: 검색 문서가 질문과 관련 있는가?
2. **Groundedness**: 답변이 검색 문서에 근거하는가?
3. **Completeness**: 질문에 완전히 답했는가?

**구현 계획**:
```python
# dev_v3/nodes/grader.py
class GraderNode(BaseNode):
    """답변 품질 평가 노드"""

    def __call__(self, state: RAGState) -> dict:
        grade = self.llm.invoke_with_structured_output(
            GradeResult,
            prompt=GRADING_PROMPT.format(
                question=state["question"],
                documents=state["documents"],
                answer=state["answer"]
            )
        )
        return {
            "grade": grade,
            "should_retry": grade.score < 0.7
        }

# dev_v3/graph/workflow.py
def build(self):
    # ... 기존 노드들 ...
    self.graph.add_node("grader", self.grader_node)

    # 조건부 분기: 재시도 or 종료
    self.graph.add_conditional_edges(
        "grader",
        lambda state: "retry" if state["should_retry"] else "end",
        {"retry": "query_rewrite", "end": END}
    )
```

**최대 재시도 횟수**: 2회 (무한 루프 방지)

---

### Phase 3-3: Multi-Modal (멀티모달)

#### 4.3.1 Image Captioning

**개념**: PDF 내 도면/이미지를 텍스트 설명으로 변환

**파이프라인**:
```
PDF 페이지 → 이미지 추출 → Vision LLM → Caption 생성 → 청킹에 포함
```

**구현 계획**:
```python
# dev_v3/services/preprocessing/image_processor.py
class ImageProcessor:
    def __init__(self, vision_llm: str = "gpt-4o"):
        self.client = OpenAI()

    def extract_images_from_pdf(self, pdf_path: str) -> list[Image]:
        """PDF에서 이미지 추출"""

    def generate_caption(self, image: Image, context: str) -> str:
        """이미지에 대한 설명 생성"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"이 이미지를 설명해주세요. 문맥: {context}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }]
        )
        return response.choices[0].message.content
```

#### 4.3.2 Table Extraction to Markdown

**개념**: PDF/DOCX 내 표를 Markdown 형식으로 변환하여 LLM이 이해 가능하게 처리

**현재 vs 목표**:
```
현재 (pdfplumber):
"항목금액비고재료비100만원VAT별도인건비200만원월급여"  # 셀 구분 없음

목표 (Markdown):
| 항목 | 금액 | 비고 |
|------|------|------|
| 재료비 | 100만원 | VAT별도 |
| 인건비 | 200만원 | 월급여 |
```

**구현 계획**:
```python
# dev_v3/services/preprocessing/table_processor.py
class TableProcessor:
    def extract_tables_from_pdf(self, pdf_path: str) -> list[Table]:
        """PDF에서 표 추출 (pdfplumber 고급 기능)"""

    def table_to_markdown(self, table: Table) -> str:
        """표 → Markdown 변환"""

    def enhance_table_context(self, markdown_table: str, surrounding_text: str) -> str:
        """표에 문맥 정보 추가"""
```

---

## 5. Production Infrastructure (프로덕션 인프라)

### 5.1 Docker 컨테이너화

**구조**:
```
docker/
├── Dockerfile              # Python 3.11 + UV
├── docker-compose.yml      # 서비스 오케스트레이션
├── .dockerignore
└── config/
    └── weaviate.yaml       # Weaviate 설정
```

**docker-compose.yml 예시**:
```yaml
version: '3.8'
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - weaviate

  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.27.0
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

### 5.2 FastAPI REST API

**엔드포인트 설계**:
```
POST /api/v1/query           # RAG 질의
POST /api/v1/upload          # 문서 업로드 및 전처리
GET  /api/v1/collections     # 컬렉션 목록
GET  /api/v1/health          # 헬스체크

POST /api/v1/admin/reindex   # 재인덱싱
DELETE /api/v1/admin/collection/{name}  # 컬렉션 삭제
```

**구현 계획**:
```python
# dev_v3/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SM-AI RAG Engine", version="3.0")

class QueryRequest(BaseModel):
    question: str
    collection: str = "default"
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = await rag_workflow.ainvoke(request.question)
    return QueryResponse(**result)
```

### 5.3 보안 (Secret Manager)

**계층**:
```
1. 환경변수: 로컬 개발용 (.env)
2. Docker Secrets: 컨테이너 환경
3. Cloud Secret Manager: 프로덕션 (AWS/GCP/Azure)
```

**구현**:
```python
# dev_v3/config/secrets.py
import os
from abc import ABC, abstractmethod

class SecretProvider(ABC):
    @abstractmethod
    def get_secret(self, key: str) -> str: ...

class EnvSecretProvider(SecretProvider):
    def get_secret(self, key: str) -> str:
        return os.getenv(key)

class AWSSecretProvider(SecretProvider):
    def get_secret(self, key: str) -> str:
        # boto3로 Secrets Manager 조회
        ...
```

### 5.4 로깅 & 모니터링

**Structured Logging**:
```python
# dev_v3/config/logging.py
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# 사용 예
logger.info("query_processed",
    question=question,
    latency_ms=latency,
    documents_retrieved=len(docs))
```

**모니터링 스택**:
```
Prometheus (메트릭 수집) → Grafana (대시보드)
                              │
                              ├─ 요청 처리량 (QPS)
                              ├─ 응답 시간 (P50, P95, P99)
                              ├─ 검색 정확도 (Precision@K)
                              └─ 에러율
```

---

## 6. Implementation Strategy (구현 전략)

### 6.1 Phase별 구현 순서 (Agentic 우선)

```
Phase 3-0: Infrastructure (1주차)
├── Docker 환경 구축
├── FastAPI 기본 엔드포인트
└── 기존 dev_v2 코드 마이그레이션

Phase 3-1: Agentic (2-3주차) ⭐ 최우선
├── Grader 노드 (답변 품질 평가)
├── Self-Correction 워크플로우 (재검색 루프)
├── 최대 재시도 로직 (무한 루프 방지)
└── 환각률 감소 효과 측정

    📌 우선 배치 이유:
    - 환각 문제가 사용자 신뢰도에 가장 치명적
    - Self-Correction만으로 환각률 30% 감소 가능
    - 기존 dev_v2 구조에서 가장 빠르게 통합 가능

Phase 3-2: Precision (4-5주차)
├── SPLADE 인코더 서비스
├── Self-Querying Retriever
└── 통합 테스트

Phase 3-3: Multi-Modal (6-7주차)
├── Image Processor
├── Table Processor
└── 전처리 파이프라인 통합

Phase 3-4: Production (8주차)
├── Secret Manager 통합
├── Prometheus 메트릭
├── Grafana 대시보드
└── 부하 테스트
```

**Phase 순서 변경 근거**:
| 기준 | Agentic 우선 | Precision 우선 |
|------|--------------|----------------|
| 사용자 체감 효과 | **즉시** (환각 감소) | 점진적 (정확도 향상) |
| 구현 복잡도 | **낮음** (기존 구조 활용) | 높음 (새 인프라 필요) |
| 비용 | API 호출 증가 | GPU/모델 비용 |
| ROI | **높음** (신뢰도 즉시 개선) | 중간 |

### 6.2 디렉토리 구조

```
dev_v3/
├── __init__.py
├── main.py                    # DI Container (dev_v2 확장)
│
├── api/                       # [NEW] FastAPI 레이어
│   ├── __init__.py
│   ├── main.py               # FastAPI 앱
│   ├── routes/
│   │   ├── query.py
│   │   ├── upload.py
│   │   └── admin.py
│   └── dependencies.py       # DI for FastAPI
│
├── config/
│   ├── settings.py           # 기존 확장
│   ├── secrets.py            # [NEW] Secret Provider
│   └── logging.py            # [NEW] Structured Logging
│
├── schemas/
│   ├── state.py              # RAGState 확장
│   ├── models.py             # 기존 + GradeResult
│   ├── preprocessing.py      # 기존
│   └── api.py                # [NEW] API 요청/응답 모델
│
├── services/
│   ├── llm.py                # 기존
│   ├── vectorstore.py        # 기존
│   ├── reranker.py           # 기존
│   ├── sparse_encoder.py     # [NEW] SPLADE
│   ├── self_query.py         # [NEW] Self-Querying
│   └── preprocessing/
│       ├── parsers.py        # 기존
│       ├── normalizer.py     # 기존
│       ├── chunking.py       # 기존
│       ├── pipeline.py       # 기존
│       ├── image_processor.py   # [NEW] Image Captioning
│       └── table_processor.py   # [NEW] Table → Markdown
│
├── nodes/
│   ├── base.py               # 기존
│   ├── query_rewrite.py      # 기존
│   ├── self_query_node.py    # [NEW]
│   ├── retriever.py          # 기존 (SPLADE 통합)
│   ├── generator.py          # 기존
│   ├── simple_generator.py   # 기존
│   └── grader.py             # [NEW] 답변 품질 평가
│
├── graph/
│   └── workflow.py           # 기존 (Self-Correction 추가)
│
├── prompts/
│   └── templates.py          # 기존 + GRADING_PROMPT
│
├── docker/                   # [NEW]
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
└── monitoring/               # [NEW]
    ├── prometheus.yml
    └── grafana/
        └── dashboards/
```

### 6.3 테스트 전략

**기존 116개 테스트 유지 + 확장**:
```
test-driven-design/
├── unit/
│   ├── test_sparse_encoder.py     # [NEW] SPLADE 테스트
│   ├── test_self_query_service.py # [NEW]
│   ├── test_grader_node.py        # [NEW]
│   ├── test_image_processor.py    # [NEW]
│   └── test_table_processor.py    # [NEW]
│
├── integration/
│   ├── test_self_correction_flow.py  # [NEW]
│   └── test_multimodal_pipeline.py   # [NEW]
│
├── e2e/
│   └── test_api_endpoints.py      # [NEW] FastAPI 테스트
│
└── performance/                   # [NEW]
    ├── test_latency.py
    └── test_throughput.py
```

---

## 7. Rationale & Evidence (근거 및 증거)

### 7.1 SPLADE 도입 근거

**컨퍼런스 인사이트**:
> "SPLADE와 같은 최신 희소 검색(Sparse Search) 기법에 대한 인지도는 낮음 (Blue Ocean)"
> - Q&A Session, 2.4절

**기술적 근거**:
1. BM25의 한계: 정확한 키워드 매칭만 가능, 동의어/유사어 처리 불가
2. Dense Vector의 한계: 전문 용어(건축 코드명 등) 검색에 취약
3. SPLADE 장점: 학습된 희소 벡터로 두 한계를 동시 극복

**증거**:
- BEIR 벤치마크에서 SPLADE는 BM25 대비 평균 15% 성능 향상
- 도메인 특화 용어 검색에서 특히 효과적 (논문: SPLADE v2, SIGIR 2022)

**선택 이유**: 업계가 아직 인지하지 못한 기술로 선점 효과 + 건축 도메인 특수 용어 처리에 적합

---

### 7.2 Self-Correction 도입 근거

**컨퍼런스 인사이트**:
> "Bad Retriever의 본질: 검색기가 나쁜 것이 아니라, 데이터가 검색 가능한 형태로 맵핑되지 않은 '전처리 실패'"
> - 2.3절 Hallucination & Retrieval

**문제 인식**:
- 업계는 환각(Hallucination)을 전처리 단계에서만 해결하려 함
- 런타임 검증(Runtime Validation)이 부재

**기술적 근거**:
1. 전처리를 아무리 잘해도 100% 완벽할 수 없음
2. Generator가 스스로 "근거 부족"을 인식하면 재검색 가능
3. Self-Reflective RAG는 단순 RAG 대비 환각률 30% 감소 (논문: Self-RAG, ICLR 2024)

**선택 이유**: 전처리 + 런타임 검증의 이중 방어선 구축으로 환각 최소화

---

### 7.3 Multi-Modal 도입 근거

**컨퍼런스 인사이트**:
> "Deep Reasoning 한계: 도면 등 복잡한 계층 구조 데이터 처리는 현재 LLM만으로 불가능하여 UI/UX로 우회 해결 중"
> - Q&A Session, 2.4절

**문제 인식**:
- 업계는 도면/이미지 처리를 "불가능"으로 치부하고 UI로 우회
- 기술적 해결책을 제시하지 못함

**기술적 근거**:
1. GPT-4 Vision의 이미지 이해 능력이 대폭 향상
2. 이미지 → 텍스트 Caption으로 변환하면 기존 RAG 파이프라인 활용 가능
3. 표(Table)도 Markdown 변환으로 LLM이 이해 가능

**선택 이유**: 업계가 우회하는 문제를 기술적으로 해결하여 차별화

---

### 7.4 프로덕션 인프라 근거

**컨퍼런스 인사이트**:
> "Performance First: 정확도보다 속도(Latency)와 처리량(Throughput)이 사용자 경험에 더 치명적"
> - 2.2절 Evaluation & Success Criteria

**기술적 근거**:
1. Docker 컨테이너화: 환경 일관성, 배포 자동화
2. FastAPI: 비동기 처리로 높은 동시성 지원
3. Prometheus + Grafana: 실시간 성능 모니터링

**선택 이유**: 정확도만큼 성능도 중요하며, 프로덕션 환경에서 측정 가능한 지표 필요

---

### 7.5 Self-Querying Retriever 근거

**컨퍼런스 인사이트**:
> "Golden Dataset: 전문가(SME)가 검증한 '정답셋' 구축이 선행되어야 성능 평가가 가능"
> - 2.1절 Data Processing & Chunking

**문제 인식**:
- Golden Dataset 수준의 정확도를 위해서는 검색 범위를 좁혀야 함
- 사용자가 메타데이터 필터를 직접 지정하기 어려움

**기술적 근거**:
1. LLM이 자연어 질문에서 메타데이터 필터를 자동 추출
2. "2024년 계약서" → `file_type=docx, year=2024` 자동 변환
3. 검색 범위 축소로 Precision 향상

**선택 이유**: 자연어로 정밀한 검색 가능, Golden Dataset 수준의 정확도 목표

---

## 8. Business Value Analysis (비즈니스 가치 분석)

### 8.1 ROI (투자 대비 효과)

| 투자 항목 | 예상 비용 | 기대 효과 |
|-----------|-----------|-----------|
| **SPLADE 모델** | GPU 서버 or 클라우드 비용 | 검색 정확도 15% 향상 → 사용자 만족도 증가 |
| **Self-Correction** | LLM API 호출 증가 (최대 3배) | 환각률 30% 감소 → 신뢰도 향상, 재작업 감소 |
| **GPT-4 Vision** | 이미지당 $0.01~0.03 | 도면/이미지 질의 가능 → 기존 불가능 영역 해결 |
| **인프라** | Docker/모니터링 구축 비용 | 배포 자동화 → 운영 비용 80% 절감 |

### 8.2 비용-편익 분석

**현재 (수작업 기반)**:
```
사용자 질문 → 검색 실패 → 수동 검색 (10분) → 답변 작성 (5분)
= 건당 15분 소요
```

**dev_v3 적용 후**:
```
사용자 질문 → Self-Correction RAG (3초) → 자동 답변
= 건당 3초 소요
```

**예상 절감 효과**:
- 월 1,000건 질의 기준: **249시간** 절감
- 연간 인건비 절감: 약 **3,000만원** (시급 10,000원 기준)

### 8.3 경쟁 우위 확보

| 경쟁사 현황 | SM-AI dev_v3 |
|-------------|--------------|
| BM25 + 단순 RAG | **SPLADE + Hybrid + Reranking** |
| 정적 파이프라인 | **Self-Correction 동적 워크플로우** |
| 텍스트만 처리 | **Multi-Modal (이미지/도면/표)** |
| 정성 평가 의존 | **RAGAS 정량 평가 + 모니터링** |

### 8.4 핵심 성과 지표 (KPI)

| KPI | 현재 | 목표 | 비즈니스 임팩트 |
|-----|------|------|-----------------|
| 답변 정확도 | ~70% | 90%+ | 고객 신뢰도 향상 |
| 응답 시간 | N/A | <3초 | 사용자 경험 개선 |
| 처리 가능 문서 유형 | 텍스트만 | 텍스트+이미지+표 | 활용 범위 확대 |
| 환각률 | ~30% | <10% | 재작업 비용 감소 |

---

## 9. Risk Assessment (리스크 평가 - 비즈니스 리스크 포함)

| 리스크 | 영향도 | 발생 확률 | 완화 전략 |
|--------|--------|-----------|-----------|
| SPLADE 모델 로딩 지연 | 중 | 중 | 사전 로딩 + 캐싱 |
| Self-Correction 무한 루프 | 고 | 저 | 최대 재시도 2회 제한 |
| GPT-4 Vision 비용 | 중 | 고 | 이미지 수 제한 + 캐싱 |
| Weaviate 스케일링 | 중 | 중 | 클러스터 모드 전환 준비 |
| 테스트 커버리지 저하 | 중 | 저 | TDD 원칙 유지 |
| **LLM API 비용 증가** | 중 | 고 | 캐싱, 요청 배치, 모델 경량화 |

---

## 10. Success Metrics (성공 지표)

### 10.1 정량적 지표

| 지표 | 현재 (dev_v2) | 목표 (dev_v3) |
|------|---------------|---------------|
| Precision@5 | 측정 중 | 85% 이상 |
| Recall@30 | 측정 중 | 90% 이상 |
| 응답 시간 (P95) | - | 3초 이내 |
| 환각률 | 측정 중 | 10% 이하 |
| 테스트 커버리지 | 116개 | 150개 이상 |

### 10.2 정성적 지표

- [ ] 도면/이미지 질문에 대한 의미 있는 답변 생성
- [ ] 메타데이터 필터링 자연어 질의 지원
- [ ] Docker 1-command 배포 가능
- [ ] 실시간 성능 대시보드 운영

---

## 11. Conclusion (결론)

**dev_v3**는 dev_v2의 견고한 기반 위에 다음 세 가지 핵심 역량을 추가한다:

1. **정밀도**: SPLADE + Self-Querying으로 검색 정확도 극대화
2. **신뢰성**: Self-Correction으로 환각 최소화
3. **범용성**: Multi-Modal로 도면/이미지/표 처리

이를 통해 SM-AI RAG Engine은 단순한 검색 도구를 넘어, **스스로 생각하고 검증하는 Agentic RAG**로 진화하며, 업계 표준을 뛰어넘는 **Frontier Engine**으로 자리매김할 것이다.

---

**[End of dev_v3 Development Plan]**
