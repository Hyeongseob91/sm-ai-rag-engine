# SM-AI RAG Engine dev_v3: 단계별 개발 로드맵

## Executive Summary

본 문서는 "Why Do RAG Fail?" 세미나에서 도출된 인사이트를 기반으로, SM-AI RAG Engine의 **점진적 고도화 전략**을 수립합니다. 핵심 원칙은 **"동작하는 시스템 우선, 최적화는 그 다음"**입니다.

| **작성일** | 2025. 12. 12 | **작성자** | AI Software Engineer |
| :--- | :--- | :--- | :--- |
| **버전** | dev_v3 | **상태** | Planning |

---

## 1. 개발 단계 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SM-AI RAG Engine dev_v3 Roadmap                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: API Serving                                               │
│  ─────────────────────                                              │
│  "dev_v2를 그대로 API로 서빙"                                         │
│  • FastAPI REST API 구축                                            │
│  • Docker 컨테이너화                                                 │
│  • 기본 모니터링 (Health Check)                                      │
│                                                                     │
│              ↓                                                      │
│                                                                     │
│  Phase 2: Inference Optimization                                    │
│  ────────────────────────────────                                   │
│  "TEI 활용 최적화 방안 연구"                                          │
│  • TEI (Text Embeddings Inference) 도입 연구                         │
│  • Python GIL 병목 해결                                              │
│  • Semantic Chunking / Reranking 속도 개선                           │
│                                                                     │
│              ↓                                                      │
│                                                                     │
│  Phase 3: Retriever Enhancement                                     │
│  ──────────────────────────────                                     │
│  "Retriever 고도화 적용 방법 연구"                                    │
│  • SPLADE (Sparse Vector) 도입                                      │
│  • 한국어 형태소 분석기 + BM25                                        │
│  • Self-Correction (Self-Reflective RAG)                            │
│  • CR Loss 기반 임베딩 Fine-tuning                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: API Serving (dev_v2 → Production)

### 2.1 목표
현재 dev_v2의 RAG 파이프라인을 **변경 없이** REST API로 서빙하여, 즉시 사용 가능한 프로덕션 환경 구축

### 2.2 세미나 인사이트 근거
> "Performance First: 정확도보다 속도(Latency)와 처리량(Throughput)이 사용자 경험에 더 치명적"
> — 우아한형제들, Session 2

### 2.3 구현 범위

#### A. FastAPI REST API
```
엔드포인트 설계:
─────────────────────────────────────────────────────
POST /api/v1/query              # RAG 질의
POST /api/v1/upload             # 문서 업로드 및 전처리
GET  /api/v1/collections        # 컬렉션 목록
GET  /api/v1/health             # 헬스체크
DELETE /api/v1/collection/{name} # 컬렉션 삭제
```

**구현 구조**:
```
dev_v3/
├── api/
│   ├── main.py                 # FastAPI 앱 진입점
│   ├── routes/
│   │   ├── query.py            # /query 엔드포인트
│   │   ├── upload.py           # /upload 엔드포인트
│   │   └── admin.py            # 관리 엔드포인트
│   ├── schemas.py              # API 요청/응답 모델
│   └── dependencies.py         # DI 설정 (기존 RAGApplication 연동)
```

**예시 코드**:
```python
# api/main.py
from fastapi import FastAPI
from dev_v2.main import RAGApplication

app = FastAPI(title="SM-AI RAG Engine", version="3.0")
rag_app = RAGApplication()  # 기존 dev_v2 그대로 사용

@app.post("/api/v1/query")
async def query(request: QueryRequest):
    result = rag_app.workflow.invoke(request.question)
    return {"answer": result["final_answer"], "sources": result["retrieved_docs"]}
```

#### B. Docker 컨테이너화
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
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

#### C. 기본 모니터링
```python
# 헬스체크 엔드포인트
@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "weaviate": rag_app.vectorstore.is_ready(),
        "document_count": rag_app.vectorstore.get_document_count()
    }
```

### 2.4 예상 산출물
| 산출물 | 설명 |
|--------|------|
| `dev_v3/api/` | FastAPI 애플리케이션 코드 |
| `docker-compose.yml` | 컨테이너 오케스트레이션 |
| `Dockerfile` | Python 3.11 + UV 기반 이미지 |
| API 문서 | Swagger UI 자동 생성 (`/docs`) |

### 2.5 성공 기준
- [ ] `docker compose up` 1-command로 전체 시스템 구동
- [ ] `/api/v1/query` 엔드포인트 정상 응답
- [ ] `/api/v1/upload` 문서 업로드 및 전처리 동작
- [ ] Weaviate 데이터 영속성 확보 (볼륨 마운트)

---

## 3. Phase 2: TEI 활용 최적화 방안 연구

### 3.1 목표
Python 기반 추론의 **구조적 한계(GIL)**를 극복하고, Embedding/Reranking 작업의 **처리량(Throughput)을 5~10배 향상**

### 3.2 세미나 인사이트 근거
> "Performance First: 서비스 관점에서는 답변의 정확도보다 응답 속도와 동시 처리량이 사용자 경험(UX)에 더 치명적"
> — 우아한형제들, Session 2

### 3.3 현재 문제점 분석

#### A. Python GIL 병목
```
현재 dev_v2 아키텍처:
─────────────────────────────────────────────────────
sentence-transformers (Python)
    ↓
[GIL Lock] → 멀티코어 활용 불가
    ↓
Semantic Chunking 시 수만 개 문장 임베딩 → 병목 발생
```

**측정 대상**:
- Semantic Chunking: 문장 단위 임베딩 (수천~수만 개)
- Reranking: Top-30 문서 CrossEncoder 점수 계산
- Query Embedding: 사용자 질문 임베딩

#### B. 현재 사용 중인 모델
```python
# dev_v2/config/settings.py
embedding_model: "text-embedding-3-small"  # OpenAI (외부 API)
reranker_model: "BAAI/bge-reranker-v2-m3"  # 로컬 CrossEncoder
```

### 3.4 TEI (Text Embeddings Inference) 연구 내용

#### A. TEI 개요
```
TEI (Hugging Face):
─────────────────────────────────────────────────────
• Rust + Candle 기반 고성능 임베딩 서버
• Python 의존성 제거 → GIL 우회
• Flash Attention + Continuous Batching 지원
• Docker 이미지로 쉬운 배포
```

**공식 문서**: https://huggingface.co/docs/text-embeddings-inference

#### B. TEI 아키텍처 연구
```
┌─────────────────────────────────────────────────────────────┐
│                    TEI Server (Rust)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ HTTP Server │ →  │  Tokenizer  │ →  │  Model Backend  │  │
│  │   (Actix)   │    │   (Fast)    │    │ (Candle/Flash)  │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         ↑                                       ↓           │
│         │              Continuous Batching      │           │
│         └───────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

#### C. 적용 방안 연구

**Option 1: Embedding 전용 TEI**
```yaml
# docker-compose.yml 확장
tei-embed:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
  command: --model-id BAAI/bge-m3 --port 8081
  ports:
    - "8081:8081"
```

**Option 2: Reranking 전용 TEI**
```yaml
tei-rerank:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
  command: --model-id BAAI/bge-reranker-v2-m3 --port 8082
  ports:
    - "8082:8082"
```

**Option 3: 통합 구성**
```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   RAG API      │ →   │  TEI Embed     │     │  TEI Rerank    │
│  (FastAPI)     │     │  (Port 8081)   │     │  (Port 8082)   │
└────────────────┘     └────────────────┘     └────────────────┘
        ↓                      ↓                      ↓
┌─────────────────────────────────────────────────────────────┐
│                      Weaviate                               │
└─────────────────────────────────────────────────────────────┘
```

#### D. 클라이언트 래퍼 설계
```python
# dev_v3/services/tei.py (연구 설계)
import httpx
from typing import List

class TEIService:
    """TEI 클라이언트 래퍼"""

    def __init__(self, embed_url: str, rerank_url: str = None):
        self._embed_url = embed_url
        self._rerank_url = rerank_url
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩"""
        response = await self._client.post(
            f"{self._embed_url}/embed",
            json={"inputs": texts}
        )
        return response.json()

    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        """문서 재순위화"""
        response = await self._client.post(
            f"{self._rerank_url}/rerank",
            json={"query": query, "texts": documents}
        )
        return [r["score"] for r in response.json()]
```

### 3.5 벤치마크 계획

| 측정 항목 | 현재 (sentence-transformers) | 목표 (TEI) |
|-----------|------------------------------|------------|
| Semantic Chunking (1000문장) | 측정 필요 | 5~10배 향상 |
| Reranking (30문서) | 측정 필요 | ms 단위 |
| Query Embedding | 측정 필요 | <10ms |
| 동시 처리량 (QPS) | 측정 필요 | 10배 향상 |

### 3.6 연구 산출물
| 산출물 | 설명 |
|--------|------|
| 벤치마크 결과 문서 | 현재 vs TEI 성능 비교 |
| TEI 통합 설계 문서 | 아키텍처 및 API 설계 |
| POC 코드 | TEIService 프로토타입 |
| docker-compose 확장 | TEI 서버 구성 |

### 3.7 연구 질문
- [ ] OpenAI Embedding과 TEI 로컬 모델의 품질 비교?
- [ ] GPU vs CPU 환경에서의 TEI 성능 차이?
- [ ] Weaviate 내장 임베딩 vs TEI 외부 호출 트레이드오프?
- [ ] 기존 dev_v2 코드 변경 최소화 방안?

---

## 4. Phase 3: Retriever 고도화 적용 방법 연구

### 4.1 목표
세미나에서 도출된 **업계 Blue Ocean 기술**을 SM-AI RAG Engine에 적용하는 방법론 연구

### 4.2 세미나 인사이트 근거
> "SPLADE와 같은 최신 희소 검색 기법에 대한 인지도는 낮음 (Blue Ocean)"
> — Q&A Session

> "검색기가 나쁜 것이 아니라, 데이터가 검색 가능한 형태로 맵핑되지 않은 것"
> — 2.3절 Hallucination & Retrieval

> "한국어 BM25에는 반드시 한국어 형태소 분석기를 결합해야 유의미한 검색 성능 확보"
> — 브레인크루, Session 3

### 4.3 연구 영역

#### A. SPLADE (Sparse Lexical and Expansion)

**개념**:
```
BM25 한계:
─────────────────────────────────────────────────────
Query: "건축 허가 절차"
→ 정확히 "건축", "허가", "절차" 포함 문서만 매칭
→ "인허가", "승인", "신청" 등 유사 용어 놓침

SPLADE 해결:
─────────────────────────────────────────────────────
Query: "건축 허가 절차"
→ 학습된 희소 벡터로 확장
→ "인허가", "건축법", "승인", "신청서" 등 연관 용어까지 매칭
```

**연구 내용**:
```python
# 연구 대상 모델
models = [
    "naver/splade-cocondenser-ensembledistil",  # 범용
    "naver/splade-v3",                           # 최신
]

# 적용 방안
class SPLADEService:
    def encode_query(self, query: str) -> Dict[int, float]:
        """질문 → 희소 벡터"""

    def encode_document(self, doc: str) -> Dict[int, float]:
        """문서 → 희소 벡터 (전처리 시 1회)"""
```

**Weaviate 통합 옵션**:
1. Weaviate SPLADE Module (내장)
2. 별도 인덱스 + 점수 병합 (외부)

#### B. 한국어 형태소 분석기 + BM25

**현재 문제**:
```
Weaviate BM25 기본 토크나이저:
─────────────────────────────────────────────────────
"건축허가절차" → ["건축허가절차"]  # 분리 안 됨
"건축 허가를 받으려면" → ["건축", "허가를", "받으려면"]  # 조사 포함
```

**목표**:
```
한국어 형태소 분석기 적용:
─────────────────────────────────────────────────────
"건축허가절차" → ["건축", "허가", "절차"]
"건축 허가를 받으려면" → ["건축", "허가", "받다"]  # 원형 복원
```

**연구 대상 형태소 분석기**:
| 분석기 | 장점 | 단점 |
|--------|------|------|
| **Kiwi** | 순수 Python, 설치 쉬움 | 속도 중간 |
| **Mecab-ko** | 속도 빠름 | 설치 복잡 (시스템 의존성) |
| **Okt (KoNLPy)** | 범용적 | 속도 느림 |

**통합 방안 연구**:
```python
# 옵션 1: 전처리 시 토큰화
class KoreanTokenizer:
    def tokenize(self, text: str) -> List[str]:
        """형태소 분석 후 명사/동사 추출"""

# 옵션 2: Weaviate 커스텀 토크나이저
# (Weaviate 설정에서 외부 토크나이저 연동 가능 여부 연구)
```

#### C. Self-Correction (Self-Reflective RAG)

**개념**:
```
기존 RAG:
─────────────────────────────────────────────────────
Query → Retrieve → Generate → Answer (끝)
→ 환각 발생해도 검증 없이 출력

Self-Correction RAG:
─────────────────────────────────────────────────────
Query → Retrieve → Generate → Grade → (부족하면 재검색)
                                  ↓
                              Answer (충분할 때)
```

**연구 내용**:
```python
# Grading 기준 연구
class GradeResult(BaseModel):
    relevance: float      # 검색 문서 관련성 (0-1)
    groundedness: float   # 답변 근거 충실도 (0-1)
    completeness: float   # 답변 완전성 (0-1)

    @property
    def should_retry(self) -> bool:
        return min(self.relevance, self.groundedness) < 0.7
```

**워크플로우 확장 연구**:
```python
# LangGraph 조건부 분기
workflow.add_conditional_edges(
    "grader",
    lambda state: "retry" if state["should_retry"] and state["retry_count"] < 2 else "end",
    {"retry": "query_rewrite", "end": END}
)
```

#### D. CR Loss 기반 임베딩 Fine-tuning

**개념**:
```
현재 문제:
─────────────────────────────────────────────────────
"환불 규정이 뭐야?" vs "환불 정책 알려줘"
→ 같은 의미지만 다른 벡터로 인식 (Lexical Variation 민감)

CR Loss 해결:
─────────────────────────────────────────────────────
• QEA (Query Embedding Alignment): 같은 의미 → 같은 벡터
• SMC (Similarity Margin Consistency): 변형에도 일관된 점수
```

**연구 내용**:
- CR Loss 논문 분석 (Amazon AGI, 2025)
- 학습 데이터셋 구축 방안 (질문 변형 페어)
- Fine-tuning 파이프라인 설계

### 4.4 3-way Hybrid Search 최종 설계

```
┌─────────────────────────────────────────────────────────────┐
│                    3-way Hybrid Search                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: "건축 허가 절차가 뭔가요?"                            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Dense    │  │   SPLADE    │  │  BM25 (한국어)      │  │
│  │  (40%)      │  │   (30%)     │  │  (30%)              │  │
│  │             │  │             │  │                     │  │
│  │  의미 기반   │  │  확장 키워드 │  │  정확 키워드         │  │
│  │  검색       │  │  검색       │  │  검색               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         ↓                ↓                   ↓              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Score Fusion (RRF)                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           CrossEncoder Reranking                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│                     Top-5 Results                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 연구 산출물

| 산출물 | 설명 |
|--------|------|
| SPLADE 적용 가이드 | 모델 선정, Weaviate 통합 방안 |
| 한국어 토크나이저 비교 문서 | Kiwi vs Mecab vs Okt 벤치마크 |
| Self-Correction 설계 문서 | GraderNode 스펙, 워크플로우 |
| CR Loss 연구 노트 | 논문 분석, 적용 가능성 평가 |
| 3-way Hybrid Search 설계서 | 점수 병합 알고리즘, 가중치 튜닝 |

### 4.6 연구 질문
- [ ] SPLADE 한국어 성능은 어떠한가? (ko-SPLADE 필요 여부)
- [ ] Weaviate에서 SPLADE를 어떻게 통합하는가?
- [ ] Self-Correction 시 LLM 비용 증가는 얼마인가?
- [ ] CR Loss Fine-tuning에 필요한 데이터 양은?
- [ ] 형태소 분석기의 Docker 배포 복잡도는?

---

## 5. 전체 로드맵 타임라인

```
┌──────────┬──────────────────────────────────────────────────────┐
│  Phase   │                      내용                            │
├──────────┼──────────────────────────────────────────────────────┤
│          │                                                      │
│ Phase 1  │  [============] API Serving                          │
│ (1-2주)  │  • FastAPI + Docker                                  │
│          │  • dev_v2 그대로 서빙                                 │
│          │                                                      │
├──────────┼──────────────────────────────────────────────────────┤
│          │                                                      │
│ Phase 2  │  [============] TEI 최적화 연구                       │
│ (2-3주)  │  • 벤치마크 수행                                      │
│          │  • TEI 통합 POC                                       │
│          │                                                      │
├──────────┼──────────────────────────────────────────────────────┤
│          │                                                      │
│ Phase 3  │  [============] Retriever 고도화 연구                 │
│ (3-4주)  │  • SPLADE, 한국어 토크나이저                          │
│          │  • Self-Correction, CR Loss                          │
│          │                                                      │
└──────────┴──────────────────────────────────────────────────────┘
```

---

## 6. 핵심 파일 경로

### 참조 문서
| 파일 | 설명 |
|------|------|
| `docs/plus_docs.md` | 세미나 인사이트 분석 (206줄) |
| `docs/dev_v3_plan.md` | 기존 개발 계획 (750줄) |

### dev_v2 핵심 코드 (확장 기반)
| 파일 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| `dev_v2/main.py` | API 연동 | - | - |
| `dev_v2/services/vectorstore.py` | - | - | 3-way Hybrid |
| `dev_v2/services/reranker.py` | - | TEI 전환 | - |
| `dev_v2/graph/workflow.py` | - | - | Self-Correction |
| `dev_v2/config/settings.py` | API 설정 | TEI 설정 | SPLADE 설정 |

---

## 7. 결론

본 로드맵은 **"동작하는 시스템 우선"** 원칙에 따라 설계되었습니다:

1. **Phase 1 (API Serving)**: dev_v2를 즉시 프로덕션에 배포할 수 있는 상태로 만듦
2. **Phase 2 (TEI 최적화)**: 성능 병목을 해결하여 서비스 품질 향상
3. **Phase 3 (Retriever 고도화)**: 세미나 인사이트 기반 Blue Ocean 기술 적용

각 Phase는 독립적으로 완료 가능하며, 이전 Phase의 안정성을 확보한 후 다음 단계로 진행합니다.

---

**[End of dev_v3 Roadmap Document]**
