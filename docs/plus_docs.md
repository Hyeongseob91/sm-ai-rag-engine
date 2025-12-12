<RAG를 잘 다루고 싶은 엔지니어 패널 토크>

주제:
Why Do RAG Fail?
장소:
판교테크노밸리 PDC A동 9층 groom
패널:
브레인크루 이경록 CEO (Teddynote)
KT DS 김재현 엔지니어
우아한형제들 김정태 엔지니어
목차
Executive Summary
Industry Key Insights
Session 1. Data Preprocessing & Chunking Strategy
Session 2. RAG Evaluation & Success Criteria
Session 3. Hallucination & Bad Retriever
Session 4. Q&A Session & Key Insights
Gap Analysis
SM-AI' Project vs. Industry Reality
Future Roadmap (Part 3: dev_v3)
Short-term: Dual-Track Retrieval Enhancement
Mid-term: Inference Optimization
Long-term: Agentic Capability
Conclusion
1. Executive Summary
본 문서는 최근 RAG 엔지니어 패널 토크("Why Do RAG Fail?")에서 논의된 업계의 주요 실패 원인과 기술적 난제들을 확인하고, 현재 본인이 연구 및 개발 중인 'SM-AI’에 사용 될 도구 중 ‘RAG Engine’의 아키텍처를 비교하여, 상업적 & 기술적 위치를 확인하는 것을 목적으로 작성하였습니다.
분석 결과, 업계는 여전히 데이터 전처리의 난이도와 정성적 평가의 모호함으로 인해 어려움을 겪고 있습니다. 
현재 SM-RAG Engine의 경우 'Semantic Chunking', 'LangGraph Router', 'Query Rewrite', 'Hybrid Search + Reranking' 등의 고도화 기법을 적용하고 있어, 제품화에 필요한 State of the Practice(SoP)는 갖추고 있는 것으로 평가됩니다.
2. Industry Key Insights
패널이 선택한 RAG 프로젝트의 주요 실패 원인과 현실적인 대응책은 총 4개의 Session으로 구성되었으나, 시간 여건 상 3개의 Session만 토크가 진행되었습니다. 내용은 아래와 같습니다.
Session 1. Data Preprocessing & Chunking Strategy
1. 비정형 데이터(표/이미지) 처리 전략 (Handling Unstructured Data)
핵심 이슈: 데이터의 도메인 성격과 프로젝트 목표에 따라 최적의 처리 방식이 상이하여, 단일한 해결책(Silver Bullet)이 부재함.
실무 접근법 (KT DS):
Case-by-Case 전략: 데이터 특성에 맞춰 유연하게 접근해야 함.
SOTA 및 SaaS 적극 활용: 자체 구축만 고집하기보다, 최신 SOTA(State-of-the-Art) 모델과 상용 SaaS 솔루션을 벤치마킹하고 조합하여 최적의 파이프라인을 구성하는 것이 필수적임.
2. 데이터 정합성 검증 및 품질 관리 (Data QC & Verification)
접근법 1 (KT DS):
정량적 기준 수립과 Golden/Silver Dataset 이론적 배경 (Microsoft Methodology):
Microsoft의 RAG 평가 프레임워크에서 고안된 개념, 데이터셋의 신뢰도와 생성 방식에 따라 등급을 나누어 관리함. 
Golden Dataset: 도메인 전문가(SME)가 직접 검수하여 확보한 '정답(Ground Truth)' 데이터셋. 
품질은 최상이지만 구축 비용이 높음. 최종 성능 평가 및 배포 기준(Acceptance Criteria)으로 활용.
Silver Dataset: LLM을 이용해 합성(Synthetic)하거나 약한 관리(Weak Supervision)하에 생성된 데이터셋. Golden 등급보다는 신뢰도가 낮지만 대량 생산이 가능하여, 초기 모델 튜닝이나 회귀 테스트(Regression Test) 용도로 활용.
활용: KT DS는 이러한 Golden Dataset을 기준점(Benchmark)으로 삼아 모델의 답변 정확도(Accuracy)를 최우선 순위로 측정하고 검증함.
접근법 2 (우아한형제들): 
메타데이터 기반 맥락 강화 
PDF Binary 분류 및 태깅: 텍스트 추출 전, PDF 파일을 Binary 레벨에서 분석하여 문서의 유형(예: 스캔본, 텍스트형, 복합형 등)을 분류하고, 단순 텍스트 외의 구조적 정보를 추출하여 메타데이터 태깅(Metadata Tagging)을 수행함.
Context Injection: 파일의 전반적인 요약 내용을 메타데이터에 강제 주입(Injection)하여, LLM이 파편화된 청크(Chunk)만 보는 것이 아니라 파일 전체의 성격과 요약을 먼저 인지하고 매핑할 수 있도록 유도함.
3. 계층적 구조(Hierarchy/Graph) 데이터 처리
Context Preservation (우아한형제들):
2단계 리트리버(Two-stage Retrieval) 전략:
1단계 (Retrieval): 질문과 유사한 청크를 탐색.
2단계 (Contextual Refinement): 찾은 청크가 속한 상위/하위 맥락(Parent/Child)이나 연결된 그래프 정보를 다시 한번 호출하여 답변 생성에 필요한 문맥을 재구성함.
(예시) 효과: "건강검진 지원 금액"을 검색했을 때, 단순 검색은 텍스트에 포함된 '30만 원'이라는 답변만 추출합니다. 반면 2단계 처리를 적용하면 해당 금액이 '근속 3년 미만 사원'에게만 적용되는 하위 조항임을 인식하고, '임원 및 10년 이상 근속자(50만 원)' 규정과 구분하여 답변함으로써 정보의 정확성과 신뢰도를 확보합니다.
4. 청킹(Chunking) 전략 수립 기준
Service-Oriented Design (우아한형제들):
기능 정의 우선: 구현하려는 최종 기능을 먼저 정의하고 역산하여 전략을 수립.
Cross-Check: LLM이 데이터를 '읽는 방식'과 '이해하는 방식'을 고려하여, 분할된 데이터가 상호 검증(Cross-Check) 가능하도록 설계.
구문적 정합성(Syntax)과 의미적 완결성(Semantics)의 동시 확보
읽는 방식 고려
마크다운 헤더, 표 형식, JSON 구조 등이 청킹 과정에서 파괴되지 않도록 하여, LLM이 데이터의 **구조(Structure)**를 명확히 파싱할 수 있게 함.
이해하는 방식 고려
대명사나 조건문이 포함된 경우, 참조 대상(Entity)이나 전제 조건(Context)이 동일한 청크 내에 포함되거나 메타데이터로 주입되도록 하여, LLM의 **어텐션 메커니즘(Attention Mechanism)**이 올바르게 작동하도록 함.
Cross-Check
위 두 가지가 모두 충족되었을 때 비로소 분할된 데이터가 원본 데이터와 동일한 정보를 담고 있는지 검증 가능함.
Context Bridging과 Manual Mapping (브레인크루):
실무 사례 (도면 리딩 프로젝트): 건축/설계 도면의 경우, 하나의 도면만으로 답을 낼 수 없고 참조 도면(Reference Drawing)을 교차 확인해야만 정보 추출이 가능한 경우가 빈번함.
문제점: LLM은 분산된 도면 간의 암묵적인 연결 고리나 깊이(Depth) 있는 참조 관계를 스스로 파악하여 가져오기 어려움.
해결책 (Bridge 구축): 데이터 정보가 분산되어 있거나 계층 깊이가 깊은 경우, 사람이 직접 매핑(Mapping) 가이드라인을 수립하고 수작업으로 연결 고리(Bridge)를 만들어줘야 함. 자동화가 어렵더라도 이 과정이 선행되어야 RAG의 실질적인 성능을 확보할 수 있음.
Session 2. RAG Evaluation & Success Criteria
1. 평가 지표의 우선순위 (Hierarchy of Metrics)
Performance First (우아한형제들):
속도와 처리량(Latency & Throughput): 
서비스 관점에서는 답변의 정확도보다 응답 속도와 동시 처리량이 사용자 경험(UX)에 더 치명적일 수 있음.
따라서 시스템적 성능 지표를 최우선으로 확보해야 함.
정확도(Accuracy) 지표: 
성능이 확보된 후 검토하는 3대 핵심 지표. 
답변 관련성 (Answer Relevance): 질문에 적절한 답을 했는가?
컨텍스트 관련성 (Context Relevance): 리트리버가 가져온 문서가 질문과 연관이 있는가?
답변 충실성 (Faithfulness): 답변이 가져온 컨텍스트에 기반하여 생성되었는가? (Hallucination 방지)
2. 평가 도구 및 프레임워크 (Tools & Frameworks)
Open Source Frameworks (KT DS):
RAGAS 활용: 정량적 평가를 위해 오픈소스 프레임워크인 RAGAS를 주로 활용.
Customization: 프로젝트의 성격과 PM/고객사의 요구사항에 따라 지표를 커스터마이징하여 사용.
Management Platform (브레인크루):
LangSmith: 단순 평가를 넘어, 조직 단위의 QA 관리와 로그 추적을 위해 LangSmith와 같은 LLM Ops 도구를 적극 활용함.
3. 프로젝트 성공 기준과 합의 (Acceptance Criteria & Agreement)
The "CEO Metric" (정성 평가의 중요성):
실사용자 만족도: 수치화된 정량 지표(Score)보다 중요한 것은 결국 의사결정권자나 실무자가 체감하는 "만족도"임.
Case-by-Case: 프로젝트의 도메인과 목표에 따라 평가의 잣대는 유동적이어야 함.
코너 케이스(Corner Case)와 매몰 비용 방지:
The 80/20 Rule: 테스트 케이스 10개 중 8개가 통과되었다면, 나머지 2개(Corner Case)를 억지로 통과시키려 해서는 안 됨.
Trade-off: 남은 2개를 해결하려다 보면 프롬프트나 로직이 과적합(Overfitting)되어 기존에 잘 동작하던 8개의 성능이 저하되거나(Side Effect), 시스템 복잡도가 기형적으로 증가함.
Pre-Agreement (사전 합의): 따라서 프로젝트 착수 단계(기획 단계)에서 "어느 수준까지 수용할 것인가"에 대한 목표 승인 기준(Success Criteria)을 고객사와 명확히 합의하는 것이 RAG 프로젝트의 성패를 좌우함.
4. Key Takeaway (Insight)
No Silver Bullet in Evaluation: RAG 개발의 최전선(Frontier)에 있는 기업들조차 표준화된 단 하나의 평가 룰을 가지고 있지 않음.
Flexible Strategy: 정량적 지표(RAGAS 등)는 참고용일 뿐, 실제로는 성능과 비즈니스 요구사항(정성 평가) 사이의 균형을 맞추는 것이 RAG 엔지니어의 핵심 역량.
Session 3. Hallucination & Bad Retriever
1. 환각(Hallucination) 저감 전략
Engineering Approach (KT DS):
Advanced RAG & Modularization: 명확한 정답(Silver Bullet)은 존재하지 않음. RAG 파이프라인을 모듈화(Modularization)하여 각 단계별 기능을 세밀하게 조정(Fine-tuning)하고, 다양한 프롬프트 엔지니어링(Prompt Engineering) 기법을 적용하며 최적점을 찾아야 함.
Tracing & Debugging: 오픈소스 Tracing 도구를 도입하여 테스트 과정을 추적하고, 환각이 발생하는 특정 구간(Retrieval 단계인지, Generation 단계인지 등)을 찾아내어 개선하는 엔지니어링 접근이 필요함.
Data-Centric Approach (우아한형제들):
Data Grounding: 모듈 검토와 더불어 근본적으로 "질문에 대한 정답이 우리 데이터베이스(DB)에 실제로 존재하는가?"를 점검해야 함. 양질의 데이터가 없다면 모델은 환각을 일으킬 수밖에 없음.
Strategic Exclusion (브레인크루):
Zero Tolerance: 핀테크(Fintech)와 같이 오답이 치명적인 리스크가 되는 도메인에서는 생성형 AI 사용을 배제하는 것도 하나의 전략임.
AI ≠ Generative AI: 문제 해결을 위해 반드시 생성형 모델을 쓸 필요는 없으며, 상황에 따라 예측형/분류형 AI 등 다른 방식을 선택하는 유연함이 필요.
2. Bad Retriever에 대한 재정의와 해결
Ambiguity of Definition:
'나쁜 검색기(Bad Retriever)'에 대한 정의가 모호하다.
(예: Context 10개 추출 시 실패했으나, 20개 추출 시 성공했다면 이를 단순히 검색기 성능의 문제로 볼 것인가?)
Dependency on Preprocessing:
검색기가 색인된 데이터를 못 찾는 것은 알고리즘의 문제라기보다, 데이터가 검색 가능한 형태로 매핑(Mapping)되지 않았을 확률이 높음.
Key Insight: 검색기 고도화도 중요하지만, 결국 문제의 본질은 데이터 전처리(Preprocessing)와 상세한 매핑 여부로 귀결됨. (Session 1의 중요성 재확인)
3. 임베딩 모델(Embedding Model) 및 검색 전략
Model Selection (브레인크루):
활용 모델: Qwen 계열(Qwen-3-8B, 4B, 0.6B) 등 성능이 검증된 모델을 주로 활용.
Client Constraints: 고객사의 보안 정책이나 정서(중국 모델 기피 등)에 따라 사용할 수 있는 모델이 제한되므로, 이에 맞춰 유연하게 모델을 변경할 수 있어야 함.
Hybrid Search Optimization:BM25 & Tokenizer: 
키워드 검색(BM25)과 벡터 검색을 결합한 하이브리드 검색(Hybrid Search) 시, 한국어의 특성을 반영하기 위해 반드시 한국어 형태소 분석기(Tokenizer)를 BM25에 결합해야만 유의미한 검색 성능을 확보할 수 있음.
Session 4. Q&A Session & Key Insights
현장에서 직접 질의한 내용과 이에 대한 패널의 답변을 정리해보았습니다.

왼)브레인크루 CEO 이경록(TeddyNote)
Q1. 복잡한 도면 데이터의 Multi-hop Reasoning 처리 한계와 UX적 해결
질문 배경:
도면 해석은 단일 도면이 아닌 [평면도 → 재료 마감표 or 구조 일람표 → 상세도 → 단면 상세도]로 이어지는 깊고, 넓은 Depth의 연관성 추론과 상호 참조가 필수적인데, RAG 엔진이 이러한 다층적(Multi-hop) 연결 구조를 어떻게 처리했는지, 실제로 LLM이 이를 해석하고 따라갔는지 질의.
답변:
기술적 한계: 해당 수준의 깊이 있는 추론(Deep Reasoning)은 현재 LLM만으로는 불가능했음. 초기에는 GraphRAG(Neo4j) 도입을 시도했으나 비용 및 구축 난이도 문제로 중단함.
해결책 (UX 중심):
AI 모델의 추론 능력 대신 UI/UX로 문제를 해결함. 사용자가 특정 정보를 요청하면, 관련된 키워드의 검색 결과를 우측 패널에 리스트업(List-up)하여 전문가가 즉시 참고할 수 있도록 보조 도구(Copilot) 형태로 제공하여 프로젝트를 완수함.
Q2. 사내 기술 엔진(R&D) 구축 시 Vector DB 선정 전략
질문 배경:
특정 프로젝트가 아닌, 범용적인 기술 경쟁력 확보를 위한 사내 R&D 단계에서 도메인별로 선호되거나 추천하는 Vector DB(Milvus, Qdrant, Weaviate 등)가 있는지 질의함.
답변:
Legacy 호환성 우선: RAG 도입 기업 대부분은 이미 기존 RDB(PostgreSQL 등)를 운용 중임.
PGVector의 강세:
익숙하지 않은 새로운 전용 Vector DB(Milvus, Qdrant)를 도입하기보다는, 기존 RDB 환경에 플러그인 형태로 붙일 수 있는 PGVector를 압도적으로 많이 사용함. 성능보다는 운영 효율성과 데이터 통합 편의성이 우선시됨.
Q3. Sparse Search 고도화 (SPLADE) 적용 여부
질문 배경:
Retriever 성능 향상을 위해 기존 BM25를 넘어선 SPLADE(Sparse Lexical and Expansion Model) 등의 최신 기법 적용 경험을 질의함.
답변:
적용 경험이 없으며, 해당 기술에 대해 인지하지 못하고 있음.
(현장에서는 아직 보편화되지 않았거나, BM25+Tokenizer 수준에 머물러 있으며, 비효율적인 사항으로 보는듯)
3. Gap Analysis
이번 컨퍼런스에서 논의된 "Why Do RAG Fail?"의 주요 원인들과 현재 진행 중인 'SM-AI (dev_v2)'의 아키텍처를 비교 분석하였습니다.
3.1. SM-AI RAG Engine (dev_v2) Architecture
Core Architecture
LangGraph & Router Pattern: 사용자의 질문 의도를 파악하여 VectorStore(검색) 경로와 LLM(일반 대화) 경로를 동적으로 분기하는 유연한 구조 적용.
Modular Design: Services(로직), Nodes(작업 단위), Graph(흐름)로 철저히 분리된 아키텍처로 유지보수성 및 확장성 확보.
Advanced Data Pipeline
Unified File Parser: PDF, DOCX, XLSX 등 다양한 포맷을 정규화된 RawDocument로 변환.
Semantic Chunking: 기존의 고정 크기(Fixed-size) 분할이 아닌, 임베딩 유사도(Cosine Similarity) 기반의 의미적 청킹을 구현하여 문맥 단절(Context Fragmentation) 문제를 알고리즘으로 해결하려고 함.
Retrieval & Generation Strategy
Query Rewrite: 모호한 사용자 질문을 3~5개의 구체적 쿼리로 확장하여 Recall(재현율) 향상.
Hybrid Search: BM25(키워드) + Dense Vector(의미) 결합.
CrossEncoder Reranking: 검색된 문서(Top-30)를 BGE-M3 모델로 재순위화하여 최종 Top-5의 정밀도(Precision) 극대화.
3.2. Feature Comparison Matrix (기능 비교 분석)

3.3. Technical Deep Dive: Semantic Chunking의 가치
언급된 Bridge 구축(Mapping)과 맥락 단절 방지에 대해 알고리즘적 접근(Semantic Chunking)으로 해답을 제시합니다.
Legacy 방식 (Fixed-size Chunking):
단순히 500자, 1000자 단위로 자름.
문제점: 문장의 중간이 잘리거나, 주제가 바뀌는 지점을 포착하지 못해 리트리버가 엉뚱한 내용을 가져옴.
SM-AI 방식 (Semantic Chunking):
Embedding Similarity: 문장 단위로 임베딩을 수행하고, 인접 문장 간의 유사도(cosine similarity)를 계산.
Dynamic Breakpoint: 유사도가 급격히 떨어지는 지점(Percentile 95%)을 '주제가 바뀌는 지점'으로 인식하여 절단.
Safety Guard: 너무 작거나 큰 청크가 생기지 않도록 min/max_chunk_size로 보정.
효과: 사람이 글을 나누는 것처럼, "의미 덩어리" 단위로 데이터가 저장되므로 검색 정확도가 비약적으로 상승.
4. Future Roadmap (Part 3: dev_v3)
프로덕션 레벨에서의 성능 한계(Python GIL, Query Variation Sensitivity)를 극복하기 위해 품질과 속도,
두 축을 동시에 고도화하는 전략을 수립하였습니다.
4.1. Short-term: Dual-Track Retrieval Enhancement [정밀도 및 강건성 확보]
Dense와 Sparse 검색을 각각 고도화하여, '키워드 불일치'와 '질문 뉘앙스 차이'를 동시에 해결하는 완전 무결한 검색 파이프라인을 구축.
Dense Retriever 고도화: Coherence Ranking (CR) Loss 도입
배경: 현재 Dense Retriever는 "환불 규정이 뭐야?"와 "환불 정책 알려줘"를 다른 벡터로 인식하는 등, 질문의 미세한 어휘 변화(Lexical Variation)에 민감하여 검색 일관성(Coherence)이 떨어지는 문제가 있음.
전략: 최신 연구(Amazon AGI, 2025)에 기반한 Coherence Ranking (CR) Loss 2를 도입하여 임베딩 모델을 추가 학습(Fine-tuning)한다. 
Query Embedding Alignment (QEA): 의미가 같은 질문들은 벡터 공간에서 동일한 지점을 가리키도록 강제
Similarity Margin Consistency (SMC): 질문의 형태가 바뀌어도 문서와의 유사도 점수가 일정하게 유지되도록 조정
기대 효과: 별도의 Query Rewriter(LLM) 없이도 검색 모델 자체가 다양한 질문 변형에 강건해지며, 검색 일관성(RBO)을 약 15% 이상 향상시킬 것으로 기대.
Sparse Retriever 고도화: SPLADE (Learned Sparse Vector) 도입
배경: 현재 사용 중인 BM25는 단순 빈도 기반이라 건축 도메인의 전문 용어(코드명, 약어)나 문맥적 키워드를 학습할 수 없음.
전략: 학습 가능한 희소 벡터인 SPLADE를 도입하여, BM25의 한계를 넘는 'Deep Learning 기반 키워드 매칭'을 구현한다.
기대 효과: "의미 기반(Dense/CR Loss)"과 "키워드 기반(Sparse/SPLADE)"이 상호 보완되어, 현업에서 놓치기 쉬운 Long-tail 질의까지 커버리지 확대.
4.2. Mid-term: Inference Optimization [속도 및 처리량 극대화]
Python 기반 추론의 구조적 한계를 극복하고, 고부하 작업(Semantic Chunking, Reranking)의 실시간성을 확보하기 위해 인프라 레벨의 최적화 필요.
TEI (Text Embeddings Inference) 도입
문제점: 현재 dev_v2는 Python 라이브러리(sentence-transformers)를 사용하고 있어, GIL로 인해 멀티 코어를 온전히 활용하지 못함. 특히 Semantic Chunking 시 수만 개의 문장을 임베딩할 때 병목 발생 위험.
전략: Hugging Face의 TEI (Rust 기반)를 도입하여 서빙 레이어를 교체한다. 
Rust & Candle: Python 의존성을 제거하고 Rust로 재작성된 고성능 백엔드 활용.
Flash Attention & Continuous Batching: 최신 어텐션 최적화 기술과 동적 배칭을 통해 처리량(Throughput)을 극대화.
기대 효과:
Semantic Chunking 속도 5배~10배 향상 예상.
Reranking(BGE-M3) Latency를 밀리초(ms) 단위로 단축하여 실시간 서비스 가능성 확보.
4.3. Long-term: Agentic Capability [추론 및 해결 능력 강화] → Part 4. 이후
단순 검색을 넘어, 스스로 오류를 수정하고 복잡한 데이터를 해석하는 에이전트(Agent)로 진화 필요.
Self-Correction (Self-Reflective RAG)
전략: Generator의 답변을 스스로 검증(Grading)하고, 근거가 부족하면 CR Loss로 학습된 모델을 활용해 질문을 내부적으로 재정의하여 재검색하는 루프(Loop) 구현.
Multi-Modal Parsing (Vision)
전략: 도면 내 이미지를 텍스트로 변환(Captioning)하거나 표(Table)를 Markdown으로 구조화하여, 텍스트뿐만 아니라 시각 정보까지 검색 가능한 데이터베이스 구축.
5. Conclusion
결론적으로, 현재 진행 중인 SM-AI Project와 이에 탑재될 RAG Engine은, 현업에서 겪는 구조적 한계와 고객사의 요구사항을 효과적으로 극복할 수 있는 고도화된 아키텍처를 이미 확보한 것으로 평가됩니다.
현재 상용화 단계의 최대 병목 구간은 '데이터 간의 맥락을 온전히 보존하는 전처리 파이프라인의 자동화'에 있습니다.
이는 당사의 GIS-based R&D Project에서도 핵심 기술로 대두된 만큼, 연구소 차원의 최우선 해결 목표로 설정하여 역량을 집중해야 합니다.
나아가 향후 로드맵인 SPLADE(희소 검색) 및 Coherence Ranking (CR) Loss 도입과 Self-Correction Workflow(자가 교정) 구축이 완료된다면, 단순한 사내 기술 내재화를 넘어 AI 솔루션 시장을 선도하는 'RAG Frontier Engine'으로서 기술적 우위를 확보 할 수 있을 것이라 예상합니다.

[Reference]
작성일: 2025. 12. 11.
작성자 : AX, AI Researcher Hyeongseob Kim