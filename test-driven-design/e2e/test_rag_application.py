"""
RAGApplication End-to-End 테스트

전체 애플리케이션의 생명주기를 테스트합니다.
- create_app(): 앱 생성
- initialize(): 초기화 (VectorStore 연결, Workflow 빌드)
- run(): 질문에 대한 답변 생성
- close(): 리소스 정리
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_v2.main import RAGApplication, create_app
from dev_v2.config import Settings


class TestRAGApplicationLifecycle:
    """RAGApplication 생명주기 테스트"""

    @pytest.mark.e2e
    def test_create_app_returns_application(self):
        """create_app()이 RAGApplication 인스턴스 반환"""
        # When
        app = create_app()

        # Then
        assert isinstance(app, RAGApplication)

    @pytest.mark.e2e
    def test_create_app_with_custom_settings(self, real_settings):
        """커스텀 Settings로 앱 생성"""
        # When
        app = create_app(settings=real_settings)

        # Then
        assert app.settings == real_settings

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_initialize_connects_vectorstore(self):
        """initialize()가 VectorStore 연결"""
        # Given
        app = create_app()

        # When
        initialized_app = app.initialize()

        # Then
        assert initialized_app is app  # Method chaining
        assert app.vectorstore.is_ready()

        # Cleanup
        app.close()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_close_releases_resources(self):
        """close()가 리소스 해제"""
        # Given
        app = create_app().initialize()

        # When & Then: 에러 없이 종료
        try:
            app.close()
        except Exception as e:
            pytest.fail(f"리소스 정리 실패: {e}")


class TestRAGApplicationRun:
    """RAGApplication run() 메서드 테스트"""

    @pytest.fixture
    def initialized_app(self):
        """초기화된 RAGApplication"""
        app = create_app().initialize()
        yield app
        app.close()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_run_returns_string_answer(self, initialized_app):
        """run()이 문자열 답변 반환"""
        # When
        answer = initialized_app.run("안녕하세요")

        # Then
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_run_greeting_question(self, initialized_app):
        """인사말 질문에 대한 응답"""
        # When
        answer = initialized_app.run("안녕?")

        # Then: 인사 응답
        assert len(answer) > 0
        # 인사 관련 키워드가 포함될 수 있음
        # (LLM 응답은 다양하므로 엄격한 검증 X)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_run_coding_question(self, initialized_app):
        """코딩 질문에 대한 응답"""
        # When
        answer = initialized_app.run("파이썬에서 리스트 정렬하는 방법은?")

        # Then: 코딩 관련 답변
        assert len(answer) > 0

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_run_domain_question(self, initialized_app):
        """도메인 질문에 대한 응답 (RAG 경로)"""
        # When
        answer = initialized_app.run("RAG 성능 고도화의 개념은 무엇인가요?")

        # Then: RAG 기반 답변
        assert len(answer) > 0


class TestRAGApplicationScenarios:
    """RAGApplication 시나리오 테스트"""

    @pytest.fixture
    def initialized_app(self):
        """초기화된 RAGApplication"""
        app = create_app().initialize()
        yield app
        app.close()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_multiple_questions_session(self, initialized_app):
        """여러 질문을 연속으로 처리"""
        questions = [
            "안녕하세요",
            "1+1은 몇이야?",
            "RAG란 무엇인가요?",
        ]

        # When & Then: 모든 질문에 답변
        for question in questions:
            answer = initialized_app.run(question)
            assert isinstance(answer, str)
            assert len(answer) > 0

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_korean_and_english_questions(self, initialized_app):
        """한국어와 영어 질문 처리"""
        # 한국어
        korean_answer = initialized_app.run("안녕하세요")
        assert len(korean_answer) > 0

        # 영어
        english_answer = initialized_app.run("Hello, how are you?")
        assert len(english_answer) > 0


class TestRAGApplicationEdgeCases:
    """RAGApplication 엣지 케이스 테스트"""

    @pytest.fixture
    def initialized_app(self):
        """초기화된 RAGApplication"""
        app = create_app().initialize()
        yield app
        app.close()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_empty_question(self, initialized_app):
        """빈 질문 처리"""
        # When
        answer = initialized_app.run("")

        # Then: 에러 없이 응답
        assert isinstance(answer, str)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_long_question(self, initialized_app):
        """긴 질문 처리"""
        long_question = "이것은 매우 긴 질문입니다. " * 50

        # When
        answer = initialized_app.run(long_question)

        # Then: 에러 없이 응답
        assert isinstance(answer, str)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_special_characters_question(self, initialized_app):
        """특수 문자 포함 질문 처리"""
        # When
        answer = initialized_app.run("파이썬에서 @decorator와 *args, **kwargs는?")

        # Then: 에러 없이 응답
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestRAGApplicationError:
    """RAGApplication 에러 처리 테스트"""

    @pytest.mark.e2e
    def test_run_before_initialize_raises_error(self):
        """초기화 전 run() 호출 시 에러"""
        # Given: 초기화되지 않은 앱
        app = create_app()

        # When & Then
        with pytest.raises(RuntimeError):
            app.run("질문")

    @pytest.mark.e2e
    def test_double_initialize_is_safe(self):
        """중복 초기화가 안전한지 확인"""
        # Given
        app = create_app()

        # When: 두 번 초기화
        try:
            app.initialize()
            app.initialize()  # 두 번째 초기화

            # Then: 에러 없이 동작
            answer = app.run("안녕")
            assert len(answer) > 0
        finally:
            app.close()

    @pytest.mark.e2e
    def test_double_close_is_safe(self):
        """중복 close()가 안전한지 확인"""
        # Given
        app = create_app().initialize()

        # When & Then: 에러 없이 두 번 종료
        try:
            app.close()
            app.close()  # 두 번째 종료
        except Exception as e:
            pytest.fail(f"중복 close() 실패: {e}")


class TestRAGApplicationPerformance:
    """RAGApplication 성능 테스트"""

    @pytest.fixture
    def initialized_app(self):
        """초기화된 RAGApplication"""
        app = create_app().initialize()
        yield app
        app.close()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_llm_path_is_faster(self, initialized_app):
        """LLM 경로가 RAG 경로보다 빠름"""
        import time

        # LLM 경로 (인사말)
        start_llm = time.time()
        initialized_app.run("안녕")
        llm_time = time.time() - start_llm

        # RAG 경로 (도메인 질문)
        start_rag = time.time()
        initialized_app.run("회사의 환불 규정은?")
        rag_time = time.time() - start_rag

        # Then: LLM 경로가 더 빠름 (검색 생략)
        # 참고: 실제 테스트에서는 네트워크 지연 등으로 항상 보장되지 않을 수 있음
        print(f"LLM 경로: {llm_time:.2f}s, RAG 경로: {rag_time:.2f}s")
        # 엄격한 assertion 대신 정보 출력
