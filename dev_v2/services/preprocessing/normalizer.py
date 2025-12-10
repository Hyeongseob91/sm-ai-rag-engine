"""
Text Normalizer - 텍스트 정규화.

여러 줄바꿈, 공백, 특수문자 등을 정리하여 일관된 형식의 텍스트로 변환합니다.
"""
import re

from ...config import Settings
from ...schemas.preprocessing import RawDocument


class TextNormalizer:
    """텍스트 정규화 클래스.

    설정에 따라 불필요한 공백 제거, 특수문자 필터링, 짧은 줄 제거 등의 정규화 작업을 수행합니다.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._config = settings.preprocessing

    def normalize(self, text: str) -> str:
        """텍스트 정규화 수행"""
        result = text

        # 1. 여러 줄바꿈을 2개로 통일
        if self._config.remove_extra_newlines:
            result = re.sub(r"\n{3,}", "\n\n", result)

        # 2. 여러 공백을 하나로
        if self._config.remove_extra_whitespace:
            result = re.sub(r"[ \t]+", " ", result)
            result = re.sub(r" +\n", "\n", result)  # 줄 끝 공백 제거

        # 3. 특수 문자 제거 (선택적)
        if self._config.remove_special_chars:
            # 한글, 영문, 숫자, 기본 구두점만 유지
            result = re.sub(r"[^가-힣a-zA-Z0-9\s.,!?\-:;()\[\]\"']", "", result)

        # 4. 짧은 줄 필터링 (선택적)
        if self._config.min_line_length > 0:
            lines = result.split("\n")
            lines = [
                line
                for line in lines
                if len(line.strip()) >= self._config.min_line_length
                or line.strip() == ""
            ]
            result = "\n".join(lines)

        return result.strip()

    def normalize_document(self, doc: RawDocument) -> RawDocument:
        """RawDocument의 내용을 정규화"""
        normalized_content = self.normalize(doc.content)
        normalized_pages = (
            [self.normalize(p) for p in doc.pages] if doc.pages else None
        )
        normalized_sheets = (
            {k: self.normalize(v) for k, v in doc.sheets.items()}
            if doc.sheets
            else None
        )

        return RawDocument(
            content=normalized_content,
            source=doc.source,
            file_type=doc.file_type,
            file_name=doc.file_name,
            metadata={**doc.metadata, "normalized": True},
            pages=normalized_pages,
            sheets=normalized_sheets,
        )
