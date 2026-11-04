"""Vietnamese word segmentation for both indexing and query text."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable


DEFAULT_VIETNAMESE_STOP_WORDS = frozenset(
    ""
    "bị bởi cả các cái càng chỉ chiếc cho chứ chưa có cũng đã đang đây đó "
    "để đến điều do được đúng gì khi không là lại làm mà mỗi một những "
    "nếu ngoài ra rất sau sẽ so sự tại theo thì trong từ với và vì vẫn về "
    "vào như nên hoặc hơn nhất này kia nó chúng tôi bạn người của ở"
    .split()
)


class VietnameseTokenizer:
    def __init__(
        self,
        tokenize: Callable[[str], str] | None = None,
        *,
        stop_words: Iterable[str] | None = None,
        remove_stop_words: bool = True,
    ) -> None:
        self._tokenize = tokenize
        configured_stop_words = DEFAULT_VIETNAMESE_STOP_WORDS if stop_words is None else stop_words
        self.stop_words = frozenset(word.casefold() for word in configured_stop_words)
        self.remove_stop_words = remove_stop_words

    def tokenize(self, text: str) -> str:
        if not text.strip():
            return ""
        if self._tokenize is None:
            try:
                from pyvi import ViTokenizer
            except ImportError as exc:
                raise ImportError("Vietnamese search requires PyVi. Install it with: pip install pyvi") from exc
            self._tokenize = ViTokenizer.tokenize
        tokenized = self._tokenize(text).lower()
        if not self.remove_stop_words:
            return tokenized
        tokens = re.split(r"\s+", tokenized.strip())
        filtered = [
            token for token in tokens
            if token and token.casefold() not in self.stop_words
        ]
        # Keep short all-stop-word queries searchable instead of producing an empty BM25 query.
        return " ".join(filtered or tokens)
