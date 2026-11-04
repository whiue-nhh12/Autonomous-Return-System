"""Profile Python functions used by the end-to-end query workflow.

The embedding model is loaded before profiling starts, so model loading is not
included in the reported query timings.

Examples:
    python -m scripts.measure "Điều kiện hoàn tiền khi sản phẩm bị lỗi là gì?"
    python -m scripts.measure --limit 8 --top 100
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from threading import Lock, local, setprofile
from time import perf_counter
from types import CodeType, FrameType
from typing import Any

from rag.embedding.embedding import Sentence_transformer
from rag.retrieval.config import get_retrieval_settings
from rag.retrieval.query_encoder import SentenceTransformerQueryEncoder
from rag.retrieval.reranker import CrossEncoderReranker
from scripts.query import run_query

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_QUERY = (
    "Tôi mua tai nghe bluetooth bị lỗi do nhà sản xuất ở ngày thứ 12 sau khi nhận hàng. "
    "Theo quy định, tôi sẽ được xử lý đổi trả thế nào?"
)


@dataclass
class FunctionTiming:
    calls: int = 0
    self_seconds: float = 0.0
    cumulative_seconds: float = 0.0


class FunctionProfiler:
    """Collect inclusive and exclusive Python function time across threads."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.timings: dict[CodeType, FunctionTiming] = defaultdict(FunctionTiming)
        self._thread_state = local()
        self._lock = Lock()
        self._project_code_cache: dict[CodeType, bool] = {}

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        del arg
        if event == "call":
            if not self._is_project_code(frame.f_code):
                return
            stack = getattr(self._thread_state, "stack", None)
            if stack is None:
                stack = []
                self._thread_state.stack = stack
            stack.append((frame.f_code, perf_counter(), 0.0))
            return

        if event != "return":
            return

        stack = getattr(self._thread_state, "stack", None)
        if not stack or stack[-1][0] is not frame.f_code:
            return

        code, started_at, child_seconds = stack.pop()
        elapsed = perf_counter() - started_at
        self_seconds = max(0.0, elapsed - child_seconds)
        if stack:
            parent_code, parent_started, parent_child_seconds = stack[-1]
            stack[-1] = (parent_code, parent_started, parent_child_seconds + elapsed)

        with self._lock:
            timing = self.timings[code]
            timing.calls += 1
            timing.self_seconds += self_seconds
            timing.cumulative_seconds += elapsed

    def _is_project_code(self, code: CodeType) -> bool:
        cached = self._project_code_cache.get(code)
        if cached is not None:
            return cached

        try:
            path = Path(code.co_filename).resolve()
            relative_path = path.relative_to(self.project_root)
        except (OSError, ValueError):
            is_project_code = False
        else:
            is_project_code = (
                path.suffix == ".py"
                and relative_path.as_posix() != "scripts/measure.py"
            )
        with self._lock:
            return self._project_code_cache.setdefault(code, is_project_code)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:,.2f}"


def _print_stage_timings(timings_ms: dict[str, float]) -> None:
    labels = {
        "initialization": "Khởi tạo pipeline và kết nối",
        "query_preparation": "Chuẩn bị query",
        "retrieval": "Truy xuất / tìm kiếm",
        "context_building": "Dựng context",
        "answer_generation": "Sinh câu trả lời",
        "total": "Tổng input -> output",
    }
    print("\n=== CÔNG ĐOẠN TỔNG HỢP ===")
    for name, label in labels.items():
        if name in timings_ms:
            print(f"{label:<34} {timings_ms[name]:>12,.2f} ms")


def _print_retrieval_timings(timings_ms: dict[str, float]) -> None:
    labels = {
        "retrieval.total": "Toàn pha retrieval",
        "query.prepare": "Chuẩn bị query (tổng)",
        "query.classification": "  Phân loại query",
        "query.tokenization": "  Tách token BM25",
        "manifest.load_published_versions": "Đọc phiên bản publish",
        "query_rewrite": "Rewrite query (nếu bật)",
        "embedding.query": "Embedding query",
        "search.total": "Text search (tổng)",
        "search.text_pipeline": "Pipeline text search",
        "search.hybrid_wall": "Hybrid search (wall time)",
        "search.keyword": "  Elasticsearch keyword search",
        "search.vector": "  Qdrant vector search",
        "search.hybrid_fusion": "  RRF fusion keyword/vector",
        "search.query_fusion": "Fusion nhiều query",
        "hyde.generate_document": "HyDE tạo tài liệu giả định",
        "embedding.hyde": "HyDE embedding",
        "hyde.vector_search": "HyDE vector search",
        "reranker": "Reranker",
        "graph.query_extraction": "Graph query extraction (LLM)",
        "graph.neo4j_query": "Neo4j query",
    }
    rows = sorted(timings_ms.items(), key=lambda item: item[1], reverse=True)

    print("\n=== CHI TIẾT PHA RETRIEVAL (xếp theo thời gian giảm dần) ===")
    print(f"{'Thời gian (ms)':>16}  Công đoạn")
    for name, elapsed_ms in rows:
        print(f"{elapsed_ms:>16,.2f}  {labels.get(name, name)}")
    print(
        "Lưu ý: các dòng tổng bao gồm một số bước con; Elasticsearch và Qdrant "
        "chạy song song nên thời gian backend chồng lấp. Không cộng các dòng để suy ra wall time."
    )


def _print_module_timings(profile: FunctionProfiler) -> None:
    module_self_seconds: dict[str, float] = defaultdict(float)
    for code, timing in profile.timings.items():
        path = Path(code.co_filename).resolve()
        try:
            module = path.relative_to(_PROJECT_ROOT).with_suffix("").as_posix().replace("/", ".")
        except ValueError:
            continue
        module_self_seconds[module] += timing.self_seconds

    print("\n=== TỔNG HỢP THEO MODULE (self time, loại các hàm Python con) ===")
    print(f"{'Module':<58} {'Tổng self time (ms)':>21}")
    for module, seconds in sorted(module_self_seconds.items(), key=lambda item: item[1], reverse=True):
        print(f"{module:<58} {_format_ms(seconds):>21}")


def _print_function_timings(profile: FunctionProfiler, top: int) -> None:
    entries = sorted(
        profile.timings.items(),
        key=lambda item: item[1].cumulative_seconds,
        reverse=True,
    )
    if top:
        entries = entries[:top]

    print("\n=== CHI TIẾT FUNCTION (sắp theo cumulative time) ===")
    print(f"{'Calls':>8} {'Self (ms)':>14} {'Cumulative (ms)':>18}  Function")
    print(
        "Self loại thời gian ở hàm con trong project; cumulative gồm cả hàm con. "
        "Thời gian function cộng từ các thread song song có thể lớn hơn wall-clock."
    )
    for code, timing in entries:
        try:
            filename = Path(code.co_filename).resolve().relative_to(_PROJECT_ROOT).as_posix()
        except ValueError:
            filename = code.co_filename
        location = f"{filename}:{code.co_firstlineno} - {code.co_name}"
        print(
            f"{timing.calls:>8} "
            f"{_format_ms(timing.self_seconds):>14} "
            f"{_format_ms(timing.cumulative_seconds):>18}  "
            f"{location}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure query processing time from input to output, excluding model loading."
    )
    parser.add_argument("query", nargs="?", help="Question to send to the retrieval system")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results per retrieval source")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Maximum number of Python functions to show (0 shows all)",
    )
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")

    args = build_parser().parse_args()
    if args.limit <= 0:
        print("--limit must be greater than zero", file=sys.stderr)
        return 2
    if args.top < 0:
        print("--top cannot be negative", file=sys.stderr)
        return 2

    query_text = args.query if args.query is not None else _DEFAULT_QUERY

    try:
        print("Đang nạp embedding model (không tính vào thời gian đo)...", file=sys.stderr)
        model = Sentence_transformer().load_model()
        query_encoder = SentenceTransformerQueryEncoder(model=model)
        retrieval_settings = get_retrieval_settings()
        reranker = None
        if retrieval_settings.enable_reranker:
            reranker = CrossEncoderReranker(
                model_name=retrieval_settings.reranker_model,
                batch_size=retrieval_settings.reranker_batch_size,
                max_length=retrieval_settings.reranker_max_length,
            )
            reranker._load()
    except Exception as exc:
        print(f"Model initialization failed (not measured): {exc}", file=sys.stderr)
        return 1

    profiler = FunctionProfiler(_PROJECT_ROOT)
    sys.setprofile(profiler)
    setprofile(profiler)
    try:
        result, _ = run_query(
            query_text,
            limit=args.limit,
            query_encoder=query_encoder,
            reranker=reranker,
        )
    except Exception as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sys.setprofile(None)
        setprofile(None)

    print(f"\nQuery: {result['input']['query']}")
    _print_function_timings(profiler, args.top)
    _print_module_timings(profiler)
    _print_retrieval_timings(result["retrieval_timings_ms"])
    _print_stage_timings(result["timings_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
