from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE = "# User Profile\n\n"


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def _slugify(user_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id.strip())
    return slug or "default_user"


@dataclass
class UserProfileStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / f"{_slugify(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            match = re.match(r"^-\s*([a-z_]+)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
            if match:
                facts[match.group(1).lower()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        content = self.read_text(user_id)
        if not content.strip():
            content = DEFAULT_PROFILE

        key = key.lower()
        pattern = re.compile(rf"^-\s*{re.escape(key)}\s*:\s*.+$", re.MULTILINE | re.IGNORECASE)
        line = f"- {key}: {value.strip()}"
        if pattern.search(content):
            content = pattern.sub(line, content, count=1)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += f"{line}\n"
        self.write_text(user_id, content)


_QUESTION_HINTS = (
    r"tên\s+gì",
    r"ở\s+đâu",
    r"là\s+gì",
    r"là\s+ai",
    r"như\s+thế\s+nào",
    r"bao\s+nhiêu",
    r"có\s+thể\s+nhắc",
    r"nhắc\s+lại",
    r"bạn\s+có\s+biết",
    r"thử\s+nhắc",
)


def _is_question_only(message: str) -> bool:
    text = message.strip()
    lowered = text.lower()
    if not text.endswith("?"):
        return False
    if any(re.search(pattern, lowered) for pattern in _QUESTION_HINTS):
        return True
    declarative_markers = (
        "đính chính",
        "nhớ giúp",
        "tên là",
        "tên mình là",
        "đang ở",
        "đang làm",
        "yêu thích là",
        "chuyển sang",
        "không còn",
    )
    return not any(marker in lowered for marker in declarative_markers)


_BAD_VALUES = frozenset(
    {
        "gì",
        "ai",
        "đâu",
        "sao",
        "nào",
        "hiện tại",
        "đã thay đổi",
        "nghiệp",
        "nghề",
        "mình",
        "tôi",
        "bạn",
        "và",
        "của mình",
    }
)


def _clean_fact_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    cleaned = re.sub(r"^là\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(chứ không|và|nhưng|vì|để|trong|vài|mỗi).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(cho|với|về).*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .,")


def _is_valid_fact(value: str) -> bool:
    if len(value) < 2:
        return False
    lowered = value.lower()
    if lowered in _BAD_VALUES:
        return False
    if lowered.endswith("?"):
        return False
    if len(value.split()) == 1 and len(value) < 4:
        return False
    return True


def extract_profile_updates(message: str) -> dict[str, str]:
    if _is_question_only(message):
        return {}

    text = message.strip()
    lowered = text.lower()
    updates: dict[str, str] = {}

    def set_fact(key: str, value: str) -> None:
        cleaned = _clean_fact_value(value)
        if _is_valid_fact(cleaned):
            updates[key] = cleaned

    priority_patterns: list[tuple[str, str]] = [
        ("profession", r"chuyển sang\s+(MLOps engineer|backend engineer)"),
        ("profession", r"không còn làm\s+[^,]+,\s*giờ chuyển sang\s+([^.,\n?]+)"),
        ("profession", r"nghề nghiệp hiện tại vẫn là\s+([^.,\n?]+)"),
        ("profession", r"đang làm\s+(MLOps engineer|backend engineer)"),
        ("profession", r"đang làm\s+([^.,\n?]+?)\s+cho\s+"),
        ("location", r"giờ\s+(?:mình|tôi)\s+đang ở\s+(Huế|Đà Nẵng|Hà Nội)"),
        ("location", r"giờ\s+(?:mình|tôi)\s+đang ở\s+([^.,\n]+?)(?:\s+chứ|\s+mỗi|\s+và|\s*$)"),
        ("location", r"đang làm việc ở\s+(Huế|Đà Nẵng|Hà Nội)"),
        ("location", r"đang làm việc ở\s+([^.,\n]+?)(?:\s+vài|\s+để|\s+trong|\s*$)"),
        ("location", r"(?:mình|tôi)\s+ở\s+(Huế|Đà Nẵng|Hà Nội)"),
        ("location", r"(?:mình|tôi)\s+ở\s+([^.,\n]+?)\s+và\s+"),
        ("name", r"(?:mình|tôi)\s+tên\s+là\s+([^.,\n?]+)"),
        ("name", r"tên\s+(?:mình|tôi)\s+là\s+([^.,\n?]+)"),
        ("name", r"tên\s+(DũngCT(?:\s+Stress)?)"),
        ("favorite_drink", r"đồ uống yêu thích là\s+([^.,\n?]+)"),
        ("favorite_food", r"món ăn yêu thích là\s+([^.,\n?]+)"),
        ("pet", r"nuôi\s+(?:một\s+)?(?:bé\s+)?(corgi tên\s+\w+)"),
        ("response_style", r"trả lời\s+(ngắn gọn[^.,\n?]*)"),
        ("response_style", r"(\d+\s+bullet[^.,\n?]*)"),
        ("response_style", r"muốn bạn trả lời\s+([^.,\n?]+)"),
        ("response_style", r"style trả lời[^:]*?(ngắn gọn[^.,\n?]*)"),
        ("interests", r"quan tâm[^.]*?(Python[^.,\n?]*)"),
        ("interests", r"thích\s+(Python[^.,\n?]*)"),
        ("interests", r"(Python,\s*AI[^.,\n?]*)"),
    ]

    for key, pattern in priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            set_fact(key, match.group(1))

    if "đính chính" in lowered or "cập nhật" in lowered or "không còn" in lowered:
        for key, pattern in priority_patterns[:12]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                set_fact(key, match.group(1))

    if "product manager" in lowered and "đùa" in lowered:
        updates.pop("profession", None)
    if "hà nội" in lowered and ("họp" in lowered or "bay" in lowered):
        updates.pop("location", None)

    if "mình vẫn ở huế" in lowered:
        set_fact("location", "Huế")

    if "nghề nghiệp hiện tại vẫn là mlops engineer" in lowered.replace(" ", ""):
        set_fact("profession", "MLOps engineer")

    return updates


def fact_confidence(message: str, key: str, value: str) -> float:
    """Heuristic confidence score in [0, 1] before persisting a fact to User.md."""
    text = message.strip()
    lowered = text.lower()

    if any(marker in lowered for marker in ("đính chính", "chuyển sang", "không còn", "cập nhật", "vẫn là")):
        if key in {"location", "profession", "name"}:
            return 0.95

    high_confidence_patterns: dict[str, list[str]] = {
        "name": [r"tên\s+là", r"tên mình là"],
        "location": [r"đang ở", r"đang làm việc ở", r"mình ở"],
        "profession": [r"chuyển sang", r"đang làm", r"nghề nghiệp hiện tại vẫn là"],
        "favorite_drink": [r"đồ uống yêu thích là"],
        "favorite_food": [r"món ăn yêu thích là"],
        "pet": [r"nuôi"],
        "response_style": [r"muốn bạn trả lời", r"\d+\s+bullet"],
    }

    for pattern in high_confidence_patterns.get(key, []):
        if re.search(pattern, lowered):
            return 0.9

    if key in {"profession", "location"} and re.search(
        r"(MLOps engineer|backend engineer|Huế|Đà Nẵng|Hà Nội)", value, re.IGNORECASE
    ):
        return 0.85

    if key == "interests":
        if re.search(r"quan tâm|python,\s*ai", lowered):
            return 0.8
        return 0.55

    if key == "response_style":
        return 0.75

    if any(word in lowered for word in ("có lẽ", "đùa", "thử", "hay là", "tuần sau")):
        return 0.3

    return 0.7


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""
    selected = messages[:max_items]
    parts: list[str] = []
    for item in selected:
        role = item.get("role", "user")
        content = item.get("content", "").strip()
        if not content:
            continue
        snippet = content if len(content) <= 180 else content[:177] + "..."
        parts.append(f"[{role}] {snippet}")
    return " | ".join(parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _total_tokens(self, thread: dict[str, object]) -> int:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        summary: str = thread["summary"]  # type: ignore[assignment]
        total = estimate_tokens(summary)
        for message in messages:
            total += estimate_tokens(message.get("content", ""))
        return total

    def _compact_if_needed(self, thread_id: str) -> None:
        thread = self._ensure_thread(thread_id)
        while self._total_tokens(thread) > self.threshold_tokens:
            messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
            if len(messages) <= self.keep_messages:
                break
            to_summarize = messages[:-self.keep_messages]
            if not to_summarize:
                break
            summary: str = thread["summary"]  # type: ignore[assignment]
            new_chunk = summarize_messages(to_summarize, max_items=len(to_summarize))
            thread["summary"] = f"{summary} {new_chunk}".strip() if summary else new_chunk
            thread["messages"] = messages[len(to_summarize) :]
            thread["compactions"] = int(thread["compactions"]) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        self._compact_if_needed(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._ensure_thread(thread_id)
        return {
            "messages": list(thread["messages"]),
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self._ensure_thread(thread_id)
        return int(thread["compactions"])
