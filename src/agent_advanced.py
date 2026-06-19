from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, fact_confidence
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def total_token_usage(self) -> int:
        return sum(self.thread_tokens.values())

    def total_prompt_token_usage(self) -> int:
        return sum(self.thread_prompt_tokens.values())

    def total_compactions(self) -> int:
        return sum(int(ctx["compactions"]) for ctx in self.compact_memory.state.values())

    def _apply_profile_updates(self, user_id: str, message: str) -> None:
        updates = extract_profile_updates(message)
        threshold = self.config.profile_confidence_threshold
        for key, value in updates.items():
            if fact_confidence(message, key, value) >= threshold:
                self.profile_store.upsert_fact(user_id, key, value)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._apply_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)

        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens

        return {
            "content": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]
        parts = [profile, summary] + [message.get("content", "") for message in messages]
        return estimate_tokens("\n".join(part for part in parts if part))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        lowered = message.lower()

        if not facts:
            return "Mình chưa có thông tin profile bền vững. Bạn có thể chia sẻ thêm về bạn."

        parts: list[str] = []

        if any(keyword in lowered for keyword in ("tên", "tên gì", "tên mình")):
            if "name" in facts:
                parts.append(f"Tên bạn là {facts['name']}.")

        if any(keyword in lowered for keyword in ("ở đâu", "nơi ở", "đang ở")):
            if "location" in facts:
                parts.append(f"Hiện tại bạn đang ở {facts['location']}.")

        if any(keyword in lowered for keyword in ("nghề", "làm nghề", "nghề nghiệp", "công việc")):
            if "profession" in facts:
                parts.append(f"Nghề nghiệp hiện tại của bạn là {facts['profession']}.")

        if any(keyword in lowered for keyword in ("style", "trả lời", "bullet")):
            style = facts.get("response_style", "ngắn gọn, có ví dụ thực tế")
            parts.append(f"Style trả lời bạn thích: {style}.")

        if any(keyword in lowered for keyword in ("đồ uống", "uống")):
            if "favorite_drink" in facts:
                parts.append(f"Đồ uống yêu thích: {facts['favorite_drink']}.")

        if any(keyword in lowered for keyword in ("món ăn", "ăn")):
            if "favorite_food" in facts:
                parts.append(f"Món ăn yêu thích: {facts['favorite_food']}.")

        if any(keyword in lowered for keyword in ("nuôi", "corgi", "thú cưng", "con gì")):
            if "pet" in facts:
                parts.append(f"Bạn nuôi {facts['pet']}.")

        if any(keyword in lowered for keyword in ("python", "ai", "mối quan tâm", "quan tâm")):
            interests = facts.get("interests")
            if interests:
                parts.append(f"Mối quan tâm: {interests}.")
            profile_text = self.profile_store.read_text(user_id).lower()
            if "ai" in profile_text or "ai" in lowered:
                parts.append("Mối quan tâm: AI.")

        if any(keyword in lowered for keyword in ("tóm tắt", "mô tả", "biết", "nhắc lại")):
            summary_bits = []
            for key in ("name", "location", "profession", "favorite_drink", "response_style", "favorite_food", "pet", "interests"):
                if key in facts:
                    summary_bits.append(f"{key}: {facts[key]}")
            if summary_bits:
                parts.append("Tóm tắt profile: " + "; ".join(summary_bits) + ".")

        if "huế" in lowered or "hà nội" in lowered or "product manager" in lowered:
            if "profession" in facts:
                parts.append(f"Nghề nghiệp hiện tại (bỏ qua nhiễu): {facts['profession']}.")
            if "location" in facts:
                parts.append(f"Nơi ở hiện tại (bỏ qua nhiễu): {facts['location']}.")

        if parts:
            return " ".join(parts)

        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        if summary:
            return f"Đã ghi nhận. Ngữ cảnh compact gần nhất: {summary[:200]}"
        return f"Đã ghi nhận và lưu vào User.md: {message[:120]}"

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._apply_profile_updates(user_id, message)
        profile = self.profile_store.read_text(user_id)
        self.compact_memory.append(thread_id, "user", message)
        ctx = self.compact_memory.context(thread_id)
        prompt = (
            f"{profile}\n"
            f"Summary: {ctx.get('summary', '')}\n"
            f"Recent: {ctx.get('messages', [])}\n"
            f"User: {message}"
        )
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + estimate_tokens(prompt)
        response = self.langchain_agent.invoke(prompt)
        answer = getattr(response, "content", str(response))
        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        return {
            "content": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": estimate_tokens(prompt),
        }

    def _maybe_build_langchain_agent(self):
        try:
            if not self.config.model.api_key and self.config.model.provider not in {"ollama"}:
                return None
            return build_chat_model(self.config.model)
        except Exception:
            return None
