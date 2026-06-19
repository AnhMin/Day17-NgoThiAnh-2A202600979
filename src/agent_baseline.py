from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def _session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def total_token_usage(self) -> int:
        return sum(session.token_usage for session in self.sessions.values())

    def total_prompt_token_usage(self) -> int:
        return sum(session.prompt_tokens_processed for session in self.sessions.values())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        context_text = "\n".join(item["content"] for item in session.messages)
        session.prompt_tokens_processed += estimate_tokens(context_text)

        answer = self._offline_response(session.messages, message)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)

        return {
            "content": answer,
            "agent_tokens": estimate_tokens(answer),
            "prompt_tokens": estimate_tokens(context_text),
        }

    def _offline_response(self, messages: list[dict[str, str]], message: str) -> str:
        lowered = message.lower()
        if "?" in message:
            for item in reversed(messages):
                if item["role"] != "user":
                    continue
                content = item["content"]
                if any(keyword in lowered for keyword in ("tên", "ở đâu", "nghề", "style", "đồ uống", "món ăn")):
                    return (
                        "Trong thread hiện tại mình chỉ có thể dựa vào ngữ cảnh gần nhất. "
                        f"Thông tin gần nhất bạn vừa nói: {content[:120]}"
                    )
        return f"Đã ghi nhận trong session này: {message[:160]}"

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        context = "\n".join(f"{item['role']}: {item['content']}" for item in session.messages)
        session.prompt_tokens_processed += estimate_tokens(context)
        response = self.langchain_agent.invoke(context + f"\nuser: {message}")
        answer = getattr(response, "content", str(response))
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)
        return {
            "content": answer,
            "agent_tokens": estimate_tokens(answer),
            "prompt_tokens": estimate_tokens(context),
        }

    def _maybe_build_langchain_agent(self):
        try:
            if not self.config.model.api_key and self.config.model.provider not in {"ollama"}:
                return None
            return build_chat_model(self.config.model)
        except Exception:
            return None
