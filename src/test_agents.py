from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    root = tmp_path
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=40,
        compact_keep_messages=2,
        profile_confidence_threshold=0.6,
        model=model,
        judge_model=model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test-user"

    path = store.write_text(user_id, "# User Profile\n\n- name: Alice\n")
    assert path.exists()
    assert "Alice" in store.read_text(user_id)

    store.upsert_fact(user_id, "location", "Huế")
    facts = store.facts(user_id)
    assert facts["location"] == "Huế"

    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed is True
    assert "Bob" in store.read_text(user_id)
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=30, keep_messages=2)
    thread_id = "thread-compact"

    for index in range(12):
        manager.append(thread_id, "user", f"Message dài số {index} " + ("x" * 40))

    assert manager.compaction_count(thread_id) >= 1
    ctx = manager.context(thread_id)
    messages = ctx["messages"]
    assert len(messages) <= 2 + 12


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    advanced.reply("u1", "thread-a", "Chào bạn, mình tên là DũngCT.")
    advanced.reply("u1", "thread-b", "Mình tên gì?")

    baseline.reply("u1", "thread-a", "Chào bạn, mình tên là DũngCT.")
    baseline_result = baseline.reply("u1", "thread-b", "Mình tên gì?")

    advanced_result = advanced.reply("u1", "thread-c", "Mình tên gì?")
    assert "DũngCT" in advanced_result["content"]
    assert "DũngCT" not in baseline_result["content"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    thread_id = "long-thread"
    for index in range(20):
        message = f"Lượt {index}: mình đang benchmark compact memory với nội dung dài " + ("y" * 60)
        baseline.reply("u1", thread_id, message)
        advanced.reply("u1", thread_id, message)

    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)
    assert advanced.compaction_count(thread_id) >= 1
    assert advanced_prompt < baseline_prompt


def test_confidence_threshold_skips_weak_facts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.profile_confidence_threshold = 0.85
    agent = AdvancedAgent(config=config, force_offline=True)

    agent.reply("u1", "t1", "Mình hay thích Python buổi sáng.")
    facts = agent.profile_store.facts("u1")
    assert "interests" not in facts

    agent.reply("u1", "t2", "Chào bạn, mình tên là Alice.")
    assert agent.profile_store.facts("u1").get("name") == "Alice"
