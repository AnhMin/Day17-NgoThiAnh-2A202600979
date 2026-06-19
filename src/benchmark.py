from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for item in expected if item.lower() in lowered)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    length_score = 1.0 if 20 <= len(answer) <= 500 else 0.5
    structure_score = 1.0 if any(marker in answer for marker in (".", ":", "-", "•")) else 0.7
    return round(min(1.0, (recall * 0.7) + (length_score * 0.15) + (structure_score * 0.15)), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_sizes: list[int] = []
    user_ids = {conv["user_id"] for conv in conversations}

    initial_memory = 0
    if hasattr(agent, "memory_file_size"):
        for user_id in user_ids:
            initial_memory += agent.memory_file_size(user_id)

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]

        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)

        for index, question in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            result = agent.reply(user_id, recall_thread, question["question"])
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(result["content"], expected))
            quality_scores.append(heuristic_quality(result["content"], expected))

        if hasattr(agent, "memory_file_size"):
            memory_sizes.append(agent.memory_file_size(user_id))

    agent_tokens = agent.total_token_usage() if hasattr(agent, "total_token_usage") else 0
    prompt_tokens = agent.total_prompt_token_usage() if hasattr(agent, "total_prompt_token_usage") else 0

    if hasattr(agent, "total_compactions"):
        compactions = agent.total_compactions()
    else:
        compactions = sum(agent.compaction_count(conv["id"]) for conv in conversations)

    final_memory = max(memory_sizes) if memory_sizes else 0
    memory_growth = max(0, final_memory - initial_memory)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            row.recall_score,
            row.response_quality,
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    return tabulate(table, headers=headers, tablefmt="github")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    standard_conversations = load_conversations(standard_path)
    stress_conversations = load_conversations(stress_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    print("## Standard Benchmark\n")
    standard_rows = [
        run_agent_benchmark("Baseline", baseline, standard_conversations, config),
        run_agent_benchmark("Advanced", advanced, standard_conversations, config),
    ]
    print(format_rows(standard_rows))

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    print("\n## Long-Context Stress Benchmark\n")
    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_conversations, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_conversations, config),
    ]
    print(format_rows(stress_rows))


if __name__ == "__main__":
    main()
