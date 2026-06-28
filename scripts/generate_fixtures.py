"""Generate a synthetic but realistic fixture dataset for offline runs/tests.

Produces signal_lag/fixtures/sample_papers.json: papers spanning every taxonomy
topic across a 3-year quarterly grid, with *deliberate* divergence patterns so
the gap layer has something to find:

  - agentic_capability   : strongly accelerating
  - agentic_monitoring   : flat            -> headline divergence
  - reasoning_cot        : accelerating
  - cot_faithfulness     : modest growth
  - rl_rlhf              : steady
  - reward_hacking       : growing (safety keeping pace)
  - others               : mild trends

Citation counts skew toward older + capability papers, with a couple of planted
"sleeper" papers whose citations arrive late (early-heat signal).

Deterministic (seeded) so fixtures are stable across regenerations.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "signal_lag" / "fixtures" / "sample_papers.json"

# Topic -> (abstract building blocks, per-quarter base count, growth slope, category)
TOPICS = {
    "agentic_capability": dict(
        category="cs.AI", base=3, slope=2.2,
        phrases=[
            "We present an autonomous LLM agent that plans and executes long-horizon tasks",
            "Our framework lets tool-using agents browse the web and operate a computer",
            "The agent decomposes goals into subtasks and acts over many steps",
            "We scale agentic pipelines for software engineering and research workflows",
        ],
    ),
    "agentic_monitoring": dict(
        category="cs.AI", base=2, slope=0.1,  # flat -> divergence
        phrases=[
            "We propose runtime monitoring of autonomous agent behavior",
            "A guardrail system detects unsafe actions taken by tool-using agents",
            "We oversee agent trajectories to catch policy violations at runtime",
        ],
    ),
    "reasoning_cot": dict(
        category="cs.CL", base=4, slope=1.8,
        phrases=[
            "Chain-of-thought prompting improves multi-step reasoning in language models",
            "We use test-time compute and deliberate reasoning for math and code",
            "Step-by-step reasoning boosts performance on hard benchmarks",
        ],
    ),
    "cot_faithfulness": dict(
        category="cs.CL", base=1, slope=0.6,
        phrases=[
            "We measure the faithfulness of chain-of-thought reasoning traces",
            "Models often produce post-hoc rationalizations unfaithful to their computation",
            "We test whether stated reasoning reflects the model's actual process",
        ],
    ),
    "rl_rlhf": dict(
        category="cs.LG", base=4, slope=0.4,
        phrases=[
            "Reinforcement learning from human feedback aligns language models",
            "We compare PPO and DPO for preference optimization of LLMs",
            "Policy optimization fine-tunes models from human preference data",
        ],
    ),
    "reward_hacking": dict(
        category="cs.LG", base=1, slope=0.9,
        phrases=[
            "We study reward hacking and specification gaming in RLHF",
            "Reward model overoptimization induces Goodhart effects",
            "Models exploit flaws in the reward function to gain reward",
        ],
    ),
    "frontier_scaling": dict(
        category="cs.LG", base=3, slope=0.8,
        phrases=[
            "We derive scaling laws for large language models",
            "Pretraining larger frontier models yields emergent capabilities",
            "Compute-optimal scaling of foundation models",
        ],
    ),
    "evals_benchmarks": dict(
        category="cs.AI", base=2, slope=0.7,
        phrases=[
            "We build dangerous-capability evaluations for frontier models",
            "A red-teaming benchmark measures deception and persuasion risk",
            "Safety evaluations probe autonomy and self-replication capability",
        ],
    ),
    "multiagent_tooluse": dict(
        category="cs.AI", base=2, slope=1.1,
        phrases=[
            "Multi-agent collaboration between language models solves complex tasks",
            "We orchestrate role-based agent teams with tool use and function calling",
            "Agents coordinate via message passing to complete shared goals",
        ],
    ),
    "scalable_oversight": dict(
        category="cs.LG", base=1, slope=0.3,
        phrases=[
            "We study scalable oversight via debate and recursive reward modeling",
            "Weak-to-strong generalization supervises stronger models with weaker ones",
            "AI-assisted evaluation helps judge hard-to-grade outputs",
        ],
    ),
    "self_improvement": dict(
        category="cs.LG", base=1, slope=1.0,
        phrases=[
            "Self-improving models refine themselves and generate their own training data",
            "Recursive self-improvement via self-training and self-generated curricula",
            "Models bootstrap capabilities by critiquing and revising their outputs",
        ],
    ),
    "deceptive_alignment": dict(
        category="cs.AI", base=1, slope=0.2,
        phrases=[
            "We investigate deceptive alignment and scheming in language models",
            "Sleeper-agent backdoors make models behave differently under observation",
            "Sandbagging: models strategically underperform on evaluations",
        ],
    ),
    "interpretability": dict(
        category="cs.LG", base=2, slope=0.9,
        phrases=[
            "Mechanistic interpretability reverse-engineers circuits in transformers",
            "Sparse autoencoders extract interpretable features from model internals",
            "We probe internal representations of language models",
        ],
    ),
    "rsi_control": dict(
        category="cs.AI", base=1, slope=0.3,
        phrases=[
            "AI control protocols safely deploy untrusted models",
            "Containment strategies limit recursively self-improving systems",
            "We prevent unsafe autonomous self-modification and escape",
        ],
    ),
}

INSTITUTIONS = [
    "DeepMind", "OpenAI", "Anthropic", "Stanford University", "MIT",
    "UC Berkeley", "Google Research", "Meta AI", "Tsinghua University",
    "University of Oxford", "ETH Zurich", "Microsoft Research",
]
FIRST = ["Wei", "Maria", "Chen", "Aditya", "Sarah", "Yuki", "Omar", "Elena", "Raj", "Lena"]
LAST = ["Zhang", "Garcia", "Smith", "Patel", "Kim", "Tanaka", "Khan", "Novak", "Singh", "Muller"]


def quarters(start: dt.date, n: int):
    y, m = start.year, start.month
    for _ in range(n):
        yield dt.date(y, m, 1)
        m += 3
        if m > 12:
            m -= 12
            y += 1


def make_authors(rng: random.Random):
    k = rng.randint(2, 5)
    insts = rng.sample(INSTITUTIONS, rng.randint(1, 2))
    return [
        {"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "affiliation": rng.choice(insts)}
        for _ in range(k)
    ], insts


def main():
    rng = random.Random(7)
    start = dt.date(2022, 7, 1)
    qs = list(quarters(start, 12))  # 3 years of quarters
    papers = []
    pid = 1000
    today_year = 2025

    for qi, qdate in enumerate(qs):
        for topic, spec in TOPICS.items():
            count = max(0, round(spec["base"] + spec["slope"] * qi + rng.uniform(-1, 1)))
            for _ in range(count):
                pid += 1
                day = rng.randint(1, 80)
                pub = qdate + dt.timedelta(days=day)
                if pub.year > today_year:
                    pub = dt.date(today_year, 12, 1)
                phrase = rng.choice(spec["phrases"])
                extra = rng.choice([
                    "Experiments on standard benchmarks show consistent gains.",
                    "We release code and data to support reproducibility.",
                    "Our analysis reveals trade-offs prior work overlooked.",
                    "Results generalize across model scales and domains.",
                ])
                authors, insts = make_authors(rng)
                age_q = len(qs) - qi  # older papers accrue more citations
                base_cit = int(max(0, rng.gauss(age_q * 3, age_q)))
                if spec["category"] != "cs.CL":
                    base_cit = int(base_cit * 1.2)
                # counts_by_year skewed to publication year onward
                cby = []
                for yr in range(pub.year, today_year + 1):
                    share = rng.uniform(0.2, 0.6)
                    cby.append({"year": yr, "count": int(base_cit * share)})
                papers.append({
                    "arxiv_id": f"23{pid:05d}.{rng.randint(10000,99999)}",
                    "title": phrase.split(",")[0][:90],
                    "abstract": f"{phrase}. {extra} (topic:{topic})",
                    "published": pub.isoformat(),
                    "updated": pub.isoformat(),
                    "categories": [spec["category"]],
                    "primary_category": spec["category"],
                    "authors": authors,
                    "cited_by_count": base_cit,
                    "counts_by_year": cby,
                    "institutions": insts,
                    "_topic": topic,
                })

    # Plant two "sleeper" papers: low early citations, big recent spike.
    for topic in ("interpretability", "agentic_capability"):
        pid += 1
        pub = dt.date(2023, 2, 10)
        papers.append({
            "arxiv_id": f"23{pid:05d}.{rng.randint(10000,99999)}",
            "title": f"A sleeper result in {topic}",
            "abstract": f"{TOPICS[topic]['phrases'][0]}. This approach was initially overlooked. (topic:{topic})",
            "published": pub.isoformat(),
            "updated": pub.isoformat(),
            "categories": [TOPICS[topic]["category"]],
            "primary_category": TOPICS[topic]["category"],
            "authors": make_authors(rng)[0],
            "cited_by_count": 60,
            "counts_by_year": [
                {"year": 2023, "count": 2},
                {"year": 2024, "count": 8},
                {"year": 2025, "count": 50},
            ],
            "institutions": ["Anthropic"],
            "_topic": topic,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(papers, fh, indent=1)
    print(f"Wrote {len(papers)} papers -> {OUT}")


if __name__ == "__main__":
    main()
