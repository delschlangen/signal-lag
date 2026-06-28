"""Plain-language definitions + a 'read more' link for every tracked category.

Single source of truth, keyed by the same topic keys as ``config/taxonomy.yaml``.
Rendered in the dashboard's Methodology tab and mirrored in the README so a reader
who doesn't live in AI-safety jargon understands exactly what each category means.
"""
from __future__ import annotations

# key -> (one-line definition, read-more URL)
GLOSSARY: dict[str, tuple[str, str]] = {
    # ---- Safety topics ----
    "interpretability": (
        "Reverse-engineering the internal computations of a model — the features and "
        "circuits inside it — so we can understand *why* it does what it does, not just "
        "what it outputs.",
        "https://transformer-circuits.pub/",
    ),
    "cot_faithfulness": (
        "Whether a model's stated step-by-step reasoning actually reflects the "
        "computation that produced its answer, or is a plausible-sounding after-the-fact "
        "rationalization. Unfaithful reasoning makes 'explainable' traces misleading.",
        "https://arxiv.org/abs/2305.04388",
    ),
    "scalable_oversight": (
        "Techniques for supervising AI on tasks too hard or too large for a human to "
        "check directly — e.g. debate, recursive reward modeling, and weak-to-strong "
        "generalization — so oversight keeps working as models surpass us.",
        "https://arxiv.org/abs/2211.03540",
    ),
    "deceptive_alignment": (
        "When a model behaves as if aligned during training and evaluation but pursues "
        "different goals when it believes it is unobserved or deployed — including "
        "sandbagging, scheming, and hidden backdoored behaviors.",
        "https://arxiv.org/abs/2401.05566",
    ),
    "reward_hacking": (
        "When a system games its objective — exploiting flaws or proxies in the reward "
        "to score highly without achieving the intended goal (specification gaming, "
        "reward-model over-optimization, Goodhart's law).",
        "https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/",
    ),
    "evals_benchmarks": (
        "Structured tests that measure potentially dangerous capabilities (deception, "
        "persuasion, autonomy, cyber, bio) and safety properties of frontier models — "
        "the dangerous-capability evals and red-teaming that inform release decisions.",
        "https://arxiv.org/abs/2305.15324",
    ),
    "rsi_control": (
        "AI control: protocols to safely use and contain a powerful, possibly-untrusted "
        "model even if it is misaligned — plus the risks from recursive self-improvement "
        "and unsafe autonomous self-modification.",
        "https://arxiv.org/abs/2312.06942",
    ),
    "agentic_monitoring": (
        "Runtime oversight of autonomous, tool-using LLM agents — guardrails and "
        "enforcement that watch what an agent actually does and catch or block unsafe "
        "actions while it runs.",
        "https://arxiv.org/abs/2503.18666",
    ),
    # ---- Capability topics ----
    "agentic_capability": (
        "LLMs that autonomously plan and act over long horizons — browsing, using a "
        "computer, calling tools, completing multi-step tasks with little human input.",
        "https://www.anthropic.com/research/building-effective-agents",
    ),
    "rl_rlhf": (
        "Training models with reinforcement learning from human feedback and preference "
        "optimization (PPO, DPO) to make them more helpful and aligned with human "
        "preferences.",
        "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
    ),
    "reasoning_cot": (
        "Eliciting explicit step-by-step reasoning and spending more compute at "
        "inference time ('thinking') to improve performance on math, code, and other "
        "hard problems.",
        "https://arxiv.org/abs/2201.11903",
    ),
    "frontier_scaling": (
        "Pretraining ever-larger foundation models and the scaling laws that predict "
        "how capability grows with data, compute, and parameters — including "
        "capabilities that emerge only at scale.",
        "https://arxiv.org/abs/2001.08361",
    ),
    "multiagent_tooluse": (
        "Multiple LLMs collaborating, plus tool use / function calling and agent "
        "orchestration — models that act through external tools and coordinate in teams.",
        "https://arxiv.org/abs/2302.04761",
    ),
    "self_improvement": (
        "Models that refine themselves or generate their own training data and "
        "curricula — self-training and recursive self-improvement loops.",
        "https://en.wikipedia.org/wiki/Recursive_self-improvement",
    ),
}


# Display order, grouped by kind (mirrors config/taxonomy.yaml).
SAFETY_KEYS = [
    "interpretability", "cot_faithfulness", "scalable_oversight", "deceptive_alignment",
    "reward_hacking", "evals_benchmarks", "rsi_control", "agentic_monitoring",
]
CAPABILITY_KEYS = [
    "agentic_capability", "rl_rlhf", "reasoning_cot", "frontier_scaling",
    "multiagent_tooluse", "self_improvement",
]


def get(key: str) -> tuple[str, str] | None:
    return GLOSSARY.get(key)
