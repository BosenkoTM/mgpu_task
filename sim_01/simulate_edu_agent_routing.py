#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симуляция токен-оптимальной маршрутизации в вузовской многоагентной системе.

Сценарий
--------
1. Преподаватель формулирует запрос.
2. Координатор выполняет локальный поиск в RAG-базе.
3. Если RAG-ответ недостаточен, координатор обращается через облачную LLM API
   к специализированным субагентам.
4. Сравниваются две политики:
   - STATIC_ALL: после неуспешного RAG вызываются все субагенты;
   - DYNAMIC_ACG: вызывается только релевантное подмножество, применяется
     последовательный выбор и ранняя остановка.
5. Основная цель — минимизировать облачные input+output tokens при ограничении
   на качество ответа преподавателю.

Зависимости
-----------
    pip install numpy pandas matplotlib

Запуск
------
    python simulate_edu_agent_routing.py

Результаты
----------
    outputs/comparison_by_context.csv
    outputs/per_query_results.csv
    outputs/agent_usage.csv
    outputs/simulation_parameters.csv
    outputs/01_tokens_by_context.png
    outputs/02_quality_by_context.png
    outputs/03_agents_by_context.png
    outputs/04_quality_cost_scatter.png
    outputs/05_route_distribution.png
    outputs/simulation_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Strategy = Literal["STATIC_ALL", "DYNAMIC_ACG"]


DOMAINS: Tuple[str, ...] = (
    "нормативная_база",
    "педагогика",
    "оценивание",
    "цифровые_инструменты",
    "проектирование_курса",
    "администрирование",
)


@dataclass(frozen=True)
class AgentProfile:
    name: str
    primary_domain: str
    expertise: Tuple[float, ...]
    reliability: float
    system_prompt_tokens: int


AGENTS: Tuple[AgentProfile, ...] = (
    AgentProfile(
        "Нормативный эксперт",
        "нормативная_база",
        (0.96, 0.22, 0.36, 0.14, 0.24, 0.58),
        0.94,
        250,
    ),
    AgentProfile(
        "Педагогический эксперт",
        "педагогика",
        (0.18, 0.96, 0.58, 0.22, 0.72, 0.20),
        0.93,
        240,
    ),
    AgentProfile(
        "Эксперт по оцениванию",
        "оценивание",
        (0.22, 0.64, 0.96, 0.22, 0.54, 0.18),
        0.92,
        235,
    ),
    AgentProfile(
        "Эксперт по цифровым инструментам",
        "цифровые_инструменты",
        (0.12, 0.34, 0.30, 0.97, 0.54, 0.24),
        0.91,
        245,
    ),
    AgentProfile(
        "Методист-проектировщик",
        "проектирование_курса",
        (0.18, 0.76, 0.52, 0.48, 0.97, 0.22),
        0.94,
        260,
    ),
    AgentProfile(
        "Административный эксперт",
        "администрирование",
        (0.60, 0.20, 0.18, 0.20, 0.26, 0.96),
        0.92,
        230,
    ),
)


# =============================================================================
# ### [CHANGE HERE: PARAMETERS TO PLAY WITH] ###
# Эти параметры следует заменить значениями, оцененными по реальным API-трассам.
# =============================================================================
@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int = 73
    train_requests: int = 1200
    test_requests: int = 1800

    # Экспериментальные лимиты контекста преподавательского сценария.
    # Это НЕ заявленные технические пределы конкретного облачного провайдера.
    teacher_context_windows: Tuple[int, ...] = (4096, 8192, 16384, 32768)

    # Распределение сложности запросов.
    p_simple: float = 0.44
    p_medium: float = 0.36
    p_complex: float = 0.20

    # Порог качества ответа преподавателю.
    quality_target: float = 0.82
    factuality_min: float = 0.84
    coverage_min: float = 0.78

    # RAG-gate: при достаточной оценке локального покрытия облачные субагенты
    # не вызываются, но финальный ответ все равно синтезируется через API.
    rag_direct_threshold: float = 0.835

    # Динамическая политика.
    max_dynamic_agents: int = 5
    marginal_quality_per_1k_tokens: float = 0.010
    dynamic_predicted_quality_target: float = 0.900
    routing_noise_std: float = 0.085
    early_stop_min_agents: int = 1

    # Токены служебных API-вызовов.
    coordinator_system_tokens: int = 320
    router_system_tokens: int = 280
    router_output_tokens: int = 150
    decision_input_tokens: int = 260
    decision_output_tokens: int = 90
    synthesis_system_tokens: int = 360
    direct_rag_system_tokens: int = 300

    # Ограничения на передаваемый RAG-контекст.
    rag_agent_excerpt_max: int = 950
    rag_synthesis_excerpt_max: int = 3600

    # Требуемый размер итогового ответа по сложности.
    output_tokens_simple: int = 520
    output_tokens_medium: int = 920
    output_tokens_complex: int = 1450

    # Синтетическая цена API только для дополнительной оценки.
    # Основной критерий эксперимента — токены.
    price_input_per_million_usd: float = 0.14
    price_output_per_million_usd: float = 0.28

    # Прочее.
    quality_noise_std: float = 0.012
    max_cloud_tokens_per_query: int = 60000
# =============================================================================
# ### [END CHANGE HERE] ###
# =============================================================================


def validate_config(cfg: SimulationConfig) -> None:
    probs = np.array([cfg.p_simple, cfg.p_medium, cfg.p_complex], dtype=float)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("Вероятности сложности должны суммироваться в 1.")
    if np.any(probs < 0):
        raise ValueError("Вероятности не могут быть отрицательными.")
    if any(window < 2048 for window in cfg.teacher_context_windows):
        raise ValueError("Контекстное окно должно быть не меньше 2048 токенов.")
    if not (0.0 < cfg.quality_target <= 1.0):
        raise ValueError("quality_target должен лежать в (0, 1].")


def stable_normal(request_id: int, channel: int, seed: int) -> float:
    """Детерминированный N(0,1) для парного сравнения стратегий."""
    x1 = (request_id + 1) * (12.9898 + channel * 0.177) + seed * 0.119
    x2 = (request_id + 1) * (39.3467 + channel * 0.137) + seed * 0.173
    u1 = (math.sin(x1) * 43758.5453123) % 1.0
    u2 = (math.sin(x2) * 24634.6345217) % 1.0
    u1 = min(max(u1, 1e-12), 1.0 - 1e-12)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def generate_queries(n: int, cfg: SimulationConfig, seed_offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    classes = rng.choice(
        np.array(["simple", "medium", "complex"]),
        size=n,
        p=[cfg.p_simple, cfg.p_medium, cfg.p_complex],
    )

    records: List[dict] = []
    for index, complexity in enumerate(classes):
        request_id = seed_offset * 1_000_000 + index

        if complexity == "simple":
            z = rng.uniform(0.08, 0.34)
            n_domains = 1
            query_tokens = int(rng.integers(170, 430))
            answer_need = cfg.output_tokens_simple
            rag_base = 0.79
        elif complexity == "medium":
            z = rng.uniform(0.35, 0.69)
            n_domains = int(rng.integers(2, 4))
            query_tokens = int(rng.integers(360, 820))
            answer_need = cfg.output_tokens_medium
            rag_base = 0.62
        else:
            z = rng.uniform(0.70, 0.98)
            n_domains = int(rng.integers(3, 6))
            query_tokens = int(rng.integers(760, 1550))
            answer_need = cfg.output_tokens_complex
            rag_base = 0.45

        selected = rng.choice(len(DOMAINS), size=n_domains, replace=False)
        raw_weights = rng.dirichlet(np.full(n_domains, 1.35))
        requirement = np.zeros(len(DOMAINS), dtype=float)
        requirement[selected] = raw_weights

        freshness_sensitive = bool(rng.random() < (0.16 + 0.19 * z))
        rag_noise = rng.normal(0.0, 0.10, size=len(DOMAINS))
        rag_coverage = np.clip(rag_base + rag_noise - 0.10 * z, 0.03, 0.96)
        rag_coverage *= (0.30 + 0.70 * (requirement > 0))

        normative_index = DOMAINS.index("нормативная_база")
        if freshness_sensitive and requirement[normative_index] > 0:
            rag_coverage[normative_index] *= rng.uniform(0.42, 0.70)

        weighted_rag = float(np.dot(requirement, rag_coverage))
        rag_confidence = float(
            np.clip(weighted_rag + rng.normal(0.0, 0.055) - 0.025 * z, 0.0, 1.0)
        )
        rag_tokens = int(
            np.clip(
                520 + 640 * n_domains + 1150 * z + rng.normal(0.0, 180),
                450,
                5200,
            )
        )

        records.append(
            {
                "request_id": request_id,
                "complexity": complexity,
                "z": z,
                "query_tokens": query_tokens,
                "answer_need_tokens": answer_need,
                "freshness_sensitive": freshness_sensitive,
                "rag_tokens": rag_tokens,
                "rag_confidence": rag_confidence,
                "requirement_json": json.dumps(requirement.tolist()),
                "rag_coverage_json": json.dumps(rag_coverage.tolist()),
            }
        )

    return pd.DataFrame(records)


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def combine_coverage(current: np.ndarray, contribution: np.ndarray) -> np.ndarray:
    """Независимое объединение покрытия: 1-(1-a)(1-b)."""
    return 1.0 - (1.0 - current) * (1.0 - contribution)


def agent_true_contribution(
    agent: AgentProfile,
    requirement: np.ndarray,
    z: float,
    request_id: int,
    agent_index: int,
    cfg: SimulationConfig,
) -> np.ndarray:
    expertise = np.asarray(agent.expertise, dtype=float)
    domain_need = 0.20 + 0.80 * (requirement > 0).astype(float)
    stochastic = 1.0 + 0.08 * stable_normal(request_id, 100 + agent_index, cfg.random_seed)
    complexity_factor = 0.78 + 0.10 * z
    contribution = expertise * agent.reliability * domain_need * complexity_factor * stochastic
    return np.clip(contribution, 0.01, 0.90)


def agent_token_costs(
    agent: AgentProfile,
    requirement: np.ndarray,
    query_tokens: int,
    rag_tokens: int,
    z: float,
    request_id: int,
    agent_index: int,
    cfg: SimulationConfig,
) -> Tuple[int, int]:
    expertise = np.asarray(agent.expertise, dtype=float)
    relevance = float(np.dot(requirement, expertise))
    rag_excerpt = min(rag_tokens, int(cfg.rag_agent_excerpt_max * (0.72 + 0.28 * relevance)))
    input_tokens = int(
        agent.system_prompt_tokens
        + query_tokens
        + rag_excerpt
        + 120
        + max(0.0, 90.0 * stable_normal(request_id, 200 + agent_index, cfg.random_seed))
    )
    output_tokens = int(
        np.clip(
            260 + 520 * relevance + 210 * z
            + 55 * stable_normal(request_id, 300 + agent_index, cfg.random_seed),
            220,
            1250,
        )
    )
    return input_tokens, output_tokens


def estimated_agent_value(
    agent: AgentProfile,
    estimated_requirement: np.ndarray,
    estimated_coverage: np.ndarray,
    query_tokens: int,
    rag_tokens: int,
    z: float,
    cfg: SimulationConfig,
) -> Tuple[float, float]:
    expertise = np.asarray(agent.expertise, dtype=float)
    need = estimated_requirement * (1.0 - estimated_coverage)
    expected_coverage_gain = float(np.dot(need, expertise * agent.reliability * (0.70 + 0.10 * z)))
    relevance = float(np.dot(estimated_requirement, expertise))
    expected_tokens = (
        agent.system_prompt_tokens
        + query_tokens
        + min(rag_tokens, cfg.rag_agent_excerpt_max)
        + 120
        + 260
        + 520 * relevance
        + 210 * z
    )
    expected_quality_gain = 0.43 * expected_coverage_gain
    return expected_quality_gain, max(expected_tokens, 1.0)


def predicted_quality_from_coverage(weighted_coverage: float, rag_confidence: float) -> float:
    return float(np.clip(0.51 + 0.39 * weighted_coverage + 0.10 * rag_confidence, 0.0, 1.0))


def select_dynamic_agents(
    row: pd.Series,
    cfg: SimulationConfig,
) -> Tuple[List[int], int, int, str]:
    """Последовательный cost-aware выбор агентов без доступа к истинному ответу."""
    requirement = parse_vector(row["requirement_json"])
    rag_coverage = parse_vector(row["rag_coverage_json"])
    request_id = int(row["request_id"])
    z = float(row["z"])
    query_tokens = int(row["query_tokens"])
    rag_tokens = int(row["rag_tokens"])

    # Router API: исходный запрос + краткий RAG-сниппет -> JSON с оценкой доменов.
    router_input = cfg.router_system_tokens + query_tokens + min(rag_tokens, 650)
    router_output = cfg.router_output_tokens

    noise = np.array(
        [stable_normal(request_id, 400 + i, cfg.random_seed) for i in range(len(DOMAINS))]
    )
    estimated_requirement = np.clip(requirement + cfg.routing_noise_std * noise, 0.0, None)
    if estimated_requirement.sum() <= 1e-9:
        estimated_requirement = np.full(len(DOMAINS), 1.0 / len(DOMAINS))
    else:
        estimated_requirement /= estimated_requirement.sum()

    estimated_coverage = np.clip(rag_coverage + 0.045 * noise, 0.0, 0.98)
    selected: List[int] = []
    remaining = set(range(len(AGENTS)))
    decision_tokens = 0
    stop_reason = "agent_limit"

    while remaining and len(selected) < cfg.max_dynamic_agents:
        weighted_coverage = float(np.dot(estimated_requirement, estimated_coverage))
        predicted_q = predicted_quality_from_coverage(weighted_coverage, float(row["rag_confidence"]))
        if len(selected) >= cfg.early_stop_min_agents and predicted_q >= cfg.dynamic_predicted_quality_target:
            stop_reason = "predicted_quality_reached"
            break

        candidates: List[Tuple[int, float, float, float]] = []
        for agent_index in sorted(remaining):
            gain, expected_tokens = estimated_agent_value(
                AGENTS[agent_index],
                estimated_requirement,
                estimated_coverage,
                query_tokens,
                rag_tokens,
                z,
                cfg,
            )
            ratio = gain / (expected_tokens / 1000.0)
            candidates.append((agent_index, gain, expected_tokens, ratio))

        best_index, best_gain, _, best_ratio = max(candidates, key=lambda item: item[3])
        if len(selected) >= cfg.early_stop_min_agents and (
            best_ratio < cfg.marginal_quality_per_1k_tokens or best_gain < 0.004
        ):
            stop_reason = "marginal_utility"
            break

        selected.append(best_index)
        remaining.remove(best_index)
        contribution_est = np.asarray(AGENTS[best_index].expertise) * AGENTS[best_index].reliability * 0.72
        estimated_coverage = combine_coverage(estimated_coverage, contribution_est)

        # Короткий API-вызов координатора для пересмотра маршрута после каждого агента.
        decision_tokens += cfg.decision_input_tokens + cfg.decision_output_tokens

    if not selected:
        stop_reason = "no_agent_selected"
    elif len(selected) >= cfg.max_dynamic_agents:
        stop_reason = "agent_limit"

    return selected, router_input + decision_tokens, router_output, stop_reason


def synthesis_quality(
    row: pd.Series,
    selected_agents: Sequence[int],
    context_window: int,
    strategy: Strategy,
    cfg: SimulationConfig,
) -> Dict[str, float | int]:
    requirement = parse_vector(row["requirement_json"])
    rag_coverage = parse_vector(row["rag_coverage_json"])
    request_id = int(row["request_id"])
    query_tokens = int(row["query_tokens"])
    rag_tokens = int(row["rag_tokens"])
    z = float(row["z"])
    desired_output = int(row["answer_need_tokens"])

    evidence: List[dict] = []
    rag_piece_tokens = min(rag_tokens, cfg.rag_synthesis_excerpt_max)
    rag_relevance = float(np.dot(requirement, rag_coverage))
    evidence.append(
        {
            "kind": "RAG",
            "tokens": rag_piece_tokens,
            "contribution": rag_coverage,
            "utility": rag_relevance + 0.08,
            "reliability": 0.96,
        }
    )

    cloud_input = 0
    cloud_output = 0
    agent_output_total = 0
    agent_reliabilities: List[float] = []

    for agent_index in selected_agents:
        agent = AGENTS[agent_index]
        contribution = agent_true_contribution(
            agent,
            requirement,
            z,
            request_id,
            agent_index,
            cfg,
        )
        input_tokens, output_tokens = agent_token_costs(
            agent,
            requirement,
            query_tokens,
            rag_tokens,
            z,
            request_id,
            agent_index,
            cfg,
        )
        cloud_input += input_tokens
        cloud_output += output_tokens
        agent_output_total += output_tokens
        agent_reliabilities.append(agent.reliability)
        evidence.append(
            {
                "kind": agent.name,
                "tokens": output_tokens,
                "contribution": contribution,
                "utility": float(np.dot(requirement, contribution)),
                "reliability": agent.reliability,
            }
        )

    # Резерв для ответа преподавателю. Чем меньше окно, тем сильнее ограничение.
    max_output_by_window = max(320, int(context_window * 0.22))
    actual_output = min(desired_output, max_output_by_window)
    answer_fit = min(1.0, actual_output / max(desired_output, 1))

    synthesis_fixed = cfg.synthesis_system_tokens + query_tokens
    evidence_capacity = max(0, context_window - synthesis_fixed - actual_output)

    # Координатор сортирует доказательства по полезности на токен.
    ranked = sorted(
        evidence,
        key=lambda item: float(item["utility"]) / max(int(item["tokens"]), 1),
        reverse=True,
    )
    included: List[Tuple[dict, float]] = []
    remaining_capacity = evidence_capacity
    total_evidence_tokens = sum(int(item["tokens"]) for item in evidence)
    for item in ranked:
        if remaining_capacity <= 0:
            included.append((item, 0.0))
            continue
        take = min(int(item["tokens"]), remaining_capacity)
        fraction = take / max(int(item["tokens"]), 1)
        included.append((item, fraction))
        remaining_capacity -= take

    final_coverage = np.zeros(len(DOMAINS), dtype=float)
    included_evidence_tokens = 0.0
    included_irrelevant_tokens = 0.0
    included_reliability_weight = 0.0
    reliability_denominator = 0.0
    rag_fraction = 0.0

    for item, fraction in included:
        if fraction <= 0:
            continue
        contribution = np.asarray(item["contribution"], dtype=float) * fraction
        final_coverage = combine_coverage(final_coverage, contribution)
        token_weight = int(item["tokens"]) * fraction
        included_evidence_tokens += token_weight
        item_relevance = float(item["utility"])
        included_irrelevant_tokens += token_weight * max(0.0, 1.0 - item_relevance)
        included_reliability_weight += token_weight * float(item["reliability"])
        reliability_denominator += token_weight
        if item["kind"] == "RAG":
            rag_fraction = fraction

    weighted_coverage = float(np.dot(requirement, final_coverage))
    overflow_ratio = max(0.0, (total_evidence_tokens - evidence_capacity) / max(total_evidence_tokens, 1))
    irrelevant_ratio = included_irrelevant_tokens / max(included_evidence_tokens, 1.0)
    mean_reliability = included_reliability_weight / max(reliability_denominator, 1.0)

    # Lost-in-the-middle / перегрузка контекста: штраф появляется при заполнении > 90%.
    utilization = (synthesis_fixed + included_evidence_tokens + actual_output) / context_window
    overload_penalty = max(0.0, utilization - 0.90) * 0.22

    factuality = np.clip(
        0.69
        + 0.245 * weighted_coverage
        + 0.055 * mean_reliability
        + 0.025 * rag_fraction
        - 0.070 * irrelevant_ratio
        - 0.055 * overflow_ratio
        - overload_penalty,
        0.0,
        1.0,
    )
    grounding = np.clip(
        0.50
        + 0.29 * rag_relevance * rag_fraction
        + 0.16 * mean_reliability
        + 0.08 * weighted_coverage
        - 0.06 * overflow_ratio,
        0.0,
        1.0,
    )
    applicability = np.clip(
        0.52 + 0.31 * weighted_coverage + 0.17 * answer_fit - 0.04 * overflow_ratio,
        0.0,
        1.0,
    )

    quality = (
        0.35 * weighted_coverage
        + 0.30 * factuality
        + 0.20 * grounding
        + 0.15 * applicability
    )
    quality += cfg.quality_noise_std * stable_normal(
        request_id,
        700 + context_window // 1024 + (0 if strategy == "STATIC_ALL" else 20),
        cfg.random_seed,
    )
    quality = float(np.clip(quality, 0.0, 1.0))

    synthesis_input = int(synthesis_fixed + included_evidence_tokens)
    cloud_input += synthesis_input
    cloud_output += actual_output

    return {
        "quality": quality,
        "coverage": weighted_coverage,
        "factuality": float(factuality),
        "grounding": float(grounding),
        "applicability": float(applicability),
        "answer_fit": float(answer_fit),
        "overflow_ratio": float(overflow_ratio),
        "context_utilization": float(utilization),
        "cloud_input_tokens": int(cloud_input),
        "cloud_output_tokens": int(cloud_output),
        "cloud_total_tokens": int(cloud_input + cloud_output),
        "synthesis_input_tokens": synthesis_input,
        "synthesis_output_tokens": actual_output,
        "agent_output_tokens": agent_output_total,
    }


def simulate_direct_rag(
    row: pd.Series,
    context_window: int,
    strategy: Strategy,
    cfg: SimulationConfig,
) -> Dict[str, float | int]:
    """RAG-only: локальное извлечение + один облачный synthesis call."""
    result = synthesis_quality(row, [], context_window, strategy, cfg)
    # Для direct RAG используется более короткий system prompt.
    reduction = cfg.synthesis_system_tokens - cfg.direct_rag_system_tokens
    result["cloud_input_tokens"] = max(0, int(result["cloud_input_tokens"]) - reduction)
    result["cloud_total_tokens"] = int(result["cloud_input_tokens"]) + int(result["cloud_output_tokens"])
    result["synthesis_input_tokens"] = max(0, int(result["synthesis_input_tokens"]) - reduction)
    return result


def simulate_one(
    row: pd.Series,
    context_window: int,
    strategy: Strategy,
    cfg: SimulationConfig,
) -> dict:
    rag_direct = float(row["rag_confidence"]) >= cfg.rag_direct_threshold
    router_input_tokens = 0
    router_output_tokens = 0
    route_stop_reason = "rag_direct"

    if rag_direct:
        selected_agents: List[int] = []
        result = simulate_direct_rag(row, context_window, strategy, cfg)
        route = "RAG_ONLY"
    else:
        if strategy == "STATIC_ALL":
            selected_agents = list(range(len(AGENTS)))
            route_stop_reason = "fixed_all_agents"
        else:
            selected_agents, router_input_tokens, router_output_tokens, route_stop_reason = select_dynamic_agents(row, cfg)
        result = synthesis_quality(row, selected_agents, context_window, strategy, cfg)
        route = "RAG_PLUS_AGENTS"

    cloud_input = int(result["cloud_input_tokens"]) + router_input_tokens
    cloud_output = int(result["cloud_output_tokens"]) + router_output_tokens
    total_tokens = cloud_input + cloud_output
    if total_tokens > cfg.max_cloud_tokens_per_query:
        total_tokens = cfg.max_cloud_tokens_per_query

    estimated_cost = (
        cloud_input * cfg.price_input_per_million_usd
        + cloud_output * cfg.price_output_per_million_usd
    ) / 1_000_000.0

    exact_answer = bool(
        float(result["quality"]) >= cfg.quality_target
        and float(result["factuality"]) >= cfg.factuality_min
        and float(result["coverage"]) >= cfg.coverage_min
        and float(result["answer_fit"]) >= 0.92
    )

    return {
        "request_id": int(row["request_id"]),
        "complexity": str(row["complexity"]),
        "context_window": context_window,
        "strategy": strategy,
        "route": route,
        "route_stop_reason": route_stop_reason,
        "rag_confidence": float(row["rag_confidence"]),
        "selected_agents": "; ".join(AGENTS[i].name for i in selected_agents),
        "agents_used": len(selected_agents),
        "quality": float(result["quality"]),
        "coverage": float(result["coverage"]),
        "factuality": float(result["factuality"]),
        "grounding": float(result["grounding"]),
        "applicability": float(result["applicability"]),
        "answer_fit": float(result["answer_fit"]),
        "overflow_ratio": float(result["overflow_ratio"]),
        "context_utilization": float(result["context_utilization"]),
        "cloud_input_tokens": cloud_input,
        "cloud_output_tokens": cloud_output,
        "cloud_total_tokens": total_tokens,
        "estimated_api_cost_usd": estimated_cost,
        "exact_answer": exact_answer,
    }


def run_simulation(queries: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    records: List[dict] = []
    for context_window in cfg.teacher_context_windows:
        for strategy in ("STATIC_ALL", "DYNAMIC_ACG"):
            for _, row in queries.iterrows():
                records.append(simulate_one(row, context_window, strategy, cfg))
    return pd.DataFrame(records)


def summarize(results: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    rows: List[dict] = []
    grouped = results.groupby(["context_window", "strategy"], observed=True)
    for (window, strategy), group in grouped:
        rows.append(
            {
                "context_window": int(window),
                "strategy": strategy,
                "mean_quality": group["quality"].mean(),
                "p10_quality": group["quality"].quantile(0.10),
                "mean_coverage": group["coverage"].mean(),
                "mean_factuality": group["factuality"].mean(),
                "exact_answer_rate": group["exact_answer"].mean(),
                "mean_cloud_tokens": group["cloud_total_tokens"].mean(),
                "median_cloud_tokens": group["cloud_total_tokens"].median(),
                "p95_cloud_tokens": group["cloud_total_tokens"].quantile(0.95),
                "mean_input_tokens": group["cloud_input_tokens"].mean(),
                "mean_output_tokens": group["cloud_output_tokens"].mean(),
                "mean_agents_used": group["agents_used"].mean(),
                "rag_only_rate": (group["route"] == "RAG_ONLY").mean(),
                "mean_overflow_ratio": group["overflow_ratio"].mean(),
                "mean_estimated_api_cost_usd": group["estimated_api_cost_usd"].mean(),
                "quality_target": cfg.quality_target,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["context_window", "strategy"]).reset_index(drop=True)

    savings_rows: List[dict] = []
    for window in cfg.teacher_context_windows:
        static = summary[(summary.context_window == window) & (summary.strategy == "STATIC_ALL")].iloc[0]
        dynamic = summary[(summary.context_window == window) & (summary.strategy == "DYNAMIC_ACG")].iloc[0]
        savings_rows.append(
            {
                "context_window": window,
                "token_saving_dynamic_vs_static_pct": 100.0 * (
                    1.0 - dynamic.mean_cloud_tokens / static.mean_cloud_tokens
                ),
                "quality_delta_dynamic_minus_static": dynamic.mean_quality - static.mean_quality,
                "exact_answer_rate_delta": dynamic.exact_answer_rate - static.exact_answer_rate,
            }
        )
    savings = pd.DataFrame(savings_rows)
    return summary.merge(savings, on="context_window", how="left")


def agent_usage_table(results: pd.DataFrame) -> pd.DataFrame:
    dynamic = results[results["strategy"] == "DYNAMIC_ACG"].copy()
    rows: List[dict] = []
    for (window, complexity), group in dynamic.groupby(["context_window", "complexity"], observed=True):
        rows.append(
            {
                "context_window": int(window),
                "complexity": complexity,
                "mean_agents_used": group["agents_used"].mean(),
                "mean_cloud_tokens": group["cloud_total_tokens"].mean(),
                "mean_quality": group["quality"].mean(),
                "exact_answer_rate": group["exact_answer"].mean(),
                "rag_only_rate": (group["route"] == "RAG_ONLY").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["context_window", "complexity"])


def plot_tokens(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy, group in summary.groupby("strategy", observed=True):
        group = group.sort_values("context_window")
        ax.plot(group["context_window"], group["mean_cloud_tokens"], marker="o", label=strategy)
    ax.set_title("Средний расход облачных токенов")
    ax.set_xlabel("Контекстное окно преподавательского сценария, токены")
    ax.set_ylabel("Средние input + output tokens на запрос")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_quality(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy, group in summary.groupby("strategy", observed=True):
        group = group.sort_values("context_window")
        ax.plot(group["context_window"], group["mean_quality"], marker="o", label=strategy)
    target = float(summary["quality_target"].iloc[0])
    ax.axhline(target, linestyle="--", label="Q target")
    ax.set_title("Среднее качество ответа преподавателю")
    ax.set_xlabel("Контекстное окно преподавательского сценария, токены")
    ax.set_ylabel("Композитное качество Q")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_agents(agent_usage: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for complexity, group in agent_usage.groupby("complexity", observed=True):
        group = group.sort_values("context_window")
        ax.plot(group["context_window"], group["mean_agents_used"], marker="o", label=complexity)
    ax.set_title("Динамическая политика: число субагентов")
    ax.set_xlabel("Контекстное окно преподавательского сценария, токены")
    ax.set_ylabel("Среднее число вызванных субагентов")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scatter(results: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sample = results.sample(min(2600, len(results)), random_state=17)
    for strategy, group in sample.groupby("strategy", observed=True):
        ax.scatter(
            group["cloud_total_tokens"],
            group["quality"],
            alpha=0.35,
            s=18,
            label=strategy,
        )
    ax.set_title("Качество–стоимость по отдельным запросам")
    ax.set_xlabel("Облачные input + output tokens")
    ax.set_ylabel("Качество Q")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_routes(results: pd.DataFrame, output: Path) -> None:
    dynamic = results[results["strategy"] == "DYNAMIC_ACG"].copy()
    route_counts = (
        dynamic.groupby(["context_window", "route"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    route_share = route_counts.div(route_counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    route_share.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Доли маршрутов динамической политики")
    ax.set_xlabel("Контекстное окно")
    ax.set_ylabel("Доля запросов")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    view = df.loc[:, columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    try:
        return view.to_markdown(index=False)
    except ImportError:
        headers = list(view.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in view.itertuples(index=False):
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)


def build_report(summary: pd.DataFrame, agent_usage: pd.DataFrame, cfg: SimulationConfig) -> str:
    compact = summary[
        [
            "context_window",
            "strategy",
            "mean_quality",
            "p10_quality",
            "exact_answer_rate",
            "mean_cloud_tokens",
            "p95_cloud_tokens",
            "mean_agents_used",
            "token_saving_dynamic_vs_static_pct",
            "quality_delta_dynamic_minus_static",
        ]
    ]
    dynamic = summary[summary["strategy"] == "DYNAMIC_ACG"].sort_values("context_window")
    best = dynamic.loc[dynamic["mean_cloud_tokens"].idxmin()]
    largest = dynamic.iloc[-1]

    report = r"""# Отчет о симуляции sim_01

## Цель

Минимизировать суммарное число облачных API-токенов при условии, что итоговый
ответ преподавателю достигает требуемого качества. Локальный RAG-поиск не
считается облачным токенным расходом; учитываются все токены, переданные и
полученные при routing, вызовах субагентов и финальном синтезе.

## Сравниваемые политики

- `STATIC_ALL`: если RAG недостаточен, вызываются все __AGENT_COUNT__ субагентов.
- `DYNAMIC_ACG`: вызываются только субагенты с максимальной ожидаемой
  маржинальной полезностью, после каждого вызова допускается ранняя остановка.

## Определение точного ответа

Ответ считается точным, если одновременно выполняются условия:

```math
Q \ge __QUALITY_TARGET__, \qquad
F \ge __FACTUALITY_MIN__, \qquad
C_{\mathrm{req}} \ge __COVERAGE_MIN__, \qquad
A_{\mathrm{fit}} \ge 0.92.
```

Здесь `Q` — композитная оценка, `F` — фактическая корректность,
`C_req` — покрытие требований преподавателя, `A_fit` — отсутствие усечения
необходимого ответа выбранным контекстным окном.

## Результаты

__SUMMARY_TABLE__

Для минимального исследованного окна __BEST_WINDOW__ токенов
динамическая политика использует в среднем __BEST_TOKENS__
облачных токенов на запрос. Для окна __LARGEST_WINDOW__ токенов
ее среднее качество составляет __LARGEST_QUALITY__, а доля точных
ответов — __LARGEST_EXACT__.

## Адаптация по сложности

__AGENT_TABLE__

## Интерпретация

Динамическая политика экономит токены по двум причинам: значительная часть
простых запросов завершается после RAG, а для остальных запросов вызывается
не весь пул, а релевантное подмножество субагентов. Увеличение контекстного
окна уменьшает усечение доказательств и ответа, однако после достижения
достаточной емкости дальнейшее расширение окна дает убывающий прирост качества.

Результаты синтетические. Для production-параметризации необходимы реальные
API-трассы, экспертные оценки преподавателей и статистика RAG-покрытия.
"""
    replacements = {
        "__AGENT_COUNT__": str(len(AGENTS)),
        "__QUALITY_TARGET__": str(cfg.quality_target),
        "__FACTUALITY_MIN__": str(cfg.factuality_min),
        "__COVERAGE_MIN__": str(cfg.coverage_min),
        "__SUMMARY_TABLE__": markdown_table(compact, compact.columns),
        "__BEST_WINDOW__": str(int(best.context_window)),
        "__BEST_TOKENS__": f"{best.mean_cloud_tokens:.1f}",
        "__LARGEST_WINDOW__": str(int(largest.context_window)),
        "__LARGEST_QUALITY__": f"{largest.mean_quality:.4f}",
        "__LARGEST_EXACT__": f"{largest.exact_answer_rate:.4f}",
        "__AGENT_TABLE__": markdown_table(agent_usage, agent_usage.columns),
    }
    for key, value in replacements.items():
        report = report.replace(key, value)
    return report

def save_parameters(cfg: SimulationConfig, output: Path) -> None:
    rows: List[dict] = []
    for key, value in asdict(cfg).items():
        if isinstance(value, tuple):
            value = ";".join(map(str, value))
        rows.append({"parameter": key, "value": value})
    pd.DataFrame(rows).to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational multi-agent token optimization simulation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Каталог результатов.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig()
    validate_config(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train set оставлен для последующей калибровки порогов; текущая версия
    # использует предзаданные policy-параметры и оценивается на test set.
    train = generate_queries(cfg.train_requests, cfg, seed_offset=1)
    test = generate_queries(cfg.test_requests, cfg, seed_offset=2)
    train.to_csv(args.output_dir / "synthetic_train_queries.csv", index=False)
    test.to_csv(args.output_dir / "synthetic_test_queries.csv", index=False)

    results = run_simulation(test, cfg)
    summary = summarize(results, cfg)
    agent_usage = agent_usage_table(results)

    results.to_csv(args.output_dir / "per_query_results.csv", index=False)
    summary.to_csv(args.output_dir / "comparison_by_context.csv", index=False)
    agent_usage.to_csv(args.output_dir / "agent_usage.csv", index=False)
    save_parameters(cfg, args.output_dir / "simulation_parameters.csv")

    plot_tokens(summary, args.output_dir / "01_tokens_by_context.png")
    plot_quality(summary, args.output_dir / "02_quality_by_context.png")
    plot_agents(agent_usage, args.output_dir / "03_agents_by_context.png")
    plot_scatter(results, args.output_dir / "04_quality_cost_scatter.png")
    plot_routes(results, args.output_dir / "05_route_distribution.png")

    report = build_report(summary, agent_usage, cfg)
    (args.output_dir / "simulation_report.md").write_text(report, encoding="utf-8")

    print("\n=== SIM_01: EDUCATIONAL MULTI-AGENT ROUTING ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
