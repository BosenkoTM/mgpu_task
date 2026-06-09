#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симуляция двух парадигм оптимизации мультиагентных LLM-систем.

Модель A:
    Статический Grid Search по фиксированным N и I.
Модель B:
    Динамическая политика маршрутизации и раннего останова,
    где N и I являются результатом траектории исполнения.

Зависимости:
    pip install numpy pandas matplotlib

Запуск:
    python multiagent_llm_optimization_simulation.py

Результаты сохраняются в каталоге:
    multiagent_simulation_outputs/

Файлы:
    static_grid_train.csv
    static_test_results.csv
    dynamic_test_results.csv
    comparison_summary.csv
    comparison_by_complexity.csv
    analytical_report.md
    01_static_tradeoff.png
    02_quality_boxplot.png
    03_cost_boxplot.png
    04_dynamic_agents_histogram.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Action = Literal["agent", "reflection"]


# =============================================================================
# ### [CHANGE HERE: PARAMETERS TO PLAY WITH] ###
# Меняйте параметры этого dataclass, чтобы моделировать собственные сценарии:
# стоимость токенов, распределение сложности, штрафы за галлюцинации,
# качество верификатора, порог ранней остановки и бюджет.
# =============================================================================
@dataclass(frozen=True)
class SimulationConfig:
    # Размер эксперимента
    train_requests: int = 900
    test_requests: int = 1200
    random_seed: int = 42

    # Распределение классов сложности: simple / medium / hard
    difficulty_prob_simple: float = 0.45
    difficulty_prob_medium: float = 0.35
    difficulty_prob_hard: float = 0.20

    # Диапазоны скрытой сложности z(x) внутри каждого класса
    simple_z_min: float = 0.05
    simple_z_max: float = 0.35
    medium_z_min: float = 0.35
    medium_z_max: float = 0.70
    hard_z_min: float = 0.70
    hard_z_max: float = 0.98

    # Границы Grid Search для Модели A
    static_n_min: int = 1
    static_n_max: int = 6
    static_i_min: int = 0
    static_i_max: int = 4

    # Максимальные границы траектории Модели B
    dynamic_n_max: int = 7
    dynamic_i_max: int = 5

    # Токенная стоимость действий
    base_agent_tokens: float = 760.0
    base_reflection_tokens: float = 470.0
    context_growth_per_step: float = 0.11
    context_quadratic_growth: float = 0.010
    cost_noise_std: float = 0.045

    # Модель прироста качества
    base_quality_easy_bonus: float = 0.20
    base_quality_intercept: float = 0.25
    baseline_quality_noise_std: float = 0.055
    agent_gain_scale: float = 0.52
    reflection_gain_scale: float = 0.31
    agent_diminishing_decay: float = 0.38
    reflection_diminishing_decay: float = 0.55
    step_quality_noise_std: float = 0.012

    # Точка насыщения и деградация / overthinking
    optimal_effort_base: float = 1.25
    optimal_effort_difficulty_scale: float = 4.50
    reflection_effort_weight: float = 0.78
    hallucination_penalty: float = 0.026
    hallucination_exponent: float = 1.55

    # Ограничения качества и бюджета
    quality_target_train: float = 0.835
    minimum_acceptable_quality: float = 0.66
    maximum_failure_rate_train: float = 0.18
    average_token_budget_static: float = 5700.0
    per_request_token_budget_dynamic: float = 5700.0

    # Параметры динамического роутинга / Early Stopping
    routing_difficulty_noise_std: float = 0.075
    verifier_noise_std: float = 0.018
    dynamic_stop_quality: float = 0.815
    lambda_quality_per_1k_tokens: float = 0.025
    minimum_predicted_gain: float = 0.010
    minimum_steps_before_marginal_stop: int = 1

    # Небольшая консервативная поправка политики на неопределенность
    uncertainty_penalty_agent: float = 0.004
    uncertainty_penalty_reflection: float = 0.006

    # Вес стоимости в fallback utility, если ограничения Grid Search невыполнимы
    fallback_cost_weight_per_1k_tokens: float = 0.018
# =============================================================================
# ### [END CHANGE HERE] ###
# =============================================================================


def validate_config(cfg: SimulationConfig) -> None:
    """Проверяет непротиворечивость основных параметров."""
    probabilities = np.array(
        [
            cfg.difficulty_prob_simple,
            cfg.difficulty_prob_medium,
            cfg.difficulty_prob_hard,
        ],
        dtype=float,
    )
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Вероятности классов сложности должны суммироваться в 1.")
    if np.any(probabilities < 0):
        raise ValueError("Вероятности классов сложности не могут быть отрицательными.")
    if cfg.static_n_min < 1 or cfg.dynamic_n_max < 1:
        raise ValueError("Число агентов должно быть не меньше 1.")
    if cfg.static_i_min < 0 or cfg.dynamic_i_max < 0:
        raise ValueError("Число итераций не может быть отрицательным.")
    if cfg.per_request_token_budget_dynamic <= 0:
        raise ValueError("Динамический бюджет должен быть положительным.")
    if not 0 <= cfg.minimum_acceptable_quality <= 1:
        raise ValueError("Порог качества должен лежать в [0, 1].")


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Численно устойчивая логистическая функция."""
    x_arr = np.asarray(x, dtype=float)
    result = np.empty_like(x_arr)
    positive = x_arr >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x_arr[positive]))
    exp_x = np.exp(x_arr[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return float(result) if np.isscalar(x) else result


def deterministic_normal(
    request_ids: np.ndarray,
    step: int,
    channel: int,
    seed: int,
) -> np.ndarray:
    """
    Детерминированный псевдослучайный N(0,1) для парного сравнения стратегий.

    Одинаковый request_id и одинаковый тип/номер действия получают одинаковый
    шум независимо от того, какая стратегия вызвала действие.
    """
    ids = np.asarray(request_ids, dtype=float)
    x1 = (
        (ids + 1.0) * (12.9898 + 0.371 * channel)
        + step * (78.233 + 0.193 * channel)
        + seed * 0.117
    )
    x2 = (
        (ids + 1.0) * (39.3467 + 0.271 * channel)
        + step * (11.135 + 0.157 * channel)
        + seed * 0.173
    )
    u1 = np.mod(np.sin(x1) * 43758.5453123, 1.0)
    u2 = np.mod(np.sin(x2) * 24634.6345217, 1.0)
    u1 = np.clip(u1, 1e-12, 1.0 - 1e-12)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def generate_requests(
    n_requests: int,
    cfg: SimulationConfig,
    seed_offset: int,
) -> pd.DataFrame:
    """Генерирует поток запросов с латентной сложностью z(x)."""
    rng = np.random.default_rng(cfg.random_seed + seed_offset)

    labels = np.array(["simple", "medium", "hard"])
    probabilities = np.array(
        [
            cfg.difficulty_prob_simple,
            cfg.difficulty_prob_medium,
            cfg.difficulty_prob_hard,
        ]
    )
    complexity_class = rng.choice(labels, size=n_requests, p=probabilities)

    z = np.empty(n_requests, dtype=float)
    ranges = {
        "simple": (cfg.simple_z_min, cfg.simple_z_max),
        "medium": (cfg.medium_z_min, cfg.medium_z_max),
        "hard": (cfg.hard_z_min, cfg.hard_z_max),
    }
    for label, (left, right) in ranges.items():
        mask = complexity_class == label
        # Beta(2,2) сохраняет вариативность, но избегает концентрации у границ.
        local = rng.beta(2.0, 2.0, size=int(mask.sum()))
        z[mask] = left + (right - left) * local

    request_id = np.arange(seed_offset * 1_000_000, seed_offset * 1_000_000 + n_requests)
    baseline_noise = rng.normal(0.0, 1.0, size=n_requests)
    router_noise = rng.normal(0.0, 1.0, size=n_requests)
    verifier_bias = rng.normal(0.0, 0.45, size=n_requests)
    agent_affinity = np.clip(rng.normal(1.0, 0.08, size=n_requests), 0.78, 1.22)
    reflection_affinity = np.clip(rng.normal(1.0, 0.10, size=n_requests), 0.72, 1.28)

    return pd.DataFrame(
        {
            "request_id": request_id.astype(int),
            "complexity_class": complexity_class,
            "z": z,
            "baseline_noise": baseline_noise,
            "router_noise": router_noise,
            "verifier_bias": verifier_bias,
            "agent_affinity": agent_affinity,
            "reflection_affinity": reflection_affinity,
        }
    )


def initial_quality(requests: pd.DataFrame, cfg: SimulationConfig) -> np.ndarray:
    """Качество до первого агентного вызова."""
    q = (
        cfg.base_quality_intercept
        + cfg.base_quality_easy_bonus * (1.0 - requests["z"].to_numpy())
        + cfg.baseline_quality_noise_std * requests["baseline_noise"].to_numpy()
    )
    return np.clip(q, 0.08, 0.52)


def effort_value(agent_count: int | np.ndarray, reflection_count: int | np.ndarray, cfg: SimulationConfig):
    return np.asarray(agent_count) + cfg.reflection_effort_weight * np.asarray(reflection_count)


def optimal_effort(z: np.ndarray | float, cfg: SimulationConfig):
    return cfg.optimal_effort_base + cfg.optimal_effort_difficulty_scale * np.asarray(z)


def degradation_penalty(
    z: np.ndarray,
    next_agent_count: int | np.ndarray,
    next_reflection_count: int | np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Штраф за превышение task-specific точки насыщения."""
    effort = effort_value(next_agent_count, next_reflection_count, cfg)
    excess = np.maximum(0.0, effort - optimal_effort(z, cfg))
    difficulty_factor = 1.18 - 0.30 * z
    return (
        cfg.hallucination_penalty
        * difficulty_factor
        * np.power(excess, cfg.hallucination_exponent)
    )


def predicted_increment(
    q: np.ndarray | float,
    z: np.ndarray | float,
    action: Action,
    action_index: int,
    next_agent_count: int | np.ndarray,
    next_reflection_count: int | np.ndarray,
    cfg: SimulationConfig,
    affinity: np.ndarray | float = 1.0,
) -> np.ndarray | float:
    """
    Ожидаемый прирост качества без стохастического шума.

    Убывающая отдача задается экспоненциальным decay.
    Сложные задачи получают больше пользы от дополнительных вычислений,
    но начинают с более низкого исходного качества.
    """
    q_arr = np.asarray(q, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    affinity_arr = np.asarray(affinity, dtype=float)
    remaining = np.maximum(0.0, 1.0 - q_arr)

    if action == "agent":
        # Простые задачи получают сильный первый шаг; у сложных задач
        # отдача распределена по более длинной траектории (медленнее decay).
        need = 1.05 - 0.20 * z_arr
        decay_scale = 0.72 + 0.75 * z_arr
        raw_gain = (
            cfg.agent_gain_scale
            * need
            * np.exp(
                -cfg.agent_diminishing_decay
                * (action_index - 1)
                / decay_scale
            )
            * remaining
            * affinity_arr
        )
    elif action == "reflection":
        # Рефлексия полезнее после появления содержательного черновика.
        need = (0.76 - 0.12 * z_arr) * (0.72 + 0.46 * q_arr)
        decay_scale = 0.78 + 0.62 * z_arr
        raw_gain = (
            cfg.reflection_gain_scale
            * need
            * np.exp(
                -cfg.reflection_diminishing_decay
                * (action_index - 1)
                / decay_scale
            )
            * remaining
            * affinity_arr
        )
    else:
        raise ValueError(f"Неизвестное действие: {action}")

    penalty = degradation_penalty(
        z_arr,
        next_agent_count,
        next_reflection_count,
        cfg,
    )
    increment = raw_gain - penalty
    return float(increment) if np.isscalar(q) and np.isscalar(z) else increment


def predicted_step_cost(
    z: np.ndarray | float,
    action: Action,
    total_steps_before: int | np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray | float:
    """Ожидаемая токенная стоимость следующего действия."""
    z_arr = np.asarray(z, dtype=float)
    steps_arr = np.asarray(total_steps_before, dtype=float)
    base = cfg.base_agent_tokens if action == "agent" else cfg.base_reflection_tokens
    context_multiplier = (
        1.0
        + cfg.context_growth_per_step * steps_arr
        + cfg.context_quadratic_growth * np.square(steps_arr)
    )
    difficulty_multiplier = 1.0 + 0.16 * z_arr
    cost = base * context_multiplier * difficulty_multiplier
    return float(cost) if np.isscalar(z) and np.isscalar(total_steps_before) else cost


def apply_step_vectorized(
    requests: pd.DataFrame,
    q: np.ndarray,
    cost: np.ndarray,
    agent_count: int,
    reflection_count: int,
    action: Action,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет один шаг ко всему набору запросов для статической модели."""
    ids = requests["request_id"].to_numpy()
    z = requests["z"].to_numpy()
    total_steps_before = agent_count + reflection_count

    if action == "agent":
        action_index = agent_count + 1
        next_agents = agent_count + 1
        next_reflections = reflection_count
        affinity = requests["agent_affinity"].to_numpy()
        quality_channel = 10
        cost_channel = 20
    else:
        action_index = reflection_count + 1
        next_agents = agent_count
        next_reflections = reflection_count + 1
        affinity = requests["reflection_affinity"].to_numpy()
        quality_channel = 11
        cost_channel = 21

    expected_gain = predicted_increment(
        q=q,
        z=z,
        action=action,
        action_index=action_index,
        next_agent_count=next_agents,
        next_reflection_count=next_reflections,
        cfg=cfg,
        affinity=affinity,
    )
    q_noise = deterministic_normal(
        ids,
        step=action_index,
        channel=quality_channel,
        seed=cfg.random_seed,
    )
    q_new = np.clip(
        q + expected_gain + cfg.step_quality_noise_std * q_noise,
        0.0,
        1.0,
    )

    expected_cost = predicted_step_cost(
        z=z,
        action=action,
        total_steps_before=total_steps_before,
        cfg=cfg,
    )
    c_noise = deterministic_normal(
        ids,
        step=action_index,
        channel=cost_channel,
        seed=cfg.random_seed,
    )
    cost_multiplier = np.clip(1.0 + cfg.cost_noise_std * c_noise, 0.72, 1.35)
    cost_new = cost + expected_cost * cost_multiplier

    return q_new, cost_new


def simulate_static_configuration(
    requests: pd.DataFrame,
    n_agents: int,
    n_reflections: int,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    """Применяет фиксированные N и I ко всем запросам."""
    q = initial_quality(requests, cfg)
    cost = np.zeros(len(requests), dtype=float)

    agents = 0
    reflections = 0

    for _ in range(n_agents):
        q, cost = apply_step_vectorized(
            requests,
            q,
            cost,
            agents,
            reflections,
            "agent",
            cfg,
        )
        agents += 1

    for _ in range(n_reflections):
        q, cost = apply_step_vectorized(
            requests,
            q,
            cost,
            agents,
            reflections,
            "reflection",
            cfg,
        )
        reflections += 1

    result = requests[
        ["request_id", "complexity_class", "z"]
    ].copy()
    result["quality"] = q
    result["tokens"] = cost
    result["agents_used"] = n_agents
    result["reflections_used"] = n_reflections
    result["steps_used"] = n_agents + n_reflections
    result["stop_reason"] = "fixed_static_configuration"
    return result


def summarize_run(
    results: pd.DataFrame,
    cfg: SimulationConfig,
) -> dict[str, float]:
    """Сводные метрики одной стратегии."""
    quality = results["quality"].to_numpy()
    tokens = results["tokens"].to_numpy()
    return {
        "mean_quality": float(np.mean(quality)),
        "median_quality": float(np.median(quality)),
        "p10_quality": float(np.quantile(quality, 0.10)),
        "success_rate": float(np.mean(quality >= cfg.minimum_acceptable_quality)),
        "failure_rate": float(np.mean(quality < cfg.minimum_acceptable_quality)),
        "mean_tokens": float(np.mean(tokens)),
        "median_tokens": float(np.median(tokens)),
        "p95_tokens": float(np.quantile(tokens, 0.95)),
        "mean_agents": float(results["agents_used"].mean()),
        "mean_reflections": float(results["reflections_used"].mean()),
    }


def grid_search_static(
    train_requests: pd.DataFrame,
    cfg: SimulationConfig,
) -> tuple[pd.DataFrame, int, int, str]:
    """
    Выполняет Grid Search и выбирает самую дешевую допустимую конфигурацию.

    Допустимость:
        E[Q] >= Q_target,
        failure_rate <= risk_limit,
        E[C] <= average_budget.
    """
    records: list[dict[str, float | int | bool]] = []

    for n_agents in range(cfg.static_n_min, cfg.static_n_max + 1):
        for n_reflections in range(cfg.static_i_min, cfg.static_i_max + 1):
            run = simulate_static_configuration(
                train_requests,
                n_agents,
                n_reflections,
                cfg,
            )
            metrics = summarize_run(run, cfg)
            feasible = (
                metrics["mean_quality"] >= cfg.quality_target_train
                and metrics["failure_rate"] <= cfg.maximum_failure_rate_train
                and metrics["mean_tokens"] <= cfg.average_token_budget_static
            )
            fallback_utility = (
                metrics["mean_quality"]
                - cfg.fallback_cost_weight_per_1k_tokens
                * (metrics["mean_tokens"] / 1000.0)
            )
            records.append(
                {
                    "N": n_agents,
                    "I": n_reflections,
                    **metrics,
                    "feasible": feasible,
                    "fallback_utility": fallback_utility,
                }
            )

    grid = pd.DataFrame(records)

    feasible_grid = grid[grid["feasible"]].copy()
    if not feasible_grid.empty:
        selected = feasible_grid.sort_values(
            by=["mean_tokens", "mean_quality", "failure_rate"],
            ascending=[True, False, True],
        ).iloc[0]
        reason = (
            "Минимальная средняя стоимость среди конфигураций, "
            "удовлетворяющих ограничениям качества, риска и бюджета."
        )
    else:
        selected = grid.sort_values(
            by=["fallback_utility", "mean_quality"],
            ascending=[False, False],
        ).iloc[0]
        reason = (
            "Ни одна конфигурация не выполнила все ограничения; "
            "использован максимум fallback utility."
        )

    return grid, int(selected["N"]), int(selected["I"]), reason


def apply_step_scalar(
    request: pd.Series,
    q: float,
    cost: float,
    agent_count: int,
    reflection_count: int,
    action: Action,
    cfg: SimulationConfig,
) -> tuple[float, float]:
    """Стохастически применяет один action к одному запросу."""
    request_id = np.array([int(request["request_id"])])
    z = float(request["z"])
    total_steps_before = agent_count + reflection_count

    if action == "agent":
        action_index = agent_count + 1
        next_agents = agent_count + 1
        next_reflections = reflection_count
        affinity = float(request["agent_affinity"])
        quality_channel = 10
        cost_channel = 20
    else:
        action_index = reflection_count + 1
        next_agents = agent_count
        next_reflections = reflection_count + 1
        affinity = float(request["reflection_affinity"])
        quality_channel = 11
        cost_channel = 21

    expected_gain = float(
        predicted_increment(
            q=q,
            z=z,
            action=action,
            action_index=action_index,
            next_agent_count=next_agents,
            next_reflection_count=next_reflections,
            cfg=cfg,
            affinity=affinity,
        )
    )
    q_noise = deterministic_normal(
        request_id,
        step=action_index,
        channel=quality_channel,
        seed=cfg.random_seed,
    )[0]
    q_new = float(
        np.clip(
            q + expected_gain + cfg.step_quality_noise_std * q_noise,
            0.0,
            1.0,
        )
    )

    expected_cost = float(
        predicted_step_cost(
            z=z,
            action=action,
            total_steps_before=total_steps_before,
            cfg=cfg,
        )
    )
    c_noise = deterministic_normal(
        request_id,
        step=action_index,
        channel=cost_channel,
        seed=cfg.random_seed,
    )[0]
    cost_multiplier = float(
        np.clip(1.0 + cfg.cost_noise_std * c_noise, 0.72, 1.35)
    )
    cost_new = cost + expected_cost * cost_multiplier
    return q_new, cost_new


def estimate_action_value(
    q_observed: float,
    z_estimated: float,
    action: Action,
    agent_count: int,
    reflection_count: int,
    cfg: SimulationConfig,
) -> tuple[float, float, float]:
    """
    Возвращает:
        conservative_gain,
        expected_cost,
        gain_per_1k_tokens.
    """
    if action == "agent":
        action_index = agent_count + 1
        next_agents = agent_count + 1
        next_reflections = reflection_count
        uncertainty_penalty = cfg.uncertainty_penalty_agent
    else:
        action_index = reflection_count + 1
        next_agents = agent_count
        next_reflections = reflection_count + 1
        uncertainty_penalty = cfg.uncertainty_penalty_reflection

    expected_gain = float(
        predicted_increment(
            q=q_observed,
            z=z_estimated,
            action=action,
            action_index=action_index,
            next_agent_count=next_agents,
            next_reflection_count=next_reflections,
            cfg=cfg,
            affinity=1.0,
        )
    )
    conservative_gain = expected_gain - uncertainty_penalty
    expected_cost = float(
        predicted_step_cost(
            z=z_estimated,
            action=action,
            total_steps_before=agent_count + reflection_count,
            cfg=cfg,
        )
    )
    ratio = conservative_gain / max(expected_cost / 1000.0, 1e-9)
    return conservative_gain, expected_cost, ratio


def simulate_dynamic_request(
    request: pd.Series,
    cfg: SimulationConfig,
) -> dict[str, float | int | str]:
    """Симулирует одну адаптивную траекторию Agentic Computation Graph."""
    request_frame = request.to_frame().T
    q = float(initial_quality(request_frame, cfg)[0])
    cost = 0.0
    agents = 0
    reflections = 0
    stop_reason = "unknown"

    z_true = float(request["z"])
    z_estimated = float(
        np.clip(
            z_true
            + cfg.routing_difficulty_noise_std * float(request["router_noise"]),
            0.0,
            1.0,
        )
    )

    decision_index = 0
    while True:
        total_steps = agents + reflections

        verifier_noise = deterministic_normal(
            np.array([int(request["request_id"])]),
            step=decision_index + 1,
            channel=31,
            seed=cfg.random_seed,
        )[0]
        q_observed = float(
            np.clip(
                q
                + cfg.verifier_noise_std * verifier_noise
                + 0.006 * float(request["verifier_bias"]),
                0.0,
                1.0,
            )
        )

        # Минимально необходим хотя бы один агентный вызов.
        if agents == 0:
            candidate_actions: list[Action] = ["agent"]
        else:
            candidate_actions = []
            if agents < cfg.dynamic_n_max:
                candidate_actions.append("agent")
            if reflections < cfg.dynamic_i_max:
                candidate_actions.append("reflection")

        if not candidate_actions:
            stop_reason = "trajectory_limits"
            break

        evaluations: list[tuple[Action, float, float, float]] = []
        for action in candidate_actions:
            gain, expected_cost, ratio = estimate_action_value(
                q_observed=q_observed,
                z_estimated=z_estimated,
                action=action,
                agent_count=agents,
                reflection_count=reflections,
                cfg=cfg,
            )
            evaluations.append((action, gain, expected_cost, ratio))

        # Сначала выбирается действие с максимальной маржинальной отдачей.
        best_action, best_gain, best_expected_cost, best_ratio = max(
            evaluations,
            key=lambda item: item[3],
        )

        if cost + best_expected_cost > cfg.per_request_token_budget_dynamic:
            stop_reason = "token_budget"
            break

        if agents >= 1 and q_observed >= cfg.dynamic_stop_quality:
            stop_reason = "quality_threshold"
            break

        if (
            total_steps >= cfg.minimum_steps_before_marginal_stop
            and (
                best_ratio < cfg.lambda_quality_per_1k_tokens
                or best_gain < cfg.minimum_predicted_gain
            )
        ):
            stop_reason = "marginal_utility"
            break

        q, cost = apply_step_scalar(
            request=request,
            q=q,
            cost=cost,
            agent_count=agents,
            reflection_count=reflections,
            action=best_action,
            cfg=cfg,
        )
        if best_action == "agent":
            agents += 1
        else:
            reflections += 1
        decision_index += 1

        # Защита от логической ошибки в пользовательских параметрах.
        if decision_index > cfg.dynamic_n_max + cfg.dynamic_i_max + 2:
            stop_reason = "safety_guard"
            break

    return {
        "request_id": int(request["request_id"]),
        "complexity_class": str(request["complexity_class"]),
        "z": z_true,
        "z_estimated": z_estimated,
        "quality": q,
        "tokens": cost,
        "agents_used": agents,
        "reflections_used": reflections,
        "steps_used": agents + reflections,
        "stop_reason": stop_reason,
    }


def simulate_dynamic_policy(
    requests: pd.DataFrame,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    """Применяет динамическую policy ко всем запросам."""
    records = [
        simulate_dynamic_request(row, cfg)
        for _, row in requests.iterrows()
    ]
    return pd.DataFrame(records)


def pareto_frontier(grid: pd.DataFrame) -> pd.Series:
    """Метка Парето-оптимальности: меньше cost и больше quality."""
    is_pareto = np.ones(len(grid), dtype=bool)
    costs = grid["mean_tokens"].to_numpy()
    qualities = grid["mean_quality"].to_numpy()

    for i in range(len(grid)):
        dominated = (
            (costs <= costs[i])
            & (qualities >= qualities[i])
            & ((costs < costs[i]) | (qualities > qualities[i]))
        )
        if np.any(dominated):
            is_pareto[i] = False
    return pd.Series(is_pareto, index=grid.index)


def build_comparison_summary(
    static_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    static_metrics = summarize_run(static_results, cfg)
    dynamic_metrics = summarize_run(dynamic_results, cfg)

    summary = pd.DataFrame(
        [static_metrics, dynamic_metrics],
        index=["Model A — static", "Model B — dynamic"],
    )
    summary.index.name = "model"
    return summary.reset_index()


def build_complexity_summary(
    static_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, frame in [
        ("Model A — static", static_results),
        ("Model B — dynamic", dynamic_results),
    ]:
        for complexity, group in frame.groupby("complexity_class", observed=True):
            metrics = summarize_run(group, cfg)
            rows.append(
                {
                    "model": model_name,
                    "complexity_class": complexity,
                    **metrics,
                }
            )
    result = pd.DataFrame(rows)
    order = pd.CategoricalDtype(
        categories=["simple", "medium", "hard"],
        ordered=True,
    )
    result["complexity_class"] = result["complexity_class"].astype(order)
    return result.sort_values(["complexity_class", "model"]).reset_index(drop=True)


def plot_static_tradeoff(
    grid: pd.DataFrame,
    selected_n: int,
    selected_i: int,
    output_path: Path,
) -> None:
    """Trade-off Cost vs Quality для статической сетки."""
    fig, ax = plt.subplots(figsize=(10, 6))
    marker_sizes = 40 + 25 * grid["I"].to_numpy()
    ax.scatter(
        grid["mean_tokens"],
        grid["mean_quality"],
        s=marker_sizes,
        alpha=0.75,
        label="Grid configurations",
    )

    pareto = grid[grid["is_pareto"]].sort_values("mean_tokens")
    ax.plot(
        pareto["mean_tokens"],
        pareto["mean_quality"],
        marker="o",
        linewidth=1.5,
        label="Pareto frontier",
    )

    selected = grid[(grid["N"] == selected_n) & (grid["I"] == selected_i)].iloc[0]
    ax.scatter(
        [selected["mean_tokens"]],
        [selected["mean_quality"]],
        marker="*",
        s=260,
        label=f"Selected N={selected_n}, I={selected_i}",
    )

    for _, row in grid.iterrows():
        ax.annotate(
            f"{int(row['N'])},{int(row['I'])}",
            (row["mean_tokens"], row["mean_quality"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_title("Model A: static grid trade-off")
    ax.set_xlabel("Mean token cost")
    ax.set_ylabel("Mean final quality")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quality_boxplot(
    static_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сравнительный boxplot качества."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [
            static_results["quality"].to_numpy(),
            dynamic_results["quality"].to_numpy(),
        ],
        showmeans=True,
    )
    ax.set_xticklabels(["Model A\nstatic", "Model B\ndynamic"])
    ax.set_title("Final quality distribution")
    ax.set_ylabel("Quality Q")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cost_boxplot(
    static_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сравнительный boxplot токенных затрат."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [
            static_results["tokens"].to_numpy(),
            dynamic_results["tokens"].to_numpy(),
        ],
        showmeans=True,
    )
    ax.set_xticklabels(["Model A\nstatic", "Model B\ndynamic"])
    ax.set_title("Token cost distribution")
    ax.set_ylabel("Tokens")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dynamic_agent_histogram(
    dynamic_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Гистограмма фактического N для динамической стратегии."""
    fig, ax = plt.subplots(figsize=(9, 6))
    min_n = int(dynamic_results["agents_used"].min())
    max_n = int(dynamic_results["agents_used"].max())
    bins = np.arange(min_n - 0.5, max_n + 1.5, 1.0)
    ax.hist(dynamic_results["agents_used"], bins=bins, rwidth=0.85)
    ax.set_title("Model B: adaptive number of agents")
    ax.set_xlabel("Agents actually used, N(τ)")
    ax.set_ylabel("Number of requests")
    ax.set_xticks(np.arange(min_n, max_n + 1))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_parameter_table(cfg: SimulationConfig) -> str:
    """Формирует сравнительную таблицу входных параметров."""
    rows = [
        ("Размер train/test", f"{cfg.train_requests}/{cfg.test_requests}", "общий", "Одинаковые выборки для честного сравнения"),
        ("Распределение сложности", f"{cfg.difficulty_prob_simple:.2f}/{cfg.difficulty_prob_medium:.2f}/{cfg.difficulty_prob_hard:.2f}", "общий", "simple/medium/hard"),
        ("Стоимость вызова агента", f"{cfg.base_agent_tokens:.0f}", "общий", "Базовая стоимость до роста контекста"),
        ("Стоимость рефлексии", f"{cfg.base_reflection_tokens:.0f}", "общий", "Базовая стоимость до роста контекста"),
        ("Рост контекста", f"{cfg.context_growth_per_step:.3f}", "общий", "Увеличивает стоимость следующих шагов"),
        ("Штраф hallucination/overthinking", f"{cfg.hallucination_penalty:.3f}", "общий", "Включается после task-specific saturation point"),
        ("Сетка N", f"{cfg.static_n_min}…{cfg.static_n_max}", "Model A", "Фиксированное число агентов"),
        ("Сетка I", f"{cfg.static_i_min}…{cfg.static_i_max}", "Model A", "Фиксированное число рефлексий"),
        ("Средний budget", f"{cfg.average_token_budget_static:.0f}", "Model A", "Ограничение при выборе статического оптимума"),
        ("Максимум N(τ)", str(cfg.dynamic_n_max), "Model B", "Предохранитель динамической траектории"),
        ("Максимум I(τ)", str(cfg.dynamic_i_max), "Model B", "Предохранитель динамической траектории"),
        ("Per-request budget", f"{cfg.per_request_token_budget_dynamic:.0f}", "Model B", "Жесткий предел на запрос"),
        ("Порог λ", f"{cfg.lambda_quality_per_1k_tokens:.3f}", "Model B", "Минимальный ожидаемый ΔQ на 1000 токенов"),
        ("Stop quality", f"{cfg.dynamic_stop_quality:.3f}", "Model B", "Остановка при достаточном verifier-score"),
        ("Шум оценки сложности", f"{cfg.routing_difficulty_noise_std:.3f}", "Model B", "Имитирует ошибку роутера ẑ(x)"),
        ("Шум верификатора", f"{cfg.verifier_noise_std:.3f}", "Model B", "Имитирует несовершенную online-оценку"),
    ]
    df = pd.DataFrame(rows, columns=["Параметр", "Значение", "Область", "Назначение"])
    return df.to_markdown(index=False)


def dataframe_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """
    pandas.to_markdown требует tabulate.
    При его отсутствии строится простая Markdown-таблица вручную.
    """
    try:
        return df.to_markdown(index=index, floatfmt=".4f")
    except ImportError:
        working = df.copy()
        if index:
            working = working.reset_index()
        headers = [str(column) for column in working.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in working.itertuples(index=False):
            formatted = []
            for value in row:
                if isinstance(value, float):
                    formatted.append(f"{value:.4f}")
                else:
                    formatted.append(str(value))
            lines.append("| " + " | ".join(formatted) + " |")
        return "\n".join(lines)


def build_analytical_report(
    cfg: SimulationConfig,
    grid: pd.DataFrame,
    selected_n: int,
    selected_i: int,
    selection_reason: str,
    summary: pd.DataFrame,
    by_complexity: pd.DataFrame,
) -> str:
    """Генерирует аналитический отчет по структуре задания."""
    static_row = summary[summary["model"] == "Model A — static"].iloc[0]
    dynamic_row = summary[summary["model"] == "Model B — dynamic"].iloc[0]

    cost_saving = 100.0 * (
        1.0 - dynamic_row["mean_tokens"] / static_row["mean_tokens"]
    )
    quality_delta = dynamic_row["mean_quality"] - static_row["mean_quality"]
    success_delta_pp = 100.0 * (
        dynamic_row["success_rate"] - static_row["success_rate"]
    )

    selected_grid_row = grid[
        (grid["N"] == selected_n) & (grid["I"] == selected_i)
    ].iloc[0]

    compact_summary = summary[
        [
            "model",
            "mean_quality",
            "median_quality",
            "p10_quality",
            "success_rate",
            "mean_tokens",
            "p95_tokens",
            "mean_agents",
            "mean_reflections",
        ]
    ].copy()

    compact_complexity = by_complexity[
        [
            "model",
            "complexity_class",
            "mean_quality",
            "success_rate",
            "mean_tokens",
            "mean_agents",
            "mean_reflections",
        ]
    ].copy()

    parameter_rows = [
        ("Размер train/test", f"{cfg.train_requests}/{cfg.test_requests}", "общий", "Одинаковые выборки для честного сравнения"),
        ("Распределение сложности", f"{cfg.difficulty_prob_simple:.2f}/{cfg.difficulty_prob_medium:.2f}/{cfg.difficulty_prob_hard:.2f}", "общий", "simple/medium/hard"),
        ("Стоимость вызова агента", f"{cfg.base_agent_tokens:.0f}", "общий", "До роста контекста"),
        ("Стоимость рефлексии", f"{cfg.base_reflection_tokens:.0f}", "общий", "До роста контекста"),
        ("Штраф overthinking", f"{cfg.hallucination_penalty:.3f}", "общий", "После task-specific saturation point"),
        ("Сетка N", f"{cfg.static_n_min}…{cfg.static_n_max}", "Model A", "Фиксированное N"),
        ("Сетка I", f"{cfg.static_i_min}…{cfg.static_i_max}", "Model A", "Фиксированное I"),
        ("Средний token budget", f"{cfg.average_token_budget_static:.0f}", "Model A", "Ограничение Grid Search"),
        ("Per-request budget", f"{cfg.per_request_token_budget_dynamic:.0f}", "Model B", "Жесткий бюджет траектории"),
        ("Порог λ", f"{cfg.lambda_quality_per_1k_tokens:.3f}", "Model B", "Минимальный ΔQ/1000 токенов"),
        ("Stop quality", f"{cfg.dynamic_stop_quality:.3f}", "Model B", "Verifier-driven stop"),
        ("Шум роутера ẑ", f"{cfg.routing_difficulty_noise_std:.3f}", "Model B", "Ошибка оценки сложности"),
    ]
    parameter_df = pd.DataFrame(
        parameter_rows,
        columns=["Параметр", "Значение", "Область", "Назначение"],
    )

    return rf"""# Сравнение статической и динамической оптимизации мультиагентных LLM-систем

## 1. Формальная постановка задачи

### Модель A: статическая оптимизация

Для обучающей выборки запросов \(X_{{train}}\) перебирается конечная сетка
\((N,I)\), где \(N\) — фиксированное число агентов, а \(I\) — фиксированное
число циклов рефлексии. Для каждого запроса применяется одна и та же
конфигурация:

\[
(N^*, I^*) =
\arg\min_{{N,I}} \mathbb{{E}}[C(N,I)]
\]

при ограничениях

\[
\mathbb{{E}}[Q(N,I)] \ge Q_{{target}}, \quad
P(Q(N,I)<Q_{{min}}) \le \alpha, \quad
\mathbb{{E}}[C(N,I)] \le B.
\]

Выбрано: **N={selected_n}, I={selected_i}**.

Причина выбора: {selection_reason}

Train-оценка выбранной точки:
\(Q={selected_grid_row['mean_quality']:.4f}\),
\(C={selected_grid_row['mean_tokens']:.1f}\) токенов,
failure rate \(={selected_grid_row['failure_rate']:.4f}\).

### Модель B: динамическая политика

Для запроса \(x\) строится траектория

\[
\tau=(s_0,a_0,s_1,a_1,\ldots,s_K),
\]

где состояние содержит наблюдаемое качество, оценку сложности
\(\hat z(x)\), число уже использованных агентов и рефлексий и остаток
токенного бюджета. Действие выбирается из
`agent`, `reflection`, `STOP`.

На каждом шаге policy сравнивает прогнозируемые действия по критерию

\[
\rho(a\mid s_t)=
\frac{{\widehat{{\Delta Q}}(a\mid s_t)}}
{{\widehat{{\Delta C}}(a\mid s_t)/1000}}.
\]

Выполняется действие с максимальным \(\rho\). STOP выбирается, если:
1. достигнут verifier-порог качества;
2. \(\rho < \lambda\);
3. прогнозируемый \(\Delta Q\) слишком мал;
4. исчерпан бюджет или лимит траектории.

### Убывающая отдача и скрытая сложность

Положительная часть прироста имеет экспоненциальное затухание:

\[
\Delta Q_k^+ \propto
(1-Q_k)\exp(-d(k-1))g(z).
\]

Task-specific точка насыщения задается как

\[
e^*(z)=e_0+e_1z,
\]

а при \(e>e^*(z)\) вводится штраф деградации

\[
P_{{overthink}} =
h(1.18-0.30z)
\max(0,e-e^*(z))^p.
\]

Поэтому простые запросы раньше достигают насыщения, а сложные обычно
требуют более длинной траектории.

## 2. Сравнительная таблица параметров

{dataframe_to_markdown_safe(parameter_df)}

## 3. Реализация на Python

Скрипт реализует:
- генерацию стратифицированного потока simple/medium/hard;
- парно-сопоставимый стохастический симулятор качества и стоимости;
- train-only Grid Search для Модели A;
- online routing и Early Stopping для Модели B;
- оценку обеих стратегий на одной независимой test-выборке;
- экспорт trace-level результатов в CSV;
- воспроизводимость через фиксированный random seed.

## 4. Визуализация результатов

Созданы файлы:
- `01_static_tradeoff.png` — Cost–Quality и Парето-фронт сетки Модели A;
- `02_quality_boxplot.png` — распределение итогового качества;
- `03_cost_boxplot.png` — распределение токенных затрат;
- `04_dynamic_agents_histogram.png` — фактическое \(N(\tau)\) Модели B.

## 5. Анализ и пояснение

### Итоговые метрики

{dataframe_to_markdown_safe(compact_summary)}

### Метрики по классам сложности

{dataframe_to_markdown_safe(compact_complexity)}

При текущей конфигурации динамическая policy изменила средний расход
токенов на **{cost_saving:.2f}%** относительно статической
конфигурации. Разность среднего качества
\(Q_B-Q_A={quality_delta:+.4f}\), а изменение success rate составляет
{success_delta_pp:+.2f} процентного пункта.

Экономия возникает не из-за бесплатного дополнительного знания, а из-за
условного выделения вычислительного бюджета. Статическая модель обязана
оплачивать N={selected_n} и I={selected_i} для каждого запроса. Динамическая
модель завершает простые запросы после достижения достаточного verifier-score,
но продолжает сложные запросы, пока прогнозируемая предельная полезность
оправдывает следующий вызов.

Алгоритмические преимущества динамического подхода:
- \(N(\tau)\) и \(I(\tau)\) зависят от запроса, а не являются глобальными
  константами;
- учитывается остаток бюджета на каждом decision point;
- уменьшается риск overthinking после локальной точки насыщения;
- можно оптимизировать tail-risk и SLA отдельно по классам сложности;
- trace-логи позволяют впоследствии заменить эвристику обучаемым router,
  contextual bandit или offline-RL policy.

Ограничение симуляции: это синтетическая среда. Численные результаты не
следует переносить в production без калибровки функций прироста, стоимости,
ошибки verifier-а и распределения сложности на реальных трассировках.
"""


def save_outputs(
    output_dir: Path,
    cfg: SimulationConfig,
    grid: pd.DataFrame,
    static_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
    summary: pd.DataFrame,
    by_complexity: pd.DataFrame,
    report: str,
    selected_n: int,
    selected_i: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    grid.to_csv(output_dir / "static_grid_train.csv", index=False)
    static_results.to_csv(output_dir / "static_test_results.csv", index=False)
    dynamic_results.to_csv(output_dir / "dynamic_test_results.csv", index=False)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    by_complexity.to_csv(output_dir / "comparison_by_complexity.csv", index=False)

    pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in asdict(cfg).items()]
    ).to_csv(output_dir / "simulation_parameters.csv", index=False)

    (output_dir / "analytical_report.md").write_text(report, encoding="utf-8")

    plot_static_tradeoff(
        grid=grid,
        selected_n=selected_n,
        selected_i=selected_i,
        output_path=output_dir / "01_static_tradeoff.png",
    )
    plot_quality_boxplot(
        static_results,
        dynamic_results,
        output_dir / "02_quality_boxplot.png",
    )
    plot_cost_boxplot(
        static_results,
        dynamic_results,
        output_dir / "03_cost_boxplot.png",
    )
    plot_dynamic_agent_histogram(
        dynamic_results,
        output_dir / "04_dynamic_agents_histogram.png",
    )


def print_console_summary(
    output_dir: Path,
    selected_n: int,
    selected_i: int,
    summary: pd.DataFrame,
) -> None:
    """Компактный вывод в консоль."""
    print("\n=== MULTI-AGENT LLM OPTIMIZATION SIMULATION ===")
    print(f"Selected static configuration: N={selected_n}, I={selected_i}")
    print()
    display_columns = [
        "model",
        "mean_quality",
        "success_rate",
        "mean_tokens",
        "p95_tokens",
        "mean_agents",
        "mean_reflections",
    ]
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs: {output_dir.resolve()}")
    print("Analytical report: analytical_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static vs dynamic optimization of multi-agent LLM workflows."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("multiagent_simulation_outputs"),
        help="Каталог для CSV, PNG и analytical_report.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig()
    validate_config(cfg)

    train_requests = generate_requests(
        n_requests=cfg.train_requests,
        cfg=cfg,
        seed_offset=1,
    )
    test_requests = generate_requests(
        n_requests=cfg.test_requests,
        cfg=cfg,
        seed_offset=2,
    )

    grid, selected_n, selected_i, selection_reason = grid_search_static(
        train_requests,
        cfg,
    )
    grid["is_pareto"] = pareto_frontier(grid)

    static_test_results = simulate_static_configuration(
        test_requests,
        n_agents=selected_n,
        n_reflections=selected_i,
        cfg=cfg,
    )
    dynamic_test_results = simulate_dynamic_policy(
        test_requests,
        cfg,
    )

    summary = build_comparison_summary(
        static_test_results,
        dynamic_test_results,
        cfg,
    )
    by_complexity = build_complexity_summary(
        static_test_results,
        dynamic_test_results,
        cfg,
    )

    report = build_analytical_report(
        cfg=cfg,
        grid=grid,
        selected_n=selected_n,
        selected_i=selected_i,
        selection_reason=selection_reason,
        summary=summary,
        by_complexity=by_complexity,
    )

    save_outputs(
        output_dir=args.output_dir,
        cfg=cfg,
        grid=grid,
        static_results=static_test_results,
        dynamic_results=dynamic_test_results,
        summary=summary,
        by_complexity=by_complexity,
        report=report,
        selected_n=selected_n,
        selected_i=selected_i,
    )
    print_console_summary(
        args.output_dir,
        selected_n,
        selected_i,
        summary,
    )


if __name__ == "__main__":
    main()
