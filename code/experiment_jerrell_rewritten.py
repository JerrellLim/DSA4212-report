#!/usr/bin/env python3
"""Reproducible comparison of CMA-ES, Gaussian-process BO, and random search.

Four experiment modes are provided:
  coco       COCO/BBOB benchmark problems; single-worker objective calls only.
  synthetic  shifted analytic functions; supports worker sweeps and simulated latency.
  noisy      synthetic functions with Gaussian/heavy-tailed noise and true-value scoring.
  highdim    a wall-clock-limited comparison of TuRBO-1, CMA-ES, and random search.

TuRBO follows the paper's local-GP trust-region strategy and Thompson sampling.
To fit a two-hour budget on CPU, this implementation uses a random-feature
approximation to sample from the GP posterior, a capped candidate set, and a
capped local training set; the control parameters match the paper's TuRBO-1 defaults.

The default BO backend is a lightweight scikit-learn GP with sampled expected
improvement. Use --bo-backend optuna for Optuna GPSampler's asynchronous parallel
trial support, especially in the worker-sweep experiment. The `noisy` mode tests
Gaussian and heavy-tailed observation noise with noise-free recommendation scoring.

Install dependencies with:
    python -m pip install numpy scipy scikit-learn cma optuna torch coco-experiment cocopp

Examples are in README_experiments.md. All budgets count objective evaluations,
including the initial point. No unbudgeted final re-evaluation is performed.
"""
from __future__ import annotations

# Avoid common BLAS/OpenMP oversubscription with GP-based BO. Set before NumPy
# and Optuna/PyTorch are imported. Users can override these environment values.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import math
import os.path
import re
import sys
import time
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Sequence

import numpy as np

try:
    import cma
except ImportError:
    cma = None

try:
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.trial import create_trial
except ImportError:
    optuna = None
    FloatDistribution = None
    create_trial = None


ALGORITHMS = ("cma", "bo", "random", "turbo")
SYNTHETIC_FUNCTIONS = ("sphere", "rastrigin", "ackley", "rosenbrock")
# Report threshold-sensitivity in addition to the user's configured --target.
# This avoids making conclusions depend on one arbitrary absolute cutoff.
DIAGNOSTIC_TARGETS = (1e-8, 1e-4, 1e-2, 1e-1, 1.0, 5.0, 10.0)


def target_column_suffix(target: float) -> str:
    """Stable, CSV-friendly suffix for a target threshold."""
    return f"{target:.0e}" if target < 0.01 else f"{target:g}"


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list such as '2,3,5,10'."""
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Expected one or more positive integers.")
    return values


def parse_positive_int_list(value: str) -> list[int]:
    return parse_int_list(value)


def parse_float_list(value: str, *, allow_zero: bool = True) -> list[float]:
    """Parse comma-separated finite, nonnegative floats."""
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric list: {value}") from exc
    if not values or any(not np.isfinite(item) or item < 0 or (not allow_zero and item == 0) for item in values):
        qualifier = "positive" if not allow_zero else "nonnegative"
        raise argparse.ArgumentTypeError(f"Expected one or more finite {qualifier} numbers.")
    return values


def parse_csv_list(value: str) -> list[str]:
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list.")
    return values


def stable_seed(*parts: object) -> int:
    """A deterministic seed independent of Python's randomized hash()."""
    data = "|".join(str(part) for part in parts).encode("utf-8")
    return zlib.crc32(data) & 0x7FFFFFFF


@dataclass
class RunResult:
    algorithm: str
    best_x: np.ndarray
    best_value: float
    evaluations: int
    wall_seconds: float
    trace: list[dict[str, float | int]]
    stop_reason: str = "budget_or_optimizer_stop"


class CountingObjective:
    """Count each true objective call and record a best-so-far trace safely."""
    def __init__(self, function: Callable[[np.ndarray], float], delay_ms: float = 0.0):
        self.function = function
        self.delay_seconds = max(0.0, float(delay_ms)) / 1000.0
        self.count = 0
        self.trace: list[dict[str, float | int]] = []
        self.best_value = math.inf
        self._lock = Lock()

    def __call__(self, x: Sequence[float]) -> float:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        value = float(self.function(np.asarray(x, dtype=float)))
        if not np.isfinite(value):
            value = float("inf")
        with self._lock:
            self.count += 1
            self.best_value = min(self.best_value, value)
            self.trace.append({
                "evaluation": self.count,
                "value": value,
                "best_so_far": float(self.best_value),
            })
        return value


def bounds_from_arrays(lower: Sequence[float], upper: Sequence[float], dim: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.size != dim or hi.size != dim:
        raise ValueError(f"Bounds must have {dim} coordinates.")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(hi <= lo):
        raise ValueError("This runner requires finite, strictly ordered lower/upper bounds.")
    return lo, hi


def evaluate_batch(
    objective: CountingObjective,
    points: Sequence[Sequence[float]],
    workers: int,
    executor: ThreadPoolExecutor | None = None,
) -> list[float]:
    if len(points) == 0:
        return []
    if workers <= 1:
        return [objective(point) for point in points]
    if executor is None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(objective, points))
    return list(executor.map(objective, points))


def run_cma(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    seed: int,
    workers: int = 1,
    delay_ms: float = 0.0,
    sigma_scale: float = 0.3,
    deadline: float | None = None,
    backend: str = "auto",
) -> RunResult:
    if backend not in ("auto", "pycma", "numpy"):
        raise ValueError("CMA backend must be auto, pycma, or numpy.")
    if backend == "numpy" or (backend == "auto" and cma is None):
        return run_cma_numpy_fallback(
            objective_fn, x0, lower, upper, budget, seed, delay_ms,
            sigma_scale, deadline,
        )
    if cma is None:
        raise RuntimeError("The pycma backend was requested but 'cma' is not installed. Install with: python -m pip install cma")
    if budget < 2:
        raise ValueError("Budget must be at least 2 evaluations.")

    started = time.perf_counter()
    objective = CountingObjective(objective_fn, delay_ms)
    best_x = np.asarray(x0, dtype=float).copy()
    best_value = objective(best_x)  # Shared, counted initial observation.
    width = upper - lower
    sigma0 = max(1e-8, float(sigma_scale) * float(np.median(width)))
    options = {
        "seed": int(seed),
        # x0 is used as the first member of the first generation, so CMA's
        # internal evaluation count matches the real number of calls.
        "maxfevals": int(budget),
        "bounds": [lower.tolist(), upper.tolist()],
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }
    es = cma.CMAEvolutionStrategy(best_x.tolist(), sigma0, options)
    stop_reason = "evaluation_budget"

    pool_context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    first_generation = True
    try:
        while objective.count < budget and not es.stop() and (deadline is None or time.perf_counter() < deadline):
            candidates = [np.asarray(x, dtype=float) for x in es.ask()]
            remaining = budget - objective.count
            if first_generation:
                # Reuse the shared, already-counted x0 observation as one member
                # of generation one rather than evaluating it twice.
                candidates[0] = best_x.copy()
                values = [best_value]
                new_points = candidates[1:1 + remaining]
                new_values = evaluate_batch(objective, new_points, workers, pool_context)
                values.extend(new_values)
                if new_values:
                    idx = int(np.argmin(new_values))
                    if new_values[idx] < best_value:
                        best_value = float(new_values[idx])
                        best_x = new_points[idx].copy()
                first_generation = False
                complete_generation = len(values) == len(candidates)
            else:
                batch = candidates[:remaining]
                values = evaluate_batch(objective, batch, workers, pool_context)
                if values:
                    idx = int(np.argmin(values))
                    if values[idx] < best_value:
                        best_value = float(values[idx])
                        best_x = batch[idx].copy()
                # CMA-ES learns from a complete generation. If the hard budget
                # allows only a partial last generation, record it but do not
                # feed an incomplete population to tell().
                complete_generation = len(batch) == len(candidates)
            if complete_generation:
                es.tell(candidates, values)
            else:
                break
        else:
            stopped = es.stop()
            if stopped:
                stop_reason = ";".join(stopped.keys()) if isinstance(stopped, dict) else str(stopped)
    finally:
        if pool_context is not None:
            pool_context.shutdown(wait=True)

    if deadline is not None and time.perf_counter() >= deadline and objective.count < budget:
        stop_reason = "wall_time_limit"
    return RunResult("cma", best_x, best_value, objective.count,
                     time.perf_counter() - started, objective.trace, stop_reason)


def run_cma_numpy_fallback(
    objective_fn: Callable[[np.ndarray], float], x0: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, budget: int, seed: int,
    delay_ms: float = 0.0, sigma_scale: float = 0.3,
    deadline: float | None = None,
) -> RunResult:
    """Compact NumPy CMA-ES fallback used only when pycma is unavailable.

    This implements the standard rank-one/rank-mu covariance and evolution-path
    updates. Install `cma` to use pycma in the full experiment.
    """
    if budget < 2:
        raise ValueError("Budget must be at least 2 evaluations.")
    started = time.perf_counter()
    objective = CountingObjective(objective_fn, delay_ms)
    lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    width = upper - lower
    mean = np.clip((np.asarray(x0, float) - lower) / width, 0.0, 1.0)
    best_x = np.asarray(x0, float).copy()
    best_value = objective(best_x)
    rng = np.random.default_rng(seed)
    dim = mean.size
    lam = 4 + int(3 * math.log(max(dim, 2)))
    mu = max(1, lam // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    cc = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
    cs = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
    sigma = max(1e-4, float(sigma_scale))
    C = np.eye(dim); B = np.eye(dim); D = np.ones(dim)
    pc = np.zeros(dim); ps = np.zeros(dim)
    eigeneval = 0
    stop_reason = "evaluation_budget"
    while objective.count < budget and (deadline is None or time.perf_counter() < deadline):
        remaining = budget - objective.count
        n_gen = min(lam, remaining)
        z = rng.normal(size=(n_gen, dim))
        y = (z * D) @ B.T
        raw = mean + sigma * y
        # Reflect at the unit-box bounds instead of piling samples on the edges.
        reflected = np.mod(raw, 2.0)
        X_unit = np.where(reflected > 1.0, 2.0 - reflected, reflected)
        X = lower + X_unit * width
        vals = evaluate_batch(objective, X, 1)
        vals_arr = np.asarray(vals, float)
        idx = int(np.argmin(vals_arr))
        if vals_arr[idx] < best_value:
            best_value = float(vals_arr[idx]); best_x = X[idx].copy()
        if n_gen < lam or len(vals_arr) < mu:
            stop_reason = "budget_partial_generation" if objective.count >= budget else "wall_time_limit"
            break
        order = np.argsort(vals_arr)
        y_sel = (X_unit[order[:mu]] - mean) / max(sigma, 1e-12)
        y_w = np.sum(weights[:, None] * y_sel, axis=0)
        old_mean = mean.copy()
        mean = np.clip(mean + sigma * y_w, 0.0, 1.0)
        invsqrtC = B @ np.diag(1.0 / np.maximum(D, 1e-12)) @ B.T
        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mu_eff) * (invsqrtC @ y_w)
        norm_ps = np.linalg.norm(ps)
        denom = math.sqrt(1.0 - (1.0 - cs) ** (2.0 * (objective.count / lam + 1.0)))
        hsig = float(norm_ps / max(denom, 1e-12) / chi_n < 1.4 + 2.0 / (dim + 1.0))
        pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mu_eff) * y_w
        rank_mu = np.zeros_like(C)
        for wi, yi in zip(weights, y_sel):
            rank_mu += wi * np.outer(yi, yi)
        C = ((1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
             + cmu * rank_mu)
        sigma *= math.exp((cs / damps) * (norm_ps / chi_n - 1.0))
        sigma = float(np.clip(sigma, 1e-5, 1.0))
        if objective.count - eigeneval > max(1, lam / max(c1 + cmu, 1e-12) / dim / 10.0):
            C = (C + C.T) / 2.0
            eigvals, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-20))
            eigeneval = objective.count
    if deadline is not None and time.perf_counter() >= deadline and objective.count < budget:
        stop_reason = "wall_time_limit"
    return RunResult("cma", best_x, best_value, objective.count,
                     time.perf_counter() - started, objective.trace, stop_reason)


def run_bo_optuna(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    seed: int,
    workers: int = 1,
    delay_ms: float = 0.0,
    bo_startup: int = 10,
) -> RunResult:
    """Optuna's GP-based Bayesian optimizer (Matérn-5/2 GP and EI acquisition)."""
    if optuna is None:
        raise RuntimeError("BO requires Optuna. Install it with: python -m pip install 'optuna>=4.8' torch")
    if not hasattr(optuna.samplers, "GPSampler"):
        raise RuntimeError("This Optuna version has no GPSampler; install a recent Optuna with: python -m pip install -U optuna torch")
    if budget < 2:
        raise ValueError("Budget must be at least 2 evaluations.")

    # Import PyTorch before starting the per-run timer so first-use import cost
    # does not distort the first BO timing relative to later runs.
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits setting inter-op threads before parallel work;
            # an earlier BO run may already have initialized that pool.
            pass
    except ImportError as exc:
        raise RuntimeError(
            "Optuna GPSampler requires PyTorch. Install a CPU build from "
            "https://pytorch.org/get-started/locally/."
        ) from exc

    started = time.perf_counter()
    objective = CountingObjective(objective_fn, delay_ms)
    x0 = np.asarray(x0, dtype=float).copy()
    initial_value = objective(x0)  # Included in the total budget.
    names = [f"x{i}" for i in range(x0.size)]
    distributions = {name: FloatDistribution(float(lo), float(hi))
                     for name, lo, hi in zip(names, lower, upper)}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.GPSampler(
        seed=int(seed),
        n_startup_trials=max(2, int(bo_startup)),
        deterministic_objective=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # Give BO the same initial observation that CMA-ES gets as its initial mean.
    study.add_trial(create_trial(
        params={name: float(value) for name, value in zip(names, x0)},
        distributions=distributions,
        value=float(initial_value),
    ))

    # Space-filling Sobol startup points improve early coverage on rugged
    # landscapes. They count against the same objective-evaluation budget.
    startup_total = min(max(2, int(bo_startup)), int(budget))
    startup_points_needed = min(startup_total - 1, budget - 1)
    if startup_points_needed > 0:
        from scipy.stats import qmc
        power = int(math.ceil(math.log2(startup_points_needed)))
        unit_points = qmc.Sobol(d=x0.size, scramble=True, seed=int(seed)).random_base2(power)
        startup_points = lower + unit_points[:startup_points_needed] * (upper - lower)
        startup_values = evaluate_batch(objective, startup_points, workers)
        for point, value in zip(startup_points, startup_values):
            study.add_trial(create_trial(
                params={name: float(item) for name, item in zip(names, point)},
                distributions=distributions,
                value=float(value),
            ))

    def trial_objective(trial) -> float:
        x = np.array([trial.suggest_float(name, float(lo), float(hi))
                      for name, lo, hi in zip(names, lower, upper)], dtype=float)
        return objective(x)

    remaining_trials = budget - 1 - startup_points_needed
    try:
        if remaining_trials > 0:
            study.optimize(
                trial_objective,
                n_trials=int(remaining_trials),
                n_jobs=max(1, int(workers)),
                show_progress_bar=False,
                gc_after_trial=False,
            )
    except ImportError as exc:
        raise RuntimeError(
            "Optuna GPSampler needs PyTorch as well as SciPy. Install CPU PyTorch using "
            "the instructions at https://pytorch.org/get-started/locally/, then install optuna."
        ) from exc

    best_trial = study.best_trial
    best_x = np.array([best_trial.params[name] for name in names], dtype=float)
    return RunResult("bo", best_x, float(best_trial.value), objective.count,
                     time.perf_counter() - started, objective.trace, "evaluation_budget")


def run_bo_sklearn(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    seed: int,
    workers: int = 1,
    delay_ms: float = 0.0,
    bo_startup: int = 10,
    observation_noise_std: float = 0.0,
) -> RunResult:
    """Lightweight sequential GP-BO using scikit-learn and sampled expected improvement.

    The input domain is scaled to [0, 1]^d. Hyperparameters are re-optimised
    every five model-based suggestions and held fixed in between, avoiding a
    full marginal-likelihood optimisation at every iteration. EI is maximised
    on a Sobol candidate pool. For asynchronous multi-worker BO, use backend
    'optuna' instead; this implementation is intentionally sequential.
    """
    if workers != 1:
        raise RuntimeError(
            "The sklearn BO backend is sequential. Use --bo-backend optuna for the "
            "parallel-worker experiment, which uses asynchronous trials."
        )
    if budget < 2:
        raise ValueError("Budget must be at least 2 evaluations.")
    try:
        from scipy.stats import norm, qmc
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:
        raise RuntimeError(
            "The fast sklearn BO backend requires scikit-learn. Install it with: "
            "python -m pip install scikit-learn"
        ) from exc

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    if np.any(width <= 0):
        raise ValueError("Every upper bound must be greater than its lower bound.")

    started = time.perf_counter()
    objective = CountingObjective(objective_fn, delay_ms)
    x0 = np.clip(np.asarray(x0, dtype=float), lower, upper)
    x_values: list[np.ndarray] = [x0.copy()]
    y_values: list[float] = [objective(x0)]

    # The common initial point is the first observation. Additional startup
    # observations use a Sobol design and count against the same true-call budget.
    startup_total = min(max(2, int(bo_startup)), int(budget))
    startup_points_needed = min(startup_total - 1, budget - 1)
    if startup_points_needed:
        power = int(math.ceil(math.log2(startup_points_needed)))
        sobol = qmc.Sobol(d=x0.size, scramble=True, seed=int(seed))
        unit_points = sobol.random_base2(power)[:startup_points_needed]
        startup_points = lower + unit_points * width
        startup_values = evaluate_batch(objective, startup_points, workers=1)
        x_values.extend(np.asarray(point, dtype=float) for point in startup_points)
        y_values.extend(float(value) for value in startup_values)

    # Length scales live in the unit-cube coordinate system. The bounds allow
    # shorter scales for rugged functions such as Rastrigin, while EI is
    # searched over a bounded Sobol pool rather than with a costly local solver.
    dimension = x0.size
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(dimension, 0.2),
        length_scale_bounds=(0.03, 5.0),
        nu=2.5,
    )
    candidate_count = max(2048, min(8192, 256 * dimension))
    model_iteration = 0

    while len(y_values) < budget:
        X = (np.asarray(x_values) - lower) / width
        y = np.asarray(y_values, dtype=float)
        # Tune kernel hyperparameters on the first model fit and then every
        # five BO steps. Reusing the last fitted kernel keeps routine fits fast.
        tune_kernel = model_iteration == 0 or model_iteration % 5 == 0
        # When running noisy experiments, pass the known noise variance to the
        # GP likelihood. Since normalize_y=True rescales targets, express the
        # variance in normalized target units. Deterministic experiments retain
        # a small numerical nugget only.
        if observation_noise_std > 0:
            target_scale = max(float(np.std(y)), 1e-12)
            gp_alpha = max(1e-10, (float(observation_noise_std) / target_scale) ** 2)
        else:
            gp_alpha = 1e-6
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=gp_alpha,
            normalize_y=True,
            optimizer="fmin_l_bfgs_b" if tune_kernel else None,
            n_restarts_optimizer=0,
            random_state=int(seed),
        )
        # Boundary warnings are common when the marginal-likelihood optimum
        # sits at a deliberately finite kernel bound. They can otherwise emit
        # hundreds of identical lines in a long benchmark; the bounds remain
        # explicit in this implementation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        kernel = gp.kernel_

        power = int(math.ceil(math.log2(candidate_count)))
        candidates = qmc.Sobol(
            d=dimension, scramble=True, seed=stable_seed(seed, "ei-candidates", model_iteration)
        ).random_base2(power)[:candidate_count]
        mean, std = gp.predict(candidates, return_std=True)
        incumbent = float(np.min(y))
        # A small, scale-aware exploration parameter for expected improvement.
        xi = 0.01 * max(float(np.std(y)), 1e-12)
        improvement = incumbent - mean - xi
        safe_std = np.maximum(std, 1e-12)
        z = improvement / safe_std
        ei = improvement * norm.cdf(z) + safe_std * norm.pdf(z)

        # Do not choose a point numerically indistinguishable from an existing
        # observation. In continuous spaces this is mostly a safeguard.
        delta = candidates[:, None, :] - X[None, :, :]
        nearest_sq = np.min(np.sum(delta * delta, axis=2), axis=1)
        ei[nearest_sq < 1e-12] = -np.inf
        next_unit = candidates[int(np.argmax(ei))]
        next_x = lower + next_unit * width
        next_y = objective(next_x)
        x_values.append(next_x)
        y_values.append(float(next_y))
        model_iteration += 1

    best_index = int(np.argmin(y_values))
    return RunResult(
        "bo", np.asarray(x_values[best_index], dtype=float), float(y_values[best_index]),
        objective.count, time.perf_counter() - started, objective.trace, "evaluation_budget",
    )


def run_bo(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    seed: int,
    workers: int = 1,
    delay_ms: float = 0.0,
    bo_startup: int = 10,
    bo_backend: str = "sklearn",
    observation_noise_std: float = 0.0,
) -> RunResult:
    """Dispatch to a fast sequential or asynchronous parallel GP-BO backend."""
    if bo_backend == "sklearn":
        return run_bo_sklearn(
            objective_fn, x0, lower, upper, budget, seed, workers, delay_ms,
            bo_startup, observation_noise_std,
        )
    if bo_backend == "optuna":
        return run_bo_optuna(
            objective_fn, x0, lower, upper, budget, seed, workers, delay_ms, bo_startup
        )
    raise ValueError(f"Unknown BO backend {bo_backend!r}; choose 'sklearn' or 'optuna'.")



def _turbo_rff_posterior_samples(gp, candidates: np.ndarray, batch_size: int,
                                seed: int, n_features: int = 128) -> np.ndarray:
    """Approximate coherent GP Thompson samples by Matheron's rule + Matérn RFF.

    Returns an array [batch_size, n_candidates]. RFF draws approximate a
    Matérn-5/2 GP prior; conditioning them on the fitted GP data gives posterior
    function draws without forming an O(n_candidates^3) covariance matrix.
    """
    from scipy.linalg import solve_triangular

    X_train = np.asarray(gp.X_train_, dtype=float)
    y_train = np.asarray(gp.y_train_, dtype=float).reshape(-1)
    dim = X_train.shape[1]
    n_train = X_train.shape[0]
    kernel = gp.kernel_
    amplitude = float(kernel.k1.constant_value)
    ls = np.asarray(kernel.k2.length_scale, dtype=float).reshape(-1)
    if ls.size == 1:
        ls = np.full(dim, float(ls[0]))
    ls = np.maximum(ls, 1e-6)

    rng = np.random.default_rng(seed)
    total_features = int(batch_size) * int(n_features)
    # Matérn-5/2 spectral frequencies follow a Student-t distribution with 5 df.
    frequencies = rng.standard_t(df=5.0, size=(dim, total_features)) / ls[:, None]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=total_features)
    weights = rng.normal(size=(batch_size, n_features))
    scale = math.sqrt(2.0 * amplitude / n_features)

    train_cos = np.cos(X_train @ frequencies + phases).reshape(n_train, batch_size, n_features)
    cand_cos = np.cos(np.asarray(candidates, dtype=float) @ frequencies + phases).reshape(
        len(candidates), batch_size, n_features
    )
    prior_train = scale * np.einsum("nqm,qm->qn", train_cos, weights, optimize=True)
    prior_cand = scale * np.einsum("nqm,qm->qn", cand_cos, weights, optimize=True)

    # Matheron's rule: prior draw + K(x, X) K(X, X)^-1 [y - prior(X)].
    residual = y_train[None, :] - prior_train
    tmp = solve_triangular(gp.L_, residual.T, lower=True, check_finite=False)
    alpha_draw = solve_triangular(gp.L_.T, tmp, lower=False, check_finite=False).T
    K_cross = gp.kernel_(np.asarray(candidates, dtype=float), X_train)
    return prior_cand + alpha_draw @ K_cross.T


def _choose_turbo_training_subset(X: np.ndarray, y: np.ndarray, cap: int,
                                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the best, newest, and a seeded sample of history for a bounded GP fit."""
    n = len(y)
    if n <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    top_n = max(1, cap // 3)
    recent_n = max(1, cap // 3)
    selected = set(np.argsort(y)[:top_n].tolist())
    selected.update(range(max(0, n - recent_n), n))
    remaining = cap - len(selected)
    if remaining > 0:
        available = np.asarray(sorted(set(range(n)) - selected), dtype=int)
        if len(available) > 0:
            selected.update(rng.choice(available, size=min(remaining, len(available)), replace=False).tolist())
    idx = np.asarray(sorted(selected), dtype=int)
    if len(idx) > cap:
        # Keep all best/latest preferentially, then truncate deterministically.
        idx = idx[-cap:]
    return X[idx], y[idx]


def run_turbo(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    budget: int, seed: int, delay_ms: float = 0.0,
    batch_size: int = 5, candidate_count: int = 1024,
    rff_features: int = 128, max_train_points: int = 200,
    fit_every_batches: int = 10, gp_maxiter: int = 15,
    deadline: float | None = None,
    observation_noise_std: float = 0.0,
) -> RunResult:
    """TuRBO-1 for box-constrained minimization (resource-aware CPU variant).

    Algorithmic controls follow Eriksson et al. (NeurIPS 2019): normalized unit
    cube, local ARD Matérn-5/2 GP, Sobol candidate set, coordinate perturbation
    probability min(1, 20/d), Thompson sampling, adaptive trust-region length,
    and restarts below L_min. Approximate RFF posterior sampling, candidate cap,
    and local-history cap bound CPU/memory cost for the 2-hour experiment.
    """
    if budget < 8:
        raise ValueError("TuRBO requires a budget of at least 8 evaluations.")
    from scipy.optimize import minimize
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.exceptions import ConvergenceWarning

    started = time.perf_counter()
    objective = CountingObjective(objective_fn, delay_ms)
    lower = np.asarray(lower, dtype=float); upper = np.asarray(upper, dtype=float)
    width = upper - lower
    dim = int(len(x0))
    x0 = np.clip(np.asarray(x0, dtype=float), lower, upper)
    x0_unit = np.clip((x0 - lower) / width, 0.0, 1.0)
    best_x = x0.copy()
    best_value = objective(best_x)  # shared initialization; counts toward budget
    # Under known observation noise, recommend by local-GP posterior mean rather
    # than selecting only the lowest noisy observation.
    model_recommendation_x = best_x.copy()
    model_recommendation_value = math.inf

    # TuRBO paper and reference implementation recommend approximately 2*d initial points.
    n_init = min(max(20, 2 * dim), max(4, budget // 2))
    n_cand = max(256, min(int(candidate_count), 100 * dim, 5000))
    batch_size = max(1, int(batch_size))
    length_init, length_max, length_min = 0.8, 1.6, 2.0 ** -7
    succ_tol = 3
    fail_tol = int(math.ceil(max(4.0 / batch_size, dim / batch_size)))
    total_batches = 0
    restarts = 0
    stop_reason = "evaluation_budget"
    rng = np.random.default_rng(seed)

    # Restart loop: fresh initial design when a trust region becomes too small.
    first_restart = True
    while objective.count < budget and (deadline is None or time.perf_counter() < deadline):
        if not first_restart and budget - objective.count < n_init:
            stop_reason = "restart_budget_too_small"
            break
        first_restart = False
        restarts += 1
        if restarts == 1:
            lhs = qmc.LatinHypercube(d=dim, seed=stable_seed(seed, "turbo-init")).random(n=n_init)
            lhs[0] = x0_unit
            X_local = lhs
            initial_real = lower + lhs[1:] * width
            y_local = [best_value]
            init_vals = evaluate_batch(objective, initial_real, workers=1)
            y_local.extend(float(v) for v in init_vals)
            local_y = np.asarray(y_local, dtype=float)
            # Update global best over the initial design.
            for xx, yy in zip(initial_real, init_vals):
                if yy < best_value:
                    best_value = float(yy); best_x = np.asarray(xx, dtype=float).copy()
        else:
            lhs = qmc.LatinHypercube(d=dim, seed=stable_seed(seed, "turbo-restart", restarts)).random(n=n_init)
            X_local = lhs
            initial_real = lower + lhs * width
            init_vals = evaluate_batch(objective, initial_real, workers=1)
            local_y = np.asarray(init_vals, dtype=float)
            for xx, yy in zip(initial_real, init_vals):
                if yy < best_value:
                    best_value = float(yy); best_x = np.asarray(xx, dtype=float).copy()

        length = length_init
        succ_count = 0
        fail_count = 0
        kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
            length_scale=np.full(dim, 0.2), length_scale_bounds=(0.005, 2.0), nu=2.5
        )
        batches_since_fit = fit_every_batches
        fit_count = 0

        while objective.count < budget and length >= length_min and (deadline is None or time.perf_counter() < deadline):
            # Train on the current restart's data; cap history to avoid cubic GP growth.
            X_train_full, y_train_full = np.asarray(X_local, float), np.asarray(local_y, float)
            X_train, y_train = _choose_turbo_training_subset(
                X_train_full, y_train_full, max( max(10, int(max_train_points)), dim + 2),
                stable_seed(seed, restarts, total_batches, "train-subset"),
            )
            y_center = float(np.median(y_train))
            y_scale = float(np.std(y_train))
            if not np.isfinite(y_scale) or y_scale < 1e-12:
                y_scale = 1.0
            y_standard = (y_train - y_center) / y_scale

            tune = (fit_count == 0) or (batches_since_fit >= max(1, fit_every_batches))
            def limited_optimizer(obj_func, initial_theta, bounds):
                opt = minimize(
                    obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                    options={"maxiter": max(3, int(gp_maxiter)), "ftol": 1e-4},
                )
                return opt.x, opt.fun
            normalized_noise_var = (float(observation_noise_std) / y_scale) ** 2
            gp_alpha = max(1e-6, normalized_noise_var)
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=gp_alpha, normalize_y=False,
                optimizer=limited_optimizer if tune else None,
                n_restarts_optimizer=0, random_state=stable_seed(seed, restarts, total_batches),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X_train, y_standard)
            # The experiment-wide deadline is checked after fitting as well as
            # between batches, so a just-finished expensive GP fit stops cleanly.
            if deadline is not None and time.perf_counter() >= deadline:
                stop_reason = "wall_time_limit"
                break
            kernel = gp.kernel_
            fit_count += 1
            batches_since_fit = 0

            # Trust-region half-widths use ARD lengthscales with unit geometric mean.
            ls = np.asarray(gp.kernel_.k2.length_scale, dtype=float).reshape(-1)
            if ls.size == 1:
                ls = np.full(dim, float(ls[0]))
            log_weights = np.log(np.maximum(ls, 1e-8))
            weights = np.exp(log_weights - np.mean(log_weights))
            if observation_noise_std > 0.0:
                # Paper's noisy-objective variant centers the TR at the sampled
                # point with the smallest posterior mean, not the smallest noisy y.
                local_posterior_mean = gp.predict(X_local) * y_scale + y_center
                center_idx = int(np.argmin(local_posterior_mean))
                if float(local_posterior_mean[center_idx]) < model_recommendation_value:
                    model_recommendation_value = float(local_posterior_mean[center_idx])
                    model_recommendation_x = lower + np.asarray(X_local[center_idx], float) * width
            else:
                center_idx = int(np.argmin(local_y))
            center = np.asarray(X_local[center_idx], float)
            lb_tr = np.clip(center - weights * length / 2.0, 0.0, 1.0)
            ub_tr = np.clip(center + weights * length / 2.0, 0.0, 1.0)

            sobol_pow = int(math.ceil(math.log2(n_cand)))
            candidates_unit = qmc.Sobol(
                d=dim, scramble=True,
                seed=stable_seed(seed, "turbo-candidates", restarts, total_batches),
            ).random_base2(sobol_pow)[:n_cand]
            candidates_unit = lb_tr + (ub_tr - lb_tr) * candidates_unit
            prob_perturb = min(20.0 / dim, 1.0)
            mask_rng = np.random.default_rng(stable_seed(seed, "turbo-mask", restarts, total_batches))
            mask = mask_rng.random((n_cand, dim)) <= prob_perturb
            empty = np.where(np.sum(mask, axis=1) == 0)[0]
            if len(empty):
                chosen_dims = mask_rng.integers(0, dim, size=len(empty))
                mask[empty, chosen_dims] = True
            candidates_unit = np.where(mask, candidates_unit, center[None, :])

            q = min(batch_size, budget - objective.count)
            try:
                posterior_draws = _turbo_rff_posterior_samples(
                    gp, candidates_unit, q,
                    stable_seed(seed, "turbo-ts", restarts, total_batches),
                    n_features=max(32, int(rff_features)),
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # Numerically safe fallback: marginal TS from GP mean/std.
                mean, std = gp.predict(candidates_unit, return_std=True)
                fallback_rng = np.random.default_rng(stable_seed(seed, "turbo-fallback-ts", total_batches))
                posterior_draws = mean[None, :] + std[None, :] * fallback_rng.normal(size=(q, n_cand))

            next_indices = []
            for sample_i in range(q):
                idx = int(np.argmin(posterior_draws[sample_i]))
                next_indices.append(idx)
                posterior_draws[:, idx] = np.inf
            X_next_unit = candidates_unit[np.asarray(next_indices, dtype=int)]
            X_next = lower + X_next_unit * width
            y_next = np.asarray(evaluate_batch(objective, X_next, workers=1), dtype=float)
            incumbent = float(np.min(local_y))
            if len(y_next) and float(np.min(y_next)) < best_value:
                best_idx = int(np.argmin(y_next))
                best_value = float(y_next[best_idx]); best_x = X_next[best_idx].copy()
            X_local = np.vstack((X_local, X_next_unit))
            local_y = np.concatenate((local_y, y_next))

            # Paper/reference TuRBO-1 success/failure streak update.
            if float(np.min(y_next)) < incumbent - 1e-3 * abs(incumbent):
                succ_count += 1; fail_count = 0
            else:
                succ_count = 0; fail_count += 1
            if succ_count >= succ_tol:
                length = min(length_max, 2.0 * length)
                succ_count = 0; fail_count = 0
            elif fail_count >= fail_tol:
                length /= 2.0
                succ_count = 0; fail_count = 0
            total_batches += 1
            batches_since_fit += 1

        if deadline is not None and time.perf_counter() >= deadline and objective.count < budget:
            stop_reason = "wall_time_limit"
            break
        if objective.count >= budget:
            stop_reason = "evaluation_budget"
            break
        if length < length_min:
            # Restart below L_min as prescribed by TuRBO-1.
            continue
        # Usually only reached if an optimizer loop ended unexpectedly.
        if objective.count < budget:
            stop_reason = "stopped_before_budget"
            break

    if observation_noise_std > 0.0 and np.isfinite(model_recommendation_value):
        best_x = np.asarray(model_recommendation_x, dtype=float).copy()
    return RunResult("turbo", best_x, float(best_value), objective.count,
                     time.perf_counter() - started, objective.trace, stop_reason)


def run_random(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    seed: int,
    workers: int = 1,
    delay_ms: float = 0.0,
    deadline: float | None = None,
) -> RunResult:
    if budget < 2:
        raise ValueError("Budget must be at least 2 evaluations.")
    started = time.perf_counter()
    objective = CountingObjective(objective_fn, delay_ms)
    best_x = np.asarray(x0, dtype=float).copy()
    best_value = objective(best_x)
    rng = np.random.default_rng(seed)
    points = rng.uniform(lower, upper, size=(budget - 1, x0.size))
    pool_context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        # Fixed-budget, non-adaptive baseline; still benefits from parallel calls.
        vals: list[float] = []
        batch_size = max(1, workers * 4)
        for begin in range(0, len(points), batch_size):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            batch = [row for row in points[begin:begin + batch_size]]
            vals.extend(evaluate_batch(objective, batch, workers, pool_context))
    finally:
        if pool_context is not None:
            pool_context.shutdown(wait=True)
    best_index = int(np.argmin(vals)) if vals else -1
    if best_index >= 0 and vals[best_index] < best_value:
        best_value = float(vals[best_index])
        best_x = points[best_index].copy()
    stop_reason = "wall_time_limit" if deadline is not None and time.perf_counter() >= deadline and objective.count < budget else "evaluation_budget"
    return RunResult("random", best_x, best_value, objective.count,
                     time.perf_counter() - started, objective.trace, stop_reason)


def make_synthetic_problem(name: str, dim: int, problem_seed: int, highdim_bounds: bool = False):
    """Return a shifted, deterministic test objective, x0 and finite box bounds.

    All functions have a known global minimum of 0 at a nonzero, seeded shift.
    This avoids accidentally starting at the optimum of centred textbook forms.
    """
    if name not in SYNTHETIC_FUNCTIONS:
        raise ValueError(f"Unknown synthetic function {name!r}; choose from {SYNTHETIC_FUNCTIONS}.")
    if highdim_bounds and name == "ackley":
        # The TuRBO paper's 200D Ackley experiment uses [-5, 10]^d.
        lo, hi = -5.0, 10.0
    elif highdim_bounds and name == "rastrigin":
        # The paper's supplementary 10D Rastrigin setting uses [-3, 4]^d.
        lo, hi = -3.0, 4.0
    elif name == "rastrigin":
        lo, hi = -5.12, 5.12
    else:
        lo, hi = -5.0, 5.0
    lower = np.full(dim, lo, dtype=float)
    upper = np.full(dim, hi, dtype=float)

    shift_rng = np.random.default_rng(stable_seed("shift", name, dim))
    shift = shift_rng.uniform(-1.5, 1.5, size=dim)
    x0_rng = np.random.default_rng(stable_seed("start", name, dim, problem_seed))
    x0 = x0_rng.uniform(lower * 0.8, upper * 0.8)
    # Keep the common starting point away from the shifted optimum on the rare
    # occasion where a small-dimensional random draw lands unusually close.
    if np.linalg.norm(x0 - shift) < 0.25:
        x0 = np.clip(x0 + 1.0, lower, upper)

    def objective(x: np.ndarray) -> float:
        z = np.asarray(x, dtype=float) - shift
        if name == "sphere":
            return float(np.dot(z, z))
        if name == "rastrigin":
            return float(10.0 * dim + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z)))
        if name == "ackley":
            # Standard Ackley function translated so its global optimum is shift.
            a, b, c = 20.0, 0.2, 2.0 * np.pi
            mean_sq = float(np.mean(z * z))
            mean_cos = float(np.mean(np.cos(c * z)))
            return float(-a * np.exp(-b * math.sqrt(mean_sq)) - np.exp(mean_cos) + a + math.e)
        if name == "rosenbrock":
            # Translation of standard Rosenbrock: optimum is again x == shift.
            q = z + 1.0
            if dim < 2:
                return float((1.0 - q[0]) ** 2)
            return float(np.sum(100.0 * (q[1:] - q[:-1] ** 2) ** 2 + (1.0 - q[:-1]) ** 2))
        raise AssertionError("Function dispatch is exhaustive.")

    return objective, x0, lower, upper


def get_coco_bounds(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        lower = np.asarray(problem.lower_bounds, dtype=float)
        upper = np.asarray(problem.upper_bounds, dtype=float)
        return bounds_from_arrays(lower, upper, dim)
    except Exception:
        # Standard BBOB search domain for the supplied suite. This fallback is
        # explicit rather than passing infinite bounds into pycma/GP sampling.
        return np.full(dim, -5.0), np.full(dim, 5.0)


def execute_algorithm(
    algorithm: str,
    function: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    seed: int,
    workers: int,
    delay_ms: float,
    bo_startup: int,
    sigma_scale: float,
    bo_backend: str = "sklearn",
    observation_noise_std: float = 0.0,
    deadline: float | None = None,
    turbo_batch_size: int = 5,
    turbo_candidates: int = 1024,
    turbo_rff_features: int = 128,
    turbo_max_train: int = 200,
    turbo_fit_every: int = 10,
    turbo_gp_maxiter: int = 15,
    cma_backend: str = "auto",
) -> RunResult:
    if algorithm == "cma":
        return run_cma(function, x0, lower, upper, budget, seed, workers, delay_ms, sigma_scale, deadline, cma_backend)
    if algorithm == "turbo":
        return run_turbo(function, x0, lower, upper, budget, seed, delay_ms,
                          turbo_batch_size, turbo_candidates, turbo_rff_features,
                          turbo_max_train, turbo_fit_every, turbo_gp_maxiter, deadline,
                          observation_noise_std=observation_noise_std)
    if algorithm == "bo":
        return run_bo(function, x0, lower, upper, budget, seed, workers, delay_ms,
                      bo_startup, bo_backend, observation_noise_std)
    if algorithm == "random":
        return run_random(function, x0, lower, upper, budget, seed, workers, delay_ms, deadline)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    fields = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serializable(value) for key, value in row.items()})


def serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return " ".join(map(str, value.tolist()))
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return value


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    keys = ("experiment", "function", "dimension", "algorithm", "algorithm_backend", "bo_backend",
            "workers", "budget", "noise_model", "noise_level", "bo_noise_handling")
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    summary = []
    for group_key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        values = np.asarray([float(row["best_value"]) for row in group])
        times = np.asarray([float(row["wall_seconds"]) for row in group])
        evals = np.asarray([int(row["evaluations"]) for row in group])
        successes = [bool(row.get("success", False)) for row in group]
        item = {
            **dict(zip(keys, group_key)),
            "runs": len(group),
            # In noisy mode, best_value is the noise-free objective at the
            # algorithm's selected recommendation; elsewhere it is the best
            # observed objective value (identical on deterministic problems).
            "best_value_mean": float(np.mean(values)),
            "best_value_median": float(np.median(values)),
            "best_value_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "success_rate": float(np.mean(successes)),
            "wall_seconds_mean": float(np.mean(times)),
            "wall_seconds_median": float(np.median(times)),
            "evaluations_mean": float(np.mean(evals)),
        }
        if "best_observed_value" in group[0]:
            noisy_best = np.asarray([float(row["best_observed_value"]) for row in group])
            item["best_observed_value_mean"] = float(np.mean(noisy_best))
            item["best_observed_value_median"] = float(np.median(noisy_best))
            item["recommendation_true_value_mean"] = float(np.mean(values))
            item["recommendation_true_value_median"] = float(np.median(values))
        # Synthetic runs contain per-threshold flags. Show the whole success
        # curve so a strict configured target cannot hide meaningful progress.
        for target in DIAGNOSTIC_TARGETS:
            suffix = target_column_suffix(target)
            flag = f"success_at_{suffix}"
            if flag in group[0]:
                item[f"success_rate_at_{suffix}"] = float(np.mean([bool(row[flag]) for row in group]))
        summary.append(item)
    return summary


def append_trace_rows(trace_rows: list[dict], result: RunResult, metadata: dict) -> None:
    for point in result.trace:
        trace_rows.append({**metadata, **point})


def run_synthetic(args, algorithms: list[str], worker_values: list[int], output_dir: Path):
    rows: list[dict] = []
    traces: list[dict] = []
    dims = parse_int_list(args.dimensions)
    functions = parse_csv_list(args.functions)
    unknown = sorted(set(functions) - set(SYNTHETIC_FUNCTIONS))
    if unknown:
        raise SystemExit(f"Unknown synthetic functions: {unknown}. Available: {', '.join(SYNTHETIC_FUNCTIONS)}")

    for function_name in functions:
        for dim in dims:
            for repeat in range(args.repeats):
                run_seed = int(args.seed + repeat)
                fn, x0, lower, upper = make_synthetic_problem(function_name, dim, run_seed)
                budget = max(2, int(math.ceil(dim * args.budget_multiplier)))
                for workers in worker_values:
                    for algorithm_index, algorithm in enumerate(algorithms):
                        alg_seed = stable_seed(args.seed, function_name, dim, repeat, algorithm)
                        result = execute_algorithm(
                            algorithm, fn, x0, lower, upper, budget, alg_seed,
                            workers, args.evaluation_delay_ms, args.bo_startup, args.cma_sigma_scale,
                            args.bo_backend,
                        )
                        metadata = {
                            "experiment": "synthetic",
                            "function": function_name,
                            "dimension": dim,
                            "algorithm": algorithm,
                            "bo_backend": args.bo_backend if algorithm == "bo" else "n/a",
                            "repeat": repeat + 1,
                            "seed": alg_seed,
                            "workers": workers,
                            "budget": budget,
                            "evaluations": result.evaluations,
                            "best_value": result.best_value,
                            "wall_seconds": result.wall_seconds,
                            "target": args.target,
                            "success": bool(result.best_value <= args.target),
                            **{
                                f"success_at_{target_column_suffix(threshold)}": bool(result.best_value <= threshold)
                                for threshold in DIAGNOSTIC_TARGETS
                            },
                            "stop_reason": result.stop_reason,
                            "evaluation_delay_ms": args.evaluation_delay_ms,
                            "best_x": result.best_x,
                        }
                        rows.append(metadata)
                        append_trace_rows(traces, result, {
                            "experiment": "synthetic", "function": function_name,
                            "dimension": dim, "algorithm": algorithm,
                            "bo_backend": args.bo_backend if algorithm == "bo" else "n/a",
                            "repeat": repeat + 1, "seed": alg_seed, "workers": workers,
                            "budget": budget,
                        })
                        print(f"{function_name:10s} d={dim:2d} {algorithm:6s} workers={workers} "
                              f"run={repeat + 1}/{args.repeats} evals={result.evaluations:4d} "
                              f"best={result.best_value:.4g} time={result.wall_seconds:.2f}s")

    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "traces.csv", traces)
    summary = summarize(rows)
    write_csv(output_dir / "summary.csv", summary)
    return rows, summary



def run_highdim(args, algorithms: list[str], output_dir: Path):
    """Run the TuRBO/CMA-ES/random comparison with a hard global wall-clock deadline."""
    started = time.perf_counter()
    deadline = started + float(args.max_runtime_hours) * 3600.0
    rows: list[dict] = []
    traces: list[dict] = []
    functions = parse_csv_list(args.functions)
    allowed = {"ackley", "rastrigin", "rosenbrock"}
    unknown = sorted(set(functions) - allowed)
    if unknown:
        raise SystemExit(f"Highdim mode accepts {sorted(allowed)}; got {unknown}.")
    dims = parse_int_list(args.dimensions)
    timed_out = False
    print(f"High-dimensional benchmark: wall-clock cap={args.max_runtime_hours:.2f}h; "
          f"dimensions={dims}; functions={functions}; algorithms={algorithms}")

    for function_name in functions:
        if timed_out: break
        for dim in dims:
            if timed_out: break
            for repeat in range(args.repeats):
                if time.perf_counter() >= deadline:
                    timed_out = True; break
                run_seed = int(args.seed + repeat)
                fn, x0, lower, upper = make_synthetic_problem(function_name, dim, run_seed, highdim_bounds=True)
                budget = max(8, int(math.ceil(dim * args.budget_multiplier)))
                # Randomized order per condition reduces systematic warm-start/order effects.
                alg_order = list(algorithms)
                np.random.default_rng(stable_seed(args.seed, function_name, dim, repeat, "order")).shuffle(alg_order)
                for algorithm in alg_order:
                    if time.perf_counter() >= deadline:
                        timed_out = True; break
                    alg_seed = stable_seed(args.seed, function_name, dim, repeat, algorithm)
                    result = execute_algorithm(
                        algorithm, fn, x0, lower, upper, budget, alg_seed,
                        1, 0.0, args.bo_startup, args.cma_sigma_scale,
                        args.bo_backend, 0.0, deadline,
                        args.turbo_batch_size, args.turbo_candidates,
                        args.turbo_rff_features, args.turbo_max_train,
                        args.turbo_fit_every, args.turbo_gp_maxiter, args.cma_backend,
                    )
                    backend = (
                        "TuRBO-1/sklearn-GP+RFF-TS" if algorithm == "turbo" else
                        (("pycma" if cma is not None else "numpy-CMA-ES-fallback")
                         if algorithm == "cma" and args.cma_backend == "auto" else
                         "pycma" if algorithm == "cma" and args.cma_backend == "pycma" else
                         "numpy-CMA-ES-fallback" if algorithm == "cma" else "numpy-random")
                    )
                    row = {
                        "experiment": "highdim", "function": function_name,
                        "dimension": dim, "algorithm": algorithm,
                        "algorithm_backend": backend, "bo_backend": "n/a",
                        "repeat": repeat + 1, "seed": alg_seed, "workers": 1,
                        "budget": budget, "evaluations": result.evaluations,
                        "best_value": result.best_value, "wall_seconds": result.wall_seconds,
                        "experiment_elapsed_seconds": time.perf_counter() - started,
                        "wall_cap_hours": args.max_runtime_hours,
                        "target": args.target, "success": bool(result.best_value <= args.target),
                        **{f"success_at_{target_column_suffix(t)}": bool(result.best_value <= t)
                           for t in DIAGNOSTIC_TARGETS},
                        "stop_reason": result.stop_reason, "best_x": result.best_x,
                    }
                    rows.append(row)
                    append_trace_rows(traces, result, {
                        "experiment": "highdim", "function": function_name,
                        "dimension": dim, "algorithm": algorithm,
                        "algorithm_backend": backend, "repeat": repeat + 1,
                        "seed": alg_seed, "workers": 1, "budget": budget,
                    })
                    print(f"{function_name:10s} d={dim:3d} {algorithm:6s} "
                          f"run={repeat + 1}/{args.repeats} evals={result.evaluations:4d}/{budget} "
                          f"best={result.best_value:.5g} time={result.wall_seconds:.2f}s "
                          f"[{result.stop_reason}]")
                    # Checkpoint after each complete run so a long experiment is recoverable.
                    write_csv(output_dir / "runs.csv", rows)
                    write_csv(output_dir / "summary.csv", summarize(rows))
                if timed_out: break

    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "traces.csv", traces)
    summary = summarize(rows)
    write_csv(output_dir / "summary.csv", summary)
    actual = time.perf_counter() - started
    with (output_dir / "experiment_status.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"wall_cap_hours={args.max_runtime_hours}\n")
        handle.write(f"elapsed_seconds={actual:.6f}\n")
        handle.write(f"runs_completed={len(rows)}\n")
        handle.write(f"time_cap_reached={timed_out or actual >= args.max_runtime_hours * 3600.0}\n")
        handle.write(f"algorithms={','.join(algorithms)}\nfunctions={','.join(functions)}\ndimensions={','.join(map(str,dims))}\n")
        handle.write(f"budget_multiplier={args.budget_multiplier}\nrepeats={args.repeats}\n")
    print(f"High-dimensional experiment ended after {actual/60.0:.1f} minutes; runs={len(rows)}; "
          f"wall cap={'reached' if timed_out or actual >= args.max_runtime_hours*3600.0 else 'not reached'}.")
    return rows, summary


def make_noisy_evaluator(
    noiseless_function: Callable[[np.ndarray], float],
    noise_model: str,
    noise_std: float,
    seed: int,
) -> Callable[[np.ndarray], float]:
    """Wrap a noise-free objective with independent, zero-mean observation noise.

    `student_t` uses a t distribution with 3 degrees of freedom divided by
    sqrt(3), giving it unit variance and making its standard deviation match
    the Gaussian model's noise_std while retaining heavier tails.
    """
    if noise_model not in ("gaussian", "student_t"):
        raise ValueError(f"Unknown noise model {noise_model!r}.")
    rng = np.random.default_rng(seed)

    def noisy_function(x: np.ndarray) -> float:
        value = float(noiseless_function(np.asarray(x, dtype=float)))
        if noise_std <= 0:
            return value
        if noise_model == "gaussian":
            noise = float(rng.normal(0.0, noise_std))
        else:
            noise = float(rng.standard_t(df=3) / math.sqrt(3.0) * noise_std)
        return value + noise

    return noisy_function


def run_noisy(args, algorithms: list[str], output_dir: Path):
    """Compare optimizers on synthetic functions with noisy observations.

    Every candidate is evaluated once under independent observation noise.
    The returned recommendation is then scored against the known deterministic
    objective. That oracle evaluation is a benchmark diagnostic and is not
    counted against the noisy black-box evaluation budget.
    """
    rows: list[dict] = []
    traces: list[dict] = []
    dims = parse_int_list(args.dimensions)
    functions = parse_csv_list(args.functions)
    unknown = sorted(set(functions) - set(SYNTHETIC_FUNCTIONS))
    if unknown:
        raise SystemExit(f"Unknown synthetic functions: {unknown}. Available: {', '.join(SYNTHETIC_FUNCTIONS)}")
    noise_models = parse_csv_list(args.noise_models)
    bad_models = sorted(set(noise_models) - {"gaussian", "student_t"})
    if bad_models:
        raise SystemExit(f"Unknown noise models: {bad_models}. Choose from: gaussian, student_t")
    noise_levels = parse_float_list(args.noise_levels)

    deadline = time.perf_counter() + args.max_runtime_hours * 3600.0
    timed_out = False
    for function_name in functions:
        if time.perf_counter() >= deadline:
            timed_out = True; break
        for dim in dims:
            if time.perf_counter() >= deadline:
                timed_out = True; break
            for repeat in range(args.repeats):
                if time.perf_counter() >= deadline:
                    timed_out = True; break
                problem_seed = int(args.seed + repeat)
                noiseless_fn, x0, lower, upper = make_synthetic_problem(function_name, dim, problem_seed)
                # Calibrate the noise scale from a fixed, seeded random design over
                # the whole domain, not f(x0): the latter can vary wildly between
                # repeats and is exceptionally large for Rosenbrock. This keeps
                # noise_std fixed across repeats for a given function/dimension.
                scale_rng = np.random.default_rng(stable_seed("noise-reference", function_name, dim))
                reference_points = scale_rng.uniform(lower, upper, size=(512, dim))
                reference_values = np.asarray([noiseless_fn(point) for point in reference_points], dtype=float)
                reference_scale = max(1.0, float(np.median(np.abs(reference_values))))
                budget = max(2, int(math.ceil(dim * args.budget_multiplier)))
                for noise_model in noise_models:
                    if timed_out:
                        break
                    for noise_level in noise_levels:
                        if timed_out:
                            break
                        noise_std = float(noise_level * reference_scale)
                        for algorithm in algorithms:
                            if time.perf_counter() >= deadline:
                                timed_out = True
                                break
                            # Keep optimizer randomness paired across noise levels/models;
                            # only the observation-noise RNG changes across these conditions.
                            alg_seed = stable_seed(
                                args.seed, function_name, dim, repeat, algorithm, "optimizer",
                            )
                            noise_seed = stable_seed(
                                args.seed, function_name, dim, repeat, noise_model,
                                noise_level, algorithm, "observation-noise",
                            )
                            noisy_fn = make_noisy_evaluator(
                                noiseless_fn, noise_model, noise_std, noise_seed
                            )
                            result = execute_algorithm(
                                algorithm, noisy_fn, x0, lower, upper, budget,
                                alg_seed, workers=1, delay_ms=0.0,
                                bo_startup=args.bo_startup,
                                sigma_scale=args.cma_sigma_scale,
                                bo_backend=args.bo_backend,
                                observation_noise_std=(noise_std if args.bo_noise_aware == "known" else 0.0),
                                deadline=deadline,
                                turbo_batch_size=args.turbo_batch_size,
                                turbo_candidates=args.turbo_candidates,
                                turbo_rff_features=args.turbo_rff_features,
                                turbo_max_train=args.turbo_max_train,
                                turbo_fit_every=args.turbo_fit_every,
                                turbo_gp_maxiter=args.turbo_gp_maxiter,
                                cma_backend=args.cma_backend,
                            )
                            # A true-objective evaluation avoids selecting a
                            # winner merely because it received a favourable
                            # noise draw. This oracle call is diagnostic only.
                            true_value = float(noiseless_fn(result.best_x))
                            backend = (
                                "TuRBO-1/local-GP+RFF-Thompson" if algorithm == "turbo" else
                                ("sklearn-GP-EI" if algorithm == "bo" and args.bo_backend == "sklearn" else
                                 "Optuna-GPSampler" if algorithm == "bo" else
                                 ("pycma" if args.cma_backend == "pycma" else
                                  ("pycma" if cma is not None else "numpy-CMA-ES-fallback"))
                                 if algorithm == "cma" else "numpy-random")
                            )
                            row = {
                                "experiment": "noisy",
                                "function": function_name,
                                "dimension": dim,
                                "algorithm": algorithm,
                                "algorithm_backend": backend,
                                "bo_backend": args.bo_backend if algorithm == "bo" else "n/a",
                                "noise_model": noise_model,
                                "noise_level": float(noise_level),
                                "noise_std_objective_units": noise_std,
                                "bo_noise_handling": args.bo_noise_aware if algorithm in ("bo", "turbo") else "n/a",
                                "surrogate_noise_handling": args.bo_noise_aware if algorithm in ("bo", "turbo") else "n/a",
                                "noise_reference_scale": reference_scale,
                                "repeat": repeat + 1,
                                "seed": alg_seed,
                                "noise_seed": noise_seed,
                                "workers": 1,
                                "budget": budget,
                                "evaluations": result.evaluations,
                                # Primary result is the noise-free quality of
                                # the recommendation, not the lowest noisy draw.
                                "best_value": true_value,
                                "recommendation_true_value": true_value,
                                "best_observed_value": result.best_value,
                                "wall_seconds": result.wall_seconds,
                                "target": args.target,
                                "success": bool(true_value <= args.target),
                                **{
                                    f"success_at_{target_column_suffix(threshold)}": bool(true_value <= threshold)
                                    for threshold in DIAGNOSTIC_TARGETS
                                },
                                "stop_reason": result.stop_reason,
                                "best_x": result.best_x,
                            }
                            rows.append(row)
                            append_trace_rows(traces, result, {
                                "experiment": "noisy", "function": function_name,
                                "dimension": dim, "algorithm": algorithm,
                                "algorithm_backend": backend,
                                "bo_backend": args.bo_backend if algorithm == "bo" else "n/a",
                                "noise_model": noise_model,
                                "noise_level": float(noise_level),
                                "noise_std_objective_units": noise_std,
                                "bo_noise_handling": args.bo_noise_aware if algorithm in ("bo", "turbo") else "n/a",
                                "surrogate_noise_handling": args.bo_noise_aware if algorithm in ("bo", "turbo") else "n/a",
                                "repeat": repeat + 1, "seed": alg_seed,
                                "noise_seed": noise_seed, "workers": 1,
                                "budget": budget,
                            })
                            print(
                                f"{function_name:10s} d={dim:2d} {noise_model:9s} "
                                f"noise={noise_level:5.3g} {algorithm:6s} "
                                f"run={repeat + 1}/{args.repeats} evals={result.evaluations:4d} "
                                f"true={true_value:.4g} observed={result.best_value:.4g} "
                                f"time={result.wall_seconds:.2f}s"
                            )

    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "traces.csv", traces)
    summary = summarize(rows)
    write_csv(output_dir / "summary.csv", summary)
    if timed_out:
        print(f"Noisy experiment reached the {args.max_runtime_hours:g}-hour wall-clock limit; "
              f"saved {len(rows)} completed runs in {output_dir.resolve()}.")
    else:
        elapsed = time.perf_counter() - (deadline - args.max_runtime_hours * 3600.0)
        print(f"Noisy experiment completed in {elapsed:.2f}s (wall-clock cap "
              f"{args.max_runtime_hours:g} hours).")
    return rows, summary


def coco_final_target_hit(problem) -> bool:
    value = getattr(problem, "final_target_hit", False)
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def run_coco(args, algorithms: list[str], output_dir: Path):
    if args.workers != 1 or args.worker_sweep:
        raise SystemExit(
            "COCO mode deliberately uses workers=1. Its problem objects maintain mutable evaluation "
            "counters and observers; use synthetic mode with --worker-sweep for a controlled parallelism test."
        )
    try:
        import cocoex
    except ImportError as exc:
        raise SystemExit("COCO mode needs coco-experiment: python -m pip install coco-experiment cocopp") from exc

    rows: list[dict] = []
    traces: list[dict] = []
    for repeat in range(args.repeats):
        for algorithm in algorithms:
            backend_label = args.bo_backend if algorithm == "bo" else "n_a"
            run_folder = output_dir / "coco" / f"{algorithm}_{backend_label}_repeat_{repeat + 1:02d}"
            run_folder.parent.mkdir(parents=True, exist_ok=True)
            suite_options = f"dimensions: {args.dimensions} function_indices: {args.function_indices}"
            suite = cocoex.Suite(args.suite_name, f"instances: {args.instances}", suite_options)
            observer = cocoex.Observer(
                args.suite_name,
                f"result_folder: {run_folder.as_posix()} algorithm_name: {algorithm}",
            )
            problems_seen = 0
            for problem in suite:
                if args.max_problems and problems_seen >= args.max_problems:
                    break
                problems_seen += 1
                problem.observe_with(observer)
                dim = int(problem.dimension)
                lower, upper = get_coco_bounds(problem, dim)
                x0 = np.asarray(problem.initial_solution, dtype=float)
                x0 = np.clip(x0, lower, upper)
                budget = max(2, int(math.ceil(dim * args.budget_multiplier)))
                function = lambda x, _problem=problem: float(_problem(x))
                seed = stable_seed(args.seed, repeat, algorithm, str(problem.id))
                result = execute_algorithm(
                    algorithm, function, x0, lower, upper, budget, seed,
                    1, 0.0, args.bo_startup, args.cma_sigma_scale, args.bo_backend,
                )
                triple = getattr(problem, "id_triple", (None, dim, None))
                success = coco_final_target_hit(problem)
                row = {
                    "experiment": "coco", "suite": args.suite_name,
                    "function": triple[0] if len(triple) > 0 else "",
                    "dimension": dim,
                    "instance": triple[2] if len(triple) > 2 else "",
                    "problem_id": str(problem.id), "algorithm": algorithm,
                    "bo_backend": args.bo_backend if algorithm == "bo" else "n/a",
                    "repeat": repeat + 1, "seed": seed, "workers": 1,
                    "budget": budget, "evaluations": result.evaluations,
                    "best_value": result.best_value, "wall_seconds": result.wall_seconds,
                    "target": "COCO final target", "success": success,
                    "stop_reason": result.stop_reason,
                    "observer_folder": run_folder.as_posix(), "best_x": result.best_x,
                }
                rows.append(row)
                append_trace_rows(traces, result, {
                    "experiment": "coco", "suite": args.suite_name,
                    "function": row["function"], "dimension": dim,
                    "instance": row["instance"], "problem_id": str(problem.id),
                    "algorithm": algorithm,
                    "bo_backend": args.bo_backend if algorithm == "bo" else "n/a",
                    "repeat": repeat + 1, "seed": seed,
                    "workers": 1, "budget": budget,
                })
                print(f"{str(problem.id):38s} {algorithm:6s} evals={result.evaluations:4d}/{budget} "
                      f"best={result.best_value:.4g} time={result.wall_seconds:.2f}s target={success}")
            del observer
            del suite

    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "traces.csv", traces)
    summary = summarize(rows)
    write_csv(output_dir / "summary.csv", summary)
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare TuRBO-1, CMA-ES, Gaussian-process BO, and random search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("experiment", choices=("coco", "synthetic", "noisy", "highdim"),
                        help="COCO/BBOB, deterministic synthetic, noisy synthetic, or time-limited high-dimensional comparison")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated subset of cma,bo,random,turbo; highdim defaults to turbo,cma,random")
    parser.add_argument("--dimensions", default=None,
                        help="Comma-separated dimension list; highdim defaults to 20,50,100,200")
    parser.add_argument("--budget-multiplier", type=float, default=None,
                        help="Evaluation budget is ceil(dimension * multiplier); highdim defaults to 5")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Independent optimizer seeds/starts per configuration; highdim defaults to 3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker count for synthetic candidate evaluations / asynchronous BO trials")
    parser.add_argument("--worker-sweep", default="",
                        help="Synthetic mode only; e.g. '1,2,4,8' to compare wall time and quality")
    parser.add_argument("--evaluation-delay-ms", type=float, default=0.0,
                        help="Synthetic mode only; simulated time per objective call, for throughput tests")
    parser.add_argument("--functions", default=None,
                        help="Functions; highdim defaults to ackley,rastrigin,rosenbrock")
    parser.add_argument("--noise-models", default="gaussian,student_t",
                        help="Noisy mode only: gaussian,student_t")
    parser.add_argument("--noise-levels", default="0,0.05,0.2",
                        help="Noisy mode only: relative observation noise standard deviations")
    parser.add_argument("--bo-noise-aware", "--surrogate-noise-aware", dest="bo_noise_aware",
                        choices=("known", "ignore"), default="known",
                        help="Noisy mode: tell GP surrogates BO and TuRBO the known observation-noise std, or ignore it")
    parser.add_argument("--function-indices", default="1-24",
                        help="COCO suite function filter, e.g. '1-5' for a quick pilot")
    parser.add_argument("--instances", default="1-5", help="COCO suite instance filter, e.g. '1-2'")
    parser.add_argument("--suite-name", default="bbob")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="COCO mode only; 0 means all selected problems")
    parser.add_argument("--bo-startup", type=int, default=10)
    parser.add_argument("--bo-backend", choices=("sklearn", "optuna"), default="sklearn")
    parser.add_argument("--cma-sigma-scale", type=float, default=0.3)
    parser.add_argument("--cma-backend", choices=("auto", "pycma", "numpy"), default="auto",
                        help="auto uses pycma when installed; numpy is a bundled reference CMA-ES fallback")
    parser.add_argument("--target", type=float, default=1e-8)
    parser.add_argument("--max-runtime-hours", type=float, default=2.0,
                        help="highdim/noisy mode global wall-clock cap; stops starting new runs at this deadline")
    parser.add_argument("--turbo-batch-size", type=int, default=5,
                        help="TuRBO candidate batch size; failure tolerance scales with d/batch_size")
    parser.add_argument("--turbo-candidates", type=int, default=256,
                        help="TuRBO's per-batch candidate-pool cap (paper uses min(100*d,5000); lower cap saves time)")
    parser.add_argument("--turbo-rff-features", type=int, default=32,
                        help="Random Fourier features used for approximate coherent GP Thompson samples")
    parser.add_argument("--turbo-max-train", type=int, default=160,
                        help="Maximum TuRBO local GP training observations; caps cubic GP cost")
    parser.add_argument("--turbo-fit-every", type=int, default=8,
                        help="Re-optimise GP hyperparameters every this many TuRBO batches")
    parser.add_argument("--turbo-gp-maxiter", type=int, default=4,
                        help="Maximum L-BFGS steps when optimising TuRBO GP hyperparameters")
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Mode-specific defaults keep the two-hour high-dimensional experiment concise.
    is_highdim = args.experiment == "highdim"
    if args.algorithms is None:
        args.algorithms = "turbo,cma,random" if is_highdim else "cma,bo,random"
    if args.dimensions is None:
        args.dimensions = "20,50,100,200" if is_highdim else "2,3,5,10"
    if args.budget_multiplier is None:
        args.budget_multiplier = 3.0 if is_highdim else 10.0
    if args.repeats is None:
        args.repeats = 3
    if args.functions is None:
        args.functions = "ackley,rastrigin,rosenbrock" if is_highdim else "sphere,rastrigin,ackley,rosenbrock"
    if args.output_dir is None:
        args.output_dir = "results/highdim_turbo_cma_random" if is_highdim else f"results/{args.experiment}"
    try:
        algorithms = parse_csv_list(args.algorithms)
        unknown = sorted(set(algorithms) - set(ALGORITHMS))
        if unknown:
            parser.error(f"Unknown algorithms: {unknown}. Choose from {', '.join(ALGORITHMS)}")
        if len(set(algorithms)) != len(algorithms):
            parser.error("List each algorithm only once.")
        dimensions = parse_int_list(args.dimensions)
        if args.repeats < 1:
            parser.error("--repeats must be >= 1.")
        if args.budget_multiplier <= 0:
            parser.error("--budget-multiplier must be positive.")
        if args.workers < 1:
            parser.error("--workers must be >= 1.")
        if args.evaluation_delay_ms < 0:
            parser.error("--evaluation-delay-ms must be >= 0.")
        if args.bo_startup < 2:
            parser.error("--bo-startup must be >= 2.")
        if "bo" in algorithms and args.bo_backend == "sklearn" and (args.workers > 1 or args.worker_sweep):
            parser.error("The sklearn BO backend is sequential; use --bo-backend optuna for --workers > 1 or --worker-sweep.")
        if args.max_runtime_hours <= 0:
            parser.error("--max-runtime-hours must be positive.")
        if min(args.turbo_batch_size, args.turbo_candidates, args.turbo_rff_features,
               args.turbo_max_train, args.turbo_fit_every, args.turbo_gp_maxiter) < 1:
            parser.error("All --turbo-* numeric settings must be positive.")
        if args.cma_backend == "pycma" and cma is None and "cma" in algorithms:
            parser.error("--cma-backend pycma requires the cma package: python -m pip install cma")
        if is_highdim:
            if set(algorithms) - {"turbo", "cma", "random"}:
                parser.error("highdim mode compares only turbo,cma,random; exclude global BO.")
            if args.workers != 1 or args.worker_sweep:
                parser.error("highdim mode runs single-worker methods for a controlled algorithm comparison.")
            if args.evaluation_delay_ms:
                parser.error("highdim mode uses analytic benchmark functions and does not use simulated latency.")
            if args.bo_backend != "sklearn":
                print("Note: --bo-backend is irrelevant in highdim mode.")
        if args.max_problems < 0:
            parser.error("--max-problems must be >= 0.")
        worker_values = parse_int_list(args.worker_sweep) if args.worker_sweep else [args.workers]
        if args.experiment == "coco" and (args.worker_sweep or args.workers != 1):
            parser.error("COCO mode requires --workers 1 and no --worker-sweep; use synthetic mode for parallelism.")
        if args.experiment in ("noisy", "highdim") and args.worker_sweep:
            parser.error(f"{args.experiment} mode does not support --worker-sweep.")
        if args.experiment == "synthetic" and args.worker_sweep:
            args.workers = 1  # worker_values, rather than this value, drives the sweep.
        else:
            worker_values = [args.workers]
        if args.experiment == "noisy" and (args.worker_sweep or args.workers != 1):
            parser.error("Noisy mode uses one worker so the random-noise stream is reproducible; use synthetic mode for worker sweeps.")
        if args.experiment == "noisy" and "bo" in algorithms and args.bo_backend != "sklearn":
            parser.error("Noisy mode currently uses the sklearn BO backend, which is configured with the known observation-noise variance; use --bo-backend sklearn.")
        if args.experiment == "noisy" and args.evaluation_delay_ms:
            parser.error("--evaluation-delay-ms is for synthetic parallelism experiments, not noisy mode.")
        if args.experiment == "coco" and args.evaluation_delay_ms:
            parser.error("--evaluation-delay-ms applies only to synthetic mode.")
        if args.experiment == "coco" and args.functions != "sphere,rastrigin,ackley,rosenbrock":
            print("Note: --functions is only used in synthetic mode; COCO uses --function-indices.")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment={args.experiment}; algorithms={algorithms}; dimensions={args.dimensions}; "
          f"budget multiplier={args.budget_multiplier}; repeats={args.repeats}; workers={worker_values}; "
          f"BO backend={args.bo_backend}")
    if args.experiment == "synthetic":
        rows, summary = run_synthetic(args, algorithms, worker_values, output_dir)
    elif args.experiment == "noisy":
        rows, summary = run_noisy(args, algorithms, output_dir)
    elif args.experiment == "highdim":
        rows, summary = run_highdim(args, algorithms, output_dir)
    else:
        rows, summary = run_coco(args, algorithms, output_dir)
    print(f"\nFinished {len(rows)} runs. Results saved to {output_dir.resolve()}")
    print(f"  Raw run table: {output_dir / 'runs.csv'}")
    print(f"  Summary table: {output_dir / 'summary.csv'}")
    print(f"  Best-so-far traces: {output_dir / 'traces.csv'}")
    if args.experiment == "coco":
        print(f"  COCO observer data: {output_dir / 'coco'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; already-written result files remain on disk.", file=sys.stderr)
        raise SystemExit(130)
