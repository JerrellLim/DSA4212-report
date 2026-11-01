#!/usr/bin/env python3
"""Experiment 3: wall-clock cost and parallel evaluations for CMA, TuRBO, and naive BO.

Requires experiment_jerrell_rewritten.py beside this file. Naive BO uses the sequential
scikit-learn GP/EI implementation and is intentionally measured at one worker only;
CMA and TuRBO evaluate candidate batches in parallel. The delay sweep is a controlled
simulation, not a claim that all real objective calls have the same cost.
"""
from __future__ import annotations
import argparse, time
from contextlib import contextmanager
from threading import Lock
from pathlib import Path
import numpy as np
import experiment_jerrell_rewritten as core


@contextmanager
def batch_workers(worker_count: int):
    """Temporarily make internal batch evaluators use worker_count where their API hardcodes 1.

    The patch is scoped and restored immediately. The experiments run serially at the
    outer level, so this does not create concurrent mutation of the shared module.
    """
    original = core.evaluate_batch
    def wrapped(objective, points, workers=1, executor=None):
        actual = worker_count if workers <= 1 else workers
        return original(objective, points, workers=actual, executor=executor)
    core.evaluate_batch = wrapped
    try:
        yield
    finally:
        core.evaluate_batch = original


def run_method(algorithm, f, x0, lo, hi, budget, seed, workers, delay_ms, args, deadline=None):
    started = time.perf_counter()
    events = []
    event_lock = Lock()
    def measured_objective(x):
        value = float(f(np.asarray(x, dtype=float)))
        with event_lock:
            events.append({"value": value, "elapsed_seconds": time.perf_counter() - started})
        return value
    if algorithm == "cma":
        # pycma parallelizes generation evaluation. The NumPy fallback is also given
        # parallel evaluation by the scoped evaluate_batch patch.
        with batch_workers(workers):
            result = core.run_cma(measured_objective, x0, lo, hi, budget, seed, workers=workers,
                                  delay_ms=delay_ms, sigma_scale=0.3, deadline=deadline,
                                  backend=args.cma_backend)
        backend = "pycma" if core.cma is not None and args.cma_backend != "numpy" else "numpy-CMA-ES-fallback"
        effective_workers = workers
    elif algorithm == "turbo":
        # The core TuRBO code currently calls evaluate_batch(..., workers=1); the
        # scoped patch sends these initial/batch evaluations to the requested pool.
        with batch_workers(workers):
            result = core.run_turbo(measured_objective, x0, lo, hi, budget, seed, delay_ms=delay_ms,
                batch_size=args.turbo_batch_size, candidate_count=args.turbo_candidates,
                rff_features=args.turbo_rff_features, max_train_points=args.turbo_max_train,
                fit_every_batches=args.turbo_fit_every, gp_maxiter=args.turbo_gp_maxiter, deadline=deadline)
        backend = "TuRBO-1-local-GP-RFF-TS"
        effective_workers = workers
    elif algorithm == "random":
        result = core.run_random(measured_objective, x0, lo, hi, budget, seed, workers=workers, delay_ms=delay_ms, deadline=deadline)
        backend = "uniform-random"
        effective_workers = workers
    elif algorithm == "bo":
        # The chosen naive BO baseline is sequential. Do not pass workers>1 to it or
        # pretend it has parallel speed-up; only run once per condition, at 1 worker.
        result = core.run_bo_sklearn(measured_objective, x0, lo, hi, budget, seed, workers=1,
                                     delay_ms=delay_ms, bo_startup=min(args.bo_startup, max(2, budget//2)))
        backend = "naive-sklearn-GP-EI-sequential"
        effective_workers = 1
    else:
        raise ValueError(algorithm)
    elapsed = time.perf_counter() - started
    return result, elapsed, backend, effective_workers, events


def summarize_rows(rows: list[dict]) -> list[dict]:
    keys = ("function", "dimension", "algorithm", "algorithm_backend", "configured_workers",
            "effective_workers", "parallel_supported", "evaluation_delay_ms", "budget", "target")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out = []
    for k, group in sorted(groups.items(), key=lambda z: tuple(str(v) for v in z[0])):
        vals=np.asarray([float(r["best_value"]) for r in group]); times=np.asarray([float(r["wall_seconds"]) for r in group])
        out.append({**dict(zip(keys,k)), "runs":len(group), "best_value_mean":float(np.mean(vals)),
                    "best_value_median":float(np.median(vals)),
                    "best_value_std":float(np.std(vals,ddof=1)) if len(group)>1 else 0.0,
                    "wall_seconds_mean":float(np.mean(times)), "wall_seconds_median":float(np.median(times)),
                    "evaluations_per_second_mean":float(np.mean([r["evaluations_per_second"] for r in group])),
                    "success_rate_reaching_target":float(np.mean([bool(r["target_hit"]) for r in group])),
                    "median_time_to_target_seconds":float(np.median([float(r["time_to_target_seconds"]) for r in group if r["time_to_target_seconds"] != ""])) if any(r["time_to_target_seconds"] != "" for r in group) else "",
                    "median_evaluations_completed_by_target":float(np.median([float(r["evaluations_completed_by_target"]) for r in group if r["evaluations_completed_by_target"] != ""])) if any(r["evaluations_completed_by_target"] != "" for r in group) else "",
                    "speedup_vs_1_worker_mean":float(np.mean([float(r["speedup_vs_1_worker"]) for r in group if r["speedup_vs_1_worker"] != ""])) if any(r["speedup_vs_1_worker"] != "" for r in group) else ""})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", default="sphere,rastrigin")
    p.add_argument("--dimensions", default="5,10", help="Keep naive BO at 10D or below.")
    p.add_argument("--budget-multiplier", type=float, default=5.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--worker-counts", default="1,2,4")
    p.add_argument("--delays-ms", default="0,20,100")
    p.add_argument("--bo-startup", type=int, default=8)
    p.add_argument("--cma-backend", choices=("auto", "pycma", "numpy"), default="auto")
    p.add_argument("--target", type=float, default=1.0, help="Target used to measure time/evaluations-to-target.")
    p.add_argument("--max-runtime-hours", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="results/cost_parallelism")
    p.add_argument("--turbo-batch-size", type=int, default=4)
    p.add_argument("--turbo-candidates", type=int, default=256)
    p.add_argument("--turbo-rff-features", type=int, default=32)
    p.add_argument("--turbo-max-train", type=int, default=100)
    p.add_argument("--turbo-fit-every", type=int, default=8)
    p.add_argument("--turbo-gp-maxiter", type=int, default=4)
    args = p.parse_args()
    functions = core.parse_csv_list(args.functions); dims = core.parse_int_list(args.dimensions)
    if any(d > 10 for d in dims): p.error("Naive BO is capped at <=10D; use --dimensions 5,10 or remove bo in code.")
    workers_list = core.parse_int_list(args.worker_counts)
    delays = core.parse_float_list(args.delays_ms)
    if args.repeats < 1 or args.budget_multiplier <= 0 or args.max_runtime_hours <= 0: p.error("Repeats, budget and time must be positive.")
    # Warm numerical-library imports before recording per-method timings. GP fitting itself
    # remains part of the measured BO/TuRBO cost.
    from scipy.optimize import minimize as _warm_minimize
    from scipy.stats import qmc as _warm_qmc, norm as _warm_norm
    from sklearn.gaussian_process import GaussianProcessRegressor as _warm_gp
    from sklearn.gaussian_process.kernels import ConstantKernel as _warm_ck, Matern as _warm_matern
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.max_runtime_hours * 3600
    rows, traces = [], []
    timed_out = False
    for fname in functions:
      for dim in dims:
       for delay in delays:
        f, _, lo, hi = core.make_synthetic_problem(fname, dim, core.stable_seed(args.seed, fname, dim))
        # Use one shared initial point for all methods/repeats at this function/dimension.
        _, x0, _, _ = core.make_synthetic_problem(fname, dim, core.stable_seed(args.seed, fname, dim, "start"))
        budget = max(8, int(np.ceil(dim * args.budget_multiplier)))
        for repeat in range(args.repeats):
          if time.perf_counter() >= deadline: timed_out = True; break
          # Paired optimizer seed is fixed across worker counts; worker scheduling can
          # still change completion order for stochastic/noisy extensions.
          for alg in ("cma", "turbo", "random"):
            for w in workers_list:
              if time.perf_counter() >= deadline: timed_out = True; break
              seed = core.stable_seed(args.seed, fname, dim, repeat, alg)
              res, elapsed, backend, effective, events = run_method(alg, f, x0, lo, hi, budget, seed, w, delay, args, deadline)
              row = {"experiment":"cost_parallelism", "function":fname, "dimension":dim, "repeat":repeat+1,
                     "algorithm":alg, "algorithm_backend":backend, "configured_workers":w,
                     "effective_workers":effective, "parallel_supported":True, "evaluation_delay_ms":delay,
                     "budget":budget, "evaluations":res.evaluations, "best_value":float(f(res.best_x)),
                     "best_observed_value":res.best_value, "wall_seconds":elapsed,
                     "target":args.target,
                     "target_hit":bool(any(ev["value"] <= args.target for ev in events)),
                     "time_to_target_seconds":min((ev["elapsed_seconds"] for ev in events if ev["value"] <= args.target), default=""),
                     "evaluations_completed_by_target":sum(1 for ev in events if ev["elapsed_seconds"] <= min((hit["elapsed_seconds"] for hit in events if hit["value"] <= args.target), default=float("inf"))) if any(ev["value"] <= args.target for ev in events) else "",
                     "evaluations_per_second":res.evaluations/max(elapsed,1e-12), "seed":seed,
                     "stop_reason":res.stop_reason, "best_x":res.best_x}
              rows.append(row)
              for pt in res.trace:
                traces.append({"function":fname,"dimension":dim,"repeat":repeat+1,"algorithm":alg,
                    "configured_workers":w,"effective_workers":effective,"evaluation_delay_ms":delay,**pt})
              print(f"{fname:10s} d={dim:2d} delay={delay:5.1f}ms {alg:5s} workers={w} "
                    f"evals={res.evaluations:3d} time={elapsed:.2f}s f={row['best_value']:.4g}")
            if timed_out: break
          if timed_out: break
          # Naive BO runs once per condition with effective worker count one; this is
          # intentional and explicit rather than repeating the same serial experiment
          # for worker counts it cannot use.
          if time.perf_counter() >= deadline: timed_out = True; break
          seed = core.stable_seed(args.seed, fname, dim, repeat, "bo")
          res, elapsed, backend, effective, events = run_method("bo", f, x0, lo, hi, budget, seed, 1, delay, args, deadline)
          row = {"experiment":"cost_parallelism", "function":fname, "dimension":dim, "repeat":repeat+1,
                 "algorithm":"bo", "algorithm_backend":backend, "configured_workers":1,
                 "effective_workers":1, "parallel_supported":False, "evaluation_delay_ms":delay,
                 "budget":budget, "evaluations":res.evaluations, "best_value":float(f(res.best_x)),
                 "best_observed_value":res.best_value, "wall_seconds":elapsed,
                 "target":args.target,
                 "target_hit":bool(any(ev["value"] <= args.target for ev in events)),
                 "time_to_target_seconds":min((ev["elapsed_seconds"] for ev in events if ev["value"] <= args.target), default=""),
                 "evaluations_completed_by_target":sum(1 for ev in events if ev["elapsed_seconds"] <= min((hit["elapsed_seconds"] for hit in events if hit["value"] <= args.target), default=float("inf"))) if any(ev["value"] <= args.target for ev in events) else "",
                 "evaluations_per_second":res.evaluations/max(elapsed,1e-12), "seed":seed,
                 "stop_reason":res.stop_reason, "best_x":res.best_x}
          rows.append(row)
          for pt in res.trace:
            traces.append({"function":fname,"dimension":dim,"repeat":repeat+1,"algorithm":"bo",
                "configured_workers":1,"effective_workers":1,"evaluation_delay_ms":delay,**pt})
          print(f"{fname:10s} d={dim:2d} delay={delay:5.1f}ms bo    workers=1 (sequential) time={elapsed:.2f}s f={row['best_value']:.4g}")
        if timed_out: break
       if timed_out: break
      if timed_out: break
    # Calculate speedup relative to the same method/function/dimension/delay/repeat at worker=1.
    lookup = {(r["function"],r["dimension"],r["repeat"],r["algorithm"],r["evaluation_delay_ms"],r["configured_workers"]):r for r in rows}
    for r in rows:
      base = lookup.get((r["function"],r["dimension"],r["repeat"],r["algorithm"],r["evaluation_delay_ms"],1))
      if r["algorithm"] == "bo":
        # BO is measured only as a sequential baseline; its value of 1.0 would be tautological.
        r["speedup_vs_1_worker"] = ""
        r["quality_change_vs_1_worker"] = ""
      else:
        r["speedup_vs_1_worker"] = (float(base["wall_seconds"])/max(float(r["wall_seconds"]),1e-12)) if base else ""
        r["quality_change_vs_1_worker"] = (float(r["best_value"])-float(base["best_value"])) if base else ""
    core.write_csv(out/"runs.csv", rows); core.write_csv(out/"traces.csv", traces)
    core.write_csv(out/"summary.csv", summarize_rows(rows))
    print(f"Saved {len(rows)} runs to {out.resolve()}; deadline reached={timed_out}.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
