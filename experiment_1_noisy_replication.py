#!/usr/bin/env python3
"""Experiment 1: allocation between new points and replicate noisy evaluations.
Includes CMA-ES, TuRBO, naive BO and non-adaptive random search as a baseline.

Requires experiment_jerrell_rewritten.py in the same directory plus its dependencies.
The raw black-box budget is fixed; repeat_count=r means each proposed location is
measured r times and the optimizer receives the replicate mean. Final recommendation
rules are assessed against the known noise-free objective, and fresh validation calls
are reported separately from the optimization budget.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import numpy as np
import experiment_jerrell_rewritten as core


def sample_noise(rng, model: str, sigma: float) -> float:
    if model == "gaussian":
        return float(rng.normal(0.0, sigma))
    # t(df=3)/sqrt(3) has unit variance.
    return float(rng.standard_t(3) / math.sqrt(3.0) * sigma)


class ReplicatedNoisyObjective:
    """Callable returning replicate mean while retaining every raw observation."""
    def __init__(self, f, model: str, sigma: float, reps: int, seed: int):
        self.f, self.model, self.sigma, self.reps = f, model, float(sigma), int(reps)
        self.rng = np.random.default_rng(seed)
        self.records: list[dict] = []

    def __call__(self, x):
        x = np.asarray(x, dtype=float).copy()
        truth = float(self.f(x))
        obs = [truth + sample_noise(self.rng, self.model, self.sigma) for _ in range(self.reps)]
        rec = {"x": x, "truth": truth, "observations": np.asarray(obs), "mean": float(np.mean(obs)),
               "minimum_observation": float(np.min(obs))}
        self.records.append(rec)
        return rec["mean"]


def get_problem(name: str, dim: int, seed: int):
    f, x0, lo, hi = core.make_synthetic_problem(name, dim, seed)
    rng = np.random.default_rng(core.stable_seed("replication-noise-scale", name, dim))
    xs = rng.uniform(lo, hi, (256, dim))
    scale = max(1.0, float(np.median(np.abs([f(x) for x in xs]))))
    return f, x0, lo, hi, scale


def recommend_points(records: list[dict], rule: str, f_true, model: str, sigma: float, seed: int,
                     validation_reps: int = 10, shortlist: int = 5):
    if not records:
        raise RuntimeError("No observations were collected.")
    if rule == "lowest_single_observation":
        i, j = min(((i, j) for i, rec in enumerate(records) for j in range(len(rec["observations"]))),
                   key=lambda pair: records[pair[0]]["observations"][pair[1]])
        x = records[i]["x"].copy()
        selected_score = float(records[i]["observations"][j])
        validation_n = 0
    elif rule == "lowest_replicate_mean":
        rec = min(records, key=lambda r: r["mean"])
        x, selected_score, validation_n = rec["x"].copy(), float(rec["mean"]), 0
    elif rule == "top5_fresh_validation":
        candidates = sorted(records, key=lambda r: r["mean"])[:min(shortlist, len(records))]
        # Fresh independent validation; these oracle-like noisy checks are reported separately
        # and do not consume the optimizer's raw-evaluation budget.
        rng = np.random.default_rng(seed)
        ranked = []
        for rec in candidates:
            truth = float(f_true(rec["x"]))
            vals = [truth + sample_noise(rng, model, sigma) for _ in range(validation_reps)]
            ranked.append((float(np.mean(vals)), rec["x"].copy()))
        selected_score, x = min(ranked, key=lambda t: t[0])
        validation_n = validation_reps * len(candidates)
    else:
        raise ValueError(rule)
    return x, selected_score, float(f_true(x)), validation_n


def run_one_algorithm(algorithm, noisy_obj, x0, lo, hi, point_budget, seed, args, obs_sigma, deadline=None):
    # The optimizers count *locations*. Each location costs args.repeat_counts raw evaluations.
    if algorithm == "cma":
        return core.run_cma(noisy_obj, x0, lo, hi, point_budget, seed,
                            workers=1, sigma_scale=args.cma_sigma_scale, backend=args.cma_backend, deadline=deadline)
    if algorithm == "turbo":
        return core.run_turbo(noisy_obj, x0, lo, hi, point_budget, seed,
                              batch_size=args.turbo_batch_size,
                              candidate_count=args.turbo_candidates,
                              rff_features=args.turbo_rff_features,
                              max_train_points=args.turbo_max_train,
                              fit_every_batches=args.turbo_fit_every,
                              gp_maxiter=args.turbo_gp_maxiter, deadline=deadline,
                              observation_noise_std=obs_sigma / math.sqrt(args.current_reps))
    if algorithm == "bo":
        return core.run_bo_sklearn(noisy_obj, x0, lo, hi, point_budget, seed,
                                   workers=1, bo_startup=min(args.bo_startup, max(2, point_budget // 2)),
                                   observation_noise_std=obs_sigma / math.sqrt(args.current_reps))
    if algorithm == "random":
        # Random search is the non-adaptive baseline. As with the other methods,
        # each proposed location returns the mean of args.current_reps raw noisy calls.
        return core.run_random(noisy_obj, x0, lo, hi, point_budget, seed,
                               workers=1, deadline=deadline)
    raise ValueError(algorithm)


def summarize_rows(rows: list[dict]) -> list[dict]:
    keys = ("function", "dimension", "algorithm", "algorithm_backend", "noise_model", "noise_level",
            "repeat_count_per_candidate", "recommendation_rule", "raw_budget", "unique_point_budget")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out = []
    for k, group in sorted(groups.items(), key=lambda z: tuple(str(v) for v in z[0])):
        true_values = np.asarray([float(r["recommendation_true_value"]) for r in group])
        times = np.asarray([float(r["wall_seconds"]) for r in group])
        obs = np.asarray([float(r["best_observed_value"]) for r in group])
        out.append({**dict(zip(keys, k)), "runs": len(group),
                    "recommendation_true_value_mean": float(np.mean(true_values)),
                    "recommendation_true_value_median": float(np.median(true_values)),
                    "recommendation_true_value_std": float(np.std(true_values, ddof=1)) if len(group)>1 else 0.0,
                    "success_rate": float(np.mean([bool(r["success"]) for r in group])),
                    "best_observed_value_median": float(np.median(obs)),
                    "wall_seconds_median": float(np.median(times)),
                    "raw_optimizer_evaluations_mean": float(np.mean([r["raw_optimizer_evaluations"] for r in group])),
                    "validation_evaluations_separate_mean": float(np.mean([r["validation_evaluations_separate"] for r in group]))})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", default="sphere,rastrigin")
    p.add_argument("--dimensions", default="5,10", help="Keep naive BO at 10D or lower by default.")
    p.add_argument("--algorithms", default="cma,turbo,bo,random", help="Subset of cma,turbo,bo,random; random is the non-adaptive baseline.")
    p.add_argument("--budget-multiplier", type=float, default=20.0, help="Raw noisy evaluations = ceil(d*multiplier).")
    p.add_argument("--repeat-counts", default="1,2,4", help="Replicates per proposed location; consumes raw budget.")
    p.add_argument("--noise-models", default="gaussian,student_t")
    p.add_argument("--noise-levels", default="0.05,0.2", help="Relative noise standard deviations.")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--bo-startup", type=int, default=8)
    p.add_argument("--cma-backend", choices=("auto", "pycma", "numpy"), default="auto")
    p.add_argument("--cma-sigma-scale", type=float, default=0.3)
    p.add_argument("--target", type=float, default=1e-2)
    p.add_argument("--validation-reps", type=int, default=10)
    p.add_argument("--max-runtime-hours", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="results/noisy_replication")
    p.add_argument("--turbo-batch-size", type=int, default=5)
    p.add_argument("--turbo-candidates", type=int, default=256)
    p.add_argument("--turbo-rff-features", type=int, default=32)
    p.add_argument("--turbo-max-train", type=int, default=120)
    p.add_argument("--turbo-fit-every", type=int, default=8)
    p.add_argument("--turbo-gp-maxiter", type=int, default=4)
    args = p.parse_args()
    functions = core.parse_csv_list(args.functions); dims = core.parse_int_list(args.dimensions)
    algorithms = core.parse_csv_list(args.algorithms)
    bad = set(algorithms) - {"cma", "turbo", "bo", "random"}
    if bad: p.error(f"Unsupported algorithms: {sorted(bad)}")
    if any(d > 10 for d in dims) and "bo" in algorithms:
        p.error("Naive BO is limited to <=10D in this study; remove bo or lower --dimensions.")
    reps_options = core.parse_int_list(args.repeat_counts)
    if any(r not in (1, 2, 4, 8) for r in reps_options): p.error("--repeat-counts supports 1,2,4,8.")
    noise_models = core.parse_csv_list(args.noise_models)
    if set(noise_models) - {"gaussian", "student_t"}: p.error("Noise models: gaussian,student_t")
    noise_levels = core.parse_float_list(args.noise_levels, allow_zero=False)
    if args.repeats < 1 or args.budget_multiplier <= 0 or args.max_runtime_hours <= 0: p.error("Repeats, budget multiplier and time limit must be positive.")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.max_runtime_hours * 3600.0
    rows, traces = [], []
    recommendation_rules = ("lowest_single_observation", "lowest_replicate_mean", "top5_fresh_validation")
    timed_out = False
    for fname in functions:
      for dim in dims:
       for model in noise_models:
        for level in noise_levels:
         f_true, x0, lo, hi, ref_scale = get_problem(fname, dim, args.seed)
         sigma = float(level * ref_scale)
         raw_budget = max(8, int(math.ceil(dim * args.budget_multiplier)))
         for rcount in reps_options:
          point_budget = raw_budget // rcount
          if point_budget < 8 and "turbo" in algorithms:
              print(f"Skip {fname} d={dim} reps/point={rcount}: only {point_budget} unique points, TuRBO needs >=8.")
              continue
          for repeat in range(args.repeats):
           if time.perf_counter() >= deadline: timed_out = True; break
           for algorithm in algorithms:
            if time.perf_counter() >= deadline: timed_out = True; break
            # Paired optimizer seed across repeat counts: each strategy starts from the
            # same point/design when budget allows; extra replicate draws are nested from
            # the same deterministic noise stream.
            seed = core.stable_seed(args.seed, fname, dim, model, level, repeat, algorithm, "optimizer")
            noisy_obj = ReplicatedNoisyObjective(f_true, model, sigma, rcount,
                                                   core.stable_seed(args.seed, fname, dim, model, level, repeat, algorithm, "noise"))
            args.current_reps = rcount
            t0 = time.perf_counter()
            try:
              result = run_one_algorithm(algorithm, noisy_obj, x0, lo, hi, point_budget, seed, args, sigma, deadline)
            except Exception as exc:
              print(f"ERROR algorithm={algorithm} f={fname} d={dim} reps={rcount}: {exc}")
              raise
            elapsed = time.perf_counter() - t0
            raw_used = result.evaluations * rcount
            for ri, rec in enumerate(noisy_obj.records, start=1):
                traces.append({"function": fname, "dimension": dim, "noise_model": model,
                    "noise_level": level, "repeat_count": rcount, "repeat": repeat + 1,
                    "algorithm": algorithm, "candidate_index": ri,
                    "replicate_mean": rec["mean"], "minimum_raw_observation": rec["minimum_observation"],
                    "true_value_at_candidate": rec["truth"], "raw_observations": rec["observations"],
                    "x": rec["x"]})
            for rule in recommendation_rules:
              rec_seed = core.stable_seed(seed, rule, "validation")
              x, selection_score, true_value, val_calls = recommend_points(
                  noisy_obj.records, rule, f_true, model, sigma, rec_seed,
                  validation_reps=args.validation_reps)
              backend_by_algorithm = {
                  "bo": "sklearn-GP",
                  "turbo": "TuRBO-1",
                  "random": "numpy-random-search",
                  "cma": ("pycma" if core.cma is not None and args.cma_backend != "numpy" else "numpy-CMA-ES-fallback"),
              }
              row = {"experiment": "noisy_replication", "function": fname, "dimension": dim,
                  "algorithm": algorithm, "algorithm_backend": backend_by_algorithm[algorithm],
                  "noise_model": model, "noise_level": level, "noise_std": sigma,
                  "repeat_count_per_candidate": rcount, "raw_budget": raw_budget,
                  "unique_point_budget": point_budget, "unique_points_evaluated": result.evaluations,
                  "raw_optimizer_evaluations": raw_used, "validation_evaluations_separate": val_calls,
                  "recommendation_rule": rule, "selection_score_noisy": selection_score,
                  "recommendation_true_value": true_value, "best_value": true_value,
                  "best_observed_value": min(float(np.min(rec["observations"])) for rec in noisy_obj.records),
                  "success": bool(true_value <= args.target), "target": args.target,
                  "repeat": repeat + 1, "seed": seed, "wall_seconds": elapsed,
                  "evaluations": raw_used, "stop_reason": result.stop_reason}
              rows.append(row)
            print(f"{fname:10s} d={dim:2d} noise={model}/{level:g} reps={rcount} {algorithm:5s} "
                  f"raw={raw_used:4d}/{raw_budget} time={elapsed:.2f}s true-best={min(r['truth'] for r in noisy_obj.records):.4g}")
           if timed_out: break
          if timed_out: break
         if timed_out: break
        if timed_out: break
       if timed_out: break
      if timed_out: break
    core.write_csv(out / "runs.csv", rows)
    core.write_csv(out / "candidate_traces.csv", traces)
    core.write_csv(out / "summary.csv", summarize_rows(rows))
    print(f"Saved {len(rows)} recommendation rows from {len(traces)} candidate records to {out.resolve()}. "
          f"Time cap reached={timed_out}.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
