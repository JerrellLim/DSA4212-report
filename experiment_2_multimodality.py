#!/usr/bin/env python3
"""Experiment 2: population/restarts, BO acquisitions, and basin-hit probability.

Requires experiment_jerrell_rewritten.py beside this file. CMA population and IPOP
restart sweeps require pycma (`pip install cma`). Naive GP-BO is intentionally capped
at 10 dimensions to prevent GP overhead dominating the experiment.
"""
from __future__ import annotations
import argparse, math, time, warnings
from pathlib import Path
import numpy as np
import experiment_jerrell_rewritten as core


def make_problem(name: str, dim: int, seed: int):
    """Return objective, start, bounds, legacy objective threshold, and true minimizer."""
    if name != "narrow_basin":
        f, x0, lo, hi = core.make_synthetic_problem(name, dim, seed)
        target = 1.0 if name == "rastrigin" else 0.5 if name == "ackley" else 1e-2
        # Keep this shift construction synchronized with core.make_synthetic_problem.
        shift_rng = np.random.default_rng(core.stable_seed("shift", name, dim))
        global_opt = shift_rng.uniform(-1.5, 1.5, size=dim)
        return f, x0, lo, hi, target, global_opt
    lo, hi = np.zeros(dim), np.ones(dim)
    local = np.full(dim, 0.20)
    global_opt = np.full(dim, 0.80)
    x0 = local.copy()
    def f(x):
        x = np.asarray(x, float)
        broad = 0.60 + 0.02 * float(np.sum((x - local) ** 2))
        narrow = 10.0 * float(np.sum((x - global_opt) ** 2))
        return min(broad, narrow)
    return f, x0, lo, hi, 0.59, global_opt


def is_global_basin_narrow_basin(x: np.ndarray) -> bool:
    """True exactly when the narrow/global branch is the active branch of f."""
    x = np.asarray(x, dtype=float)
    local = np.full(x.size, 0.20)
    global_opt = np.full(x.size, 0.80)
    broad = 0.60 + 0.02 * float(np.sum((x - local) ** 2))
    narrow = 10.0 * float(np.sum((x - global_opt) ** 2))
    return bool(narrow < broad)


def run_bo_acquisition(f, x0, lo, hi, budget: int, seed: int, acquisition: str,
                       startup: int = 8, candidate_count: int = 2048, deadline: float | None = None):
    """Sequential naive GP-BO with EI, PI, or GP-LCB/UCB acquisition."""
    from scipy.stats import norm, qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.exceptions import ConvergenceWarning
    started = time.perf_counter()
    objective = core.CountingObjective(f)
    x0 = np.clip(np.asarray(x0, float), lo, hi)
    width = hi - lo
    xs = [x0.copy()]; ys = [objective(x0)]
    n_start = min(max(2, startup), max(2, budget // 2), budget)
    need = n_start - 1
    if need:
        power = int(math.ceil(math.log2(need)))
        unit = qmc.Sobol(d=x0.size, scramble=True, seed=seed).random_base2(power)[:need]
        X0 = lo + unit * width
        ys0 = core.evaluate_batch(objective, X0, 1)
        xs.extend(np.asarray(x) for x in X0); ys.extend(map(float, ys0))
    dim = x0.size
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(dim, 0.2), length_scale_bounds=(0.03, 5.0), nu=2.5)
    iteration = 0
    while len(ys) < budget and (deadline is None or time.perf_counter() < deadline):
        X = (np.asarray(xs) - lo) / width
        y = np.asarray(ys)
        tune = iteration == 0 or iteration % 5 == 0
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
               optimizer="fmin_l_bfgs_b" if tune else None, n_restarts_optimizer=0,
               random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning); gp.fit(X, y)
        if deadline is not None and time.perf_counter() >= deadline:
            break
        kernel = gp.kernel_
        n_cand = max(512, min(int(candidate_count), 2048, 128 * dim))
        power = int(math.ceil(math.log2(n_cand)))
        C = qmc.Sobol(dim, scramble=True, seed=core.stable_seed(seed, "candidate-pool", iteration)).random_base2(power)[:n_cand]
        mean, std = gp.predict(C, return_std=True); std = np.maximum(std, 1e-12)
        incumbent = float(np.min(y)); xi = 0.01 * max(float(np.std(y)), 1e-12)
        z = (incumbent - mean - xi) / std
        if acquisition == "EI":
            scores = (incumbent - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            candidate_idx = int(np.argmax(scores))
        elif acquisition == "PI":
            scores = norm.cdf(z)
            candidate_idx = int(np.argmax(scores))
        elif acquisition == "UCB":
            # Minimization LCB: select smallest posterior mean - kappa * std.
            kappa = 2.0
            candidate_idx = int(np.argmin(mean - kappa * std))
        else:
            raise ValueError(acquisition)
        d2 = np.sum((C[candidate_idx][None, :] - X) ** 2, axis=1)
        if np.min(d2) < 1e-12:
            candidates_d2 = np.min(np.sum((C[:, None, :] - X[None, :, :]) ** 2, axis=2), axis=1)
            candidate_idx = int(np.argmax(candidates_d2))
        x_new = lo + C[candidate_idx] * width
        y_new = objective(x_new)
        xs.append(x_new); ys.append(float(y_new)); iteration += 1
    best_i = int(np.argmin(ys))
    return core.RunResult("bo", np.asarray(xs[best_i]), float(ys[best_i]), objective.count,
                          time.perf_counter() - started, objective.trace, "evaluation_budget")


def run_cma_variant_fixed(f, x0, lo, hi, budget: int, seed: int, popsize: int,
                         restarts: int, sigma_scale: float = 0.3,
                         deadline: float | None = None):
    """Run fixed-population CMA with independently seeded random restarts.

    The per-restart budget is split evenly and rounded down to complete generations.
    Actual evaluations are recorded, so any remainder is transparent rather than
    silently credited as used budget. The first point of every restart is evaluated
    once and reused as member zero of its first generation.
    """
    if core.cma is None:
        raise RuntimeError("Population/restart sweeps require pycma. Install with: python -m pip install cma")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    objective = core.CountingObjective(f)
    dim = len(x0)
    best_x = np.asarray(x0, float).copy()
    best_value = float("inf")
    n_runs = int(restarts) + 1
    chunks = [budget // n_runs] * n_runs
    for i in range(budget % n_runs):
        chunks[i] += 1
    for restart_idx, chunk_budget in enumerate(chunks):
        if chunk_budget < popsize:
            continue
        start = np.asarray(x0, float).copy() if restart_idx == 0 else rng.uniform(lo, hi)
        start_value = float(objective(start))
        if start_value < best_value:
            best_value, best_x = start_value, start.copy()
        options = {"seed": int(core.stable_seed(seed, "restart", restart_idx)),
                   "popsize": int(popsize), "maxfevals": int(chunk_budget),
                   "bounds": [lo.tolist(), hi.tolist()], "verbose": -9,
                   "verb_disp": 0, "verb_log": 0}
        es = core.cma.CMAEvolutionStrategy(start.tolist(), sigma_scale * float(np.median(hi-lo)), options)
        used = 1
        first_generation = True
        while used < chunk_budget and (deadline is None or time.perf_counter() < deadline) and (chunk_budget - used >= (popsize - 1 if first_generation else popsize)) and not es.stop():
            candidates = [np.asarray(z, float) for z in es.ask()]
            if first_generation:
                candidates[0] = start.copy()
                vals = [start_value]
                new_points = candidates[1:]
                new_values = core.evaluate_batch(objective, new_points, 1)
                vals.extend(map(float, new_values))
                used += len(new_points)
                first_generation = False
            else:
                vals = list(map(float, core.evaluate_batch(objective, candidates, 1)))
                used += len(candidates)
            if len(vals) != popsize:
                break
            es.tell(candidates, vals)
            idx = int(np.argmin(vals))
            if vals[idx] < best_value:
                best_value, best_x = float(vals[idx]), candidates[idx].copy()
    return core.RunResult("cma", best_x, best_value, objective.count,
                          time.perf_counter()-started, objective.trace, "evaluation_budget_or_restart_budget")


def parse_nonnegative_ints(value: str) -> list[int]:
    try:
        vals = [int(p.strip()) for p in value.split(",") if p.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc
    if not vals or any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("Expected comma-separated nonnegative integers.")
    return vals


def summarize_rows(rows: list[dict]) -> list[dict]:
    keys = ("function", "dimension", "algorithm", "algorithm_backend", "cma_population_size",
            "cma_population_multiplier", "cma_restarts", "bo_acquisition", "budget",
            "distance_tolerance", "success_metric")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out = []
    for k, group in sorted(groups.items(), key=lambda z: tuple(str(v) for v in z[0])):
        vals = np.asarray([float(r["best_value"]) for r in group]); times=np.asarray([float(r["wall_seconds"]) for r in group])
        out.append({**dict(zip(keys,k)), "runs":len(group), "best_value_mean":float(np.mean(vals)),
                    "best_value_median":float(np.median(vals)),
                    "best_value_std":float(np.std(vals,ddof=1)) if len(group)>1 else 0.0,
                    "distance_success_rate":float(np.mean([bool(r["distance_success"]) for r in group])),
                    "global_basin_hit_rate":(float(np.mean([bool(r["global_basin_hit"]) for r in group]))
                                              if group[0]["function"] == "narrow_basin" else ""),
                    "legacy_objective_success_rate":float(np.mean([bool(r["legacy_objective_success"]) for r in group])),
                    "distance_linf_median":float(np.median([float(r["distance_linf"]) for r in group])),
                    "wall_seconds_median":float(np.median(times)),
                    "evaluations_mean":float(np.mean([r["evaluations"] for r in group]))})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", default="rastrigin,narrow_basin")
    p.add_argument("--dimensions", default="2,5")
    p.add_argument("--budget-multiplier", type=float, default=24.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--distance-tolerance", type=float, default=0.1,
                   help="Fixed L-infinity distance tolerance to the known global minimizer (coordinate units).")
    p.add_argument("--bo-startup", type=int, default=8)
    p.add_argument(
        "--algorithms",
        default="cma,turbo,random,bo",
        help=("Comma-separated algorithms to run: cma,turbo,random,bo. "
              "BO runs EI, PI, and UCB variants. Use --algorithms cma,turbo,random to skip BO."),
    )
    p.add_argument("--cma-pop-multipliers", default="1,2,4")
    p.add_argument("--cma-restart-counts", default="0,2")
    p.add_argument("--max-runtime-hours", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="results/multimodality")
    p.add_argument("--turbo-candidates", type=int, default=256)
    p.add_argument("--turbo-rff-features", type=int, default=32)
    p.add_argument("--turbo-max-train", type=int, default=120)
    p.add_argument("--turbo-fit-every", type=int, default=8)
    p.add_argument("--turbo-gp-maxiter", type=int, default=4)
    args = p.parse_args()
    functions = core.parse_csv_list(args.functions); dims = core.parse_int_list(args.dimensions)
    if args.budget_multiplier <= 0 or args.repeats < 1 or args.max_runtime_hours <= 0: p.error("Budget/repeats/time must be positive.")
    if args.distance_tolerance <= 0: p.error("--distance-tolerance must be positive.")
    algorithms = [item.strip().lower() for item in args.algorithms.split(",") if item.strip()]
    valid_algorithms = {"cma", "turbo", "random", "bo"}
    invalid_algorithms = sorted(set(algorithms) - valid_algorithms)
    if invalid_algorithms:
        p.error(f"Unknown algorithm(s) in --algorithms: {', '.join(invalid_algorithms)}. "
                "Choose from cma,turbo,random,bo.")
    if not algorithms:
        p.error("--algorithms must include at least one algorithm.")
    # Preserve the user's order only for clarity; avoid duplicate algorithm specs.
    algorithms = list(dict.fromkeys(algorithms))

    multipliers = core.parse_int_list(args.cma_pop_multipliers)
    restart_counts = parse_nonnegative_ints(args.cma_restart_counts)
    if 0 not in restart_counts: restart_counts = [0] + restart_counts
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.max_runtime_hours * 3600
    rows, trace_rows = [], []
    timed_out = False
    if "cma" in algorithms and core.cma is None:
        print("WARNING: pycma is not installed; CMA-ES will be skipped. Install it with `python -m pip install cma` for population and restart sweeps.")
    print("Algorithms requested: " + ", ".join(algorithms))
    for fname in functions:
      for dim in dims:
       if fname == "narrow_basin" and dim > 5:
           print(f"Skip narrow_basin d={dim}: this deliberately narrow target is calibrated for d<=5."); continue
       for repeat in range(args.repeats):
        if time.perf_counter() >= deadline: timed_out = True; break
        f, x0, lo, hi, target, global_opt = make_problem(fname, dim, core.stable_seed(args.seed, fname, dim))
        budget = max(20, int(math.ceil(dim * args.budget_multiplier)))
        base_pop = max(4, 2 * dim)
        specs = []
        seen = set()
        if "cma" in algorithms:
            for pm in multipliers:
                key = ("cma", pm, 0)
                if key not in seen: specs.append(("cma", pm, 0, "")); seen.add(key)
            for restarts in restart_counts:
                key = ("cma", 1, restarts)
                if key not in seen: specs.append(("cma", 1, restarts, "")); seen.add(key)
        if "turbo" in algorithms:
            specs.append(("turbo", 1, 0, ""))
        if "random" in algorithms:
            specs.append(("random", 1, 0, ""))
        if "bo" in algorithms:
            if dim <= 10:
                specs += [("bo", 1, 0, acq) for acq in ("EI", "PI", "UCB")]
            else:
                print(f"Skip naive BO at d={dim}; dimension cap is 10.")
        if core.cma is None:
            specs = [spec for spec in specs if spec[0] != "cma"]
        for alg, pm, restart_count, acq in specs:
            if time.perf_counter() >= deadline: timed_out = True; break
            if alg == "cma":
                seed = core.stable_seed(args.seed, fname, dim, repeat, "cma-common")
            elif alg == "bo":
                # Same initial design and candidate pool across EI/PI/UCB; only acquisition differs.
                seed = core.stable_seed(args.seed, fname, dim, repeat, "bo-acquisition-common")
            else:
                seed = core.stable_seed(args.seed, fname, dim, repeat, alg)
            started = time.perf_counter()
            if alg == "cma":
                pop = max(4, base_pop * pm)
                try:
                    res = run_cma_variant_fixed(f, x0, lo, hi, budget, seed, pop, restart_count, deadline=deadline)
                except Exception as exc:
                    print(f"CMA configuration failed ({fname},d={dim},pop={pop},restarts={restart_count}): {exc}")
                    raise
                backend = "pycma-fixed-population-restarts" if restart_count else "pycma-popsize-sweep"
            elif alg == "turbo":
                if budget < 8: continue
                res = core.run_turbo(f, x0, lo, hi, budget, seed, batch_size=5,
                    candidate_count=args.turbo_candidates, rff_features=args.turbo_rff_features,
                    max_train_points=args.turbo_max_train, fit_every_batches=args.turbo_fit_every,
                    gp_maxiter=args.turbo_gp_maxiter, deadline=deadline)
                backend = "TuRBO-1-local-GP-RFF-TS"
            elif alg == "bo":
                res = run_bo_acquisition(f, x0, lo, hi, budget, seed, acq, args.bo_startup, deadline=deadline)
                backend = f"sklearn-GP-{acq}"
            else:
                res = core.run_random(f, x0, lo, hi, budget, seed, deadline=deadline)
                backend = "uniform-random"
            val = float(f(res.best_x))
            distance_linf = float(np.linalg.norm(np.asarray(res.best_x, float) - global_opt, ord=np.inf))
            distance_success = bool(distance_linf <= args.distance_tolerance)
            global_basin_hit = is_global_basin_narrow_basin(res.best_x) if fname == "narrow_basin" else ""
            success_metric = "linf_distance_to_global_minimizer"
            row = {"experiment":"multimodality", "function":fname, "dimension":dim, "repeat":repeat+1,
                   "algorithm":alg, "algorithm_backend":backend, "cma_population_size":(base_pop*pm if alg=="cma" else ""),
                   "cma_population_multiplier":(pm if alg=="cma" else ""), "cma_restarts":(restart_count if alg=="cma" else ""),
                   "bo_acquisition":acq, "budget":budget, "evaluations":res.evaluations,
                   "best_value":val,
                   # `success` always means the same dimension-fair L-infinity proximity test.
                   # Direct basin membership for narrow_basin is reported separately below.
                   "success":distance_success,
                   "success_metric":success_metric,
                   "distance_linf":distance_linf, "distance_tolerance":args.distance_tolerance,
                   "distance_success":distance_success, "global_basin_hit":global_basin_hit,
                   "legacy_objective_success":bool(val <= target), "legacy_objective_target":target,
                   "wall_seconds":time.perf_counter()-started, "seed":seed, "stop_reason":res.stop_reason,
                   "best_x":res.best_x}
            rows.append(row)
            for point in res.trace: trace_rows.append({"function":fname,"dimension":dim,"repeat":repeat+1,
                "algorithm":alg,"algorithm_backend":backend,"cma_population_size":row["cma_population_size"],
                "cma_restarts":row["cma_restarts"],"bo_acquisition":acq,
                "distance_tolerance":args.distance_tolerance,**point})
            print(f"{fname:12s} d={dim:2d} {alg:5s} {backend:20s} f={val:.5g} "
                  f"Linf={distance_linf:.3g} success={row['success']} t={row['wall_seconds']:.2f}s")
        if timed_out: break
       if timed_out: break
      if timed_out: break
    core.write_csv(out/"runs.csv", rows); core.write_csv(out/"traces.csv", trace_rows)
    core.write_csv(out/"summary.csv", summarize_rows(rows))
    print(f"Saved {len(rows)} runs to {out.resolve()}; timed_out={timed_out}; elapsed={time.perf_counter() - (deadline-args.max_runtime_hours*3600):.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
