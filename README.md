# CMA-ES vs Bayesian optimisation: experiment runner

This rewritten runner compares three methods:

- **CMA-ES (`cma`)** — the maintained `cma` / pycma implementation using the standard ask–tell interface.
- **Gaussian-process Bayesian optimisation (`bo`)** — by default, a lower-overhead scikit-learn Gaussian process with a Matérn-5/2 kernel and expected improvement (EI) maximised over a Sobol candidate pool. Kernel hyperparameters are optimised every five model-based suggestions and reused between those updates.
- **Parallel BO backend** — `--bo-backend optuna` selects Optuna's `GPSampler`, which supports asynchronous parallel trials. This is useful for Experiment B but usually has more surrogate/acquisition overhead on cheap functions.
- **Random search (`random`)** — a useful non-adaptive baseline. It replaces BFGS because finite-difference gradients make BFGS an awkward and potentially budget-inconsistent comparator for black-box optimisation.

The script has three modes. `coco` runs COCO/BBOB problems for conventional benchmark results. `synthetic` runs shifted deterministic functions, including worker sweeps. `noisy` runs a controlled noisy-objective suite and scores each returned recommendation using the corresponding known noise-free function.

## 1. Install the dependencies

Use a virtual environment. For example, on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_experiment.txt
```

The default single-worker BO backend uses scikit-learn and does not need PyTorch. Optuna's `GPSampler` requires PyTorch; a CPU build is sufficient. If `pip install -r requirements_experiment.txt` does not select a suitable PyTorch build on your machine, follow the selector at <https://pytorch.org/get-started/locally/>. The COCO Python package is named `coco-experiment` but is imported as `cocoex`.

The old imports `mkl_bugfix` and `cma.fmin_lq_surr2` are no longer used. No `scikit-optimize` install is required.

For repeatable timing on a machine with multiple BLAS/OpenMP thread pools, start PowerShell with:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
```

The script also sets these values to `1` when they have not already been set. The explicit PowerShell assignments override pre-existing values, which matters because the script deliberately does not override user-specified thread settings.

## 2. Start with a small COCO/BBOB pilot

From the directory containing `experiment_jerrell_rewritten.py`:

```powershell
python .\experiment_jerrell_rewritten.py coco `
  --algorithms cma,bo,random `
  --dimensions 2,3 `
  --instances 1-2 `
  --function-indices 1-3 `
  --budget-multiplier 10 `
  --repeats 1 `
  --output-dir results/coco_pilot
```

This is deliberately small: three function indices, two dimensions, two instances and three algorithms. When that works, broaden to (for example) `--dimensions 2,3,5,10 --instances 1-5 --function-indices 1-24`. COCO BBOB functions/instances are re-created for each algorithm/repeat so runs are isolated in their own observer folders.

**Important:** COCO mode requires `--workers 1`. COCO problem objects maintain evaluation counters and observer state; this runner does not call a single problem object concurrently. Use the synthetic experiment for a controlled parallelism comparison.

## 3. Experiment A — multimodality

The synthetic functions are all shifted away from the origin and have a known optimum value of exactly zero:

- `sphere`: unimodal control.
- `rosenbrock`: narrow, curved valley; useful for geometry/conditioning as well as a control against rugged multimodal cases.
- `rastrigin` and `ackley`: multimodal cases.

Run a pilot first:

```powershell
python .\experiment_jerrell_rewritten.py synthetic `
  --algorithms cma,bo,random `
  --functions sphere,rastrigin,ackley `
  --dimensions 2,5 `
  --budget-multiplier 10 `
  --repeats 5 `
  --bo-startup 8 `
  --output-dir results/multimodal_pilot
```

The default `--bo-backend sklearn` avoids expensive PyTorch-based acquisition optimisation in this single-worker comparison. You can select the alternative `--bo-backend optuna` if you want to compare BO implementations, but record that choice in the report. Then increase repeats (for example to 20 or 30), dimensions, and/or budget once the pilot is fast enough. The evaluation budget for each run is `ceil(dimension × budget multiplier)`. It includes the starting point and no unbudgeted final re-evaluation is made. CMA uses the shared initial point as its first population member; BO is given that same observation, followed by a space-filling Sobol startup design. All algorithms share the same initial point for a given function, dimension and repeat.

The default configured success target is `1e-8`, which is a very strict target for a budget of only `10 × dimension` evaluations. Zero successes at that threshold can be completely expected and does not alone mean the implementation is broken. The runner now reports target-sensitivity columns in `summary.csv` as well: success rates for objective values at or below `1e-8`, `1e-4`, `1e-2`, `1e-1`, `1`, `5`, and `10`. Read these alongside `best_value_median` and `best_value_std`; do not select a threshold after seeing results and then present it as if it had been pre-registered. For the final report, state a threshold in advance if you use success rates, and treat final objective value as a primary continuous outcome. These thresholds are absolute, so compare success-rate columns primarily within the same function and dimension. Do not infer a reliable ranking from a single seed. Rastrigin/Ackley versus Sphere help isolate multimodality from general implementation failures; Rosenbrock helps interpret whether a method is also affected by a curved narrow valley.

## 4. Experiment B — parallel workers and evaluation cost

A cheap mathematical function may finish so quickly that scheduling overhead dominates. To model an expensive black-box evaluation, add a controlled delay to each synthetic function call and sweep worker counts:

```powershell
python .\experiment_jerrell_rewritten.py synthetic `
  --algorithms cma,bo,random `
  --functions rastrigin,ackley `
  --dimensions 5 `
  --budget-multiplier 15 `
  --repeats 5 `
  --worker-sweep 1,2,4 `
  --evaluation-delay-ms 25 `
  --bo-startup 8 `
  --bo-backend optuna `
  --output-dir results/parallelism
```

The objective, initial point and optimizer seed are held fixed across worker counts where possible. Each worker setting receives the same function-evaluation budget. Compare both **wall-clock time** and **solution quality/success rate**:

- CMA-ES evaluates the candidates in a generation concurrently, then updates its distribution after the generation completes. Useful parallelism is capped by its population size.
- This command deliberately selects `--bo-backend optuna` because the lightweight sklearn backend is sequential. BO uses asynchronous parallel Optuna trials (`n_jobs`). Workers may ask for points before earlier calls finish, so BO can gain throughput while sacrificing some adaptivity. Optuna does not promise fully reproducible suggestion sequences for parallel `n_jobs > 1`; report this as part of the method, not as random noise-free equivalence. The worker sweep includes `1`, so the serial and parallel runs in this experiment use the same BO backend.
- Random search is a non-adaptive throughput control; with deterministic functions its best value should not materially depend on worker count when it uses the same sampled points.

`--evaluation-delay-ms` is only a throughput simulation, not part of the mathematical objective. It should be reported alongside the results. For a real expensive objective, set it to zero and replace the synthetic objective with the actual evaluator (or use COCO serially).

## 5. Experiment C — noisy objective evaluations

The `noisy` mode runs the same shifted test functions with two zero-mean observation-noise distributions:

- `gaussian`: additive Gaussian noise with standard deviation `sigma`.
- `student_t`: additive Student-t noise with 3 degrees of freedom, scaled to have the same variance as Gaussian noise. This has heavier tails and tests sensitivity to occasional large errors.

The default noise levels are `0,0.05,0.2`. They are relative to a fixed reference scale for each function and dimension:

`noise_std = noise_level × max(1, median(abs(f(x))) over 512 seeded uniform reference points)`.

The scale is computed once per function/dimension, independent of repeat, so a given noise condition has the same standard deviation across repeats. `0` is the deterministic control. The synthetic known function is used for calibration; these 512 inexpensive oracle points are not included in the optimisation budget.

Example pilot:

```powershell
python .\experiment_jerrell_rewritten.py noisy `
  --algorithms cma,bo,random `
  --functions sphere,rastrigin,ackley,rosenbrock `
  --dimensions 2,5 `
  --budget-multiplier 10 `
  --repeats 20 `
  --noise-models gaussian,student_t `
  --noise-levels 0,0.05,0.2 `
  --bo-startup 8 `
  --bo-backend sklearn `
  --bo-noise-aware known `
  --output-dir results/noisy
```

Run `noisy` with one worker to keep its random-noise stream reproducible. The objective-evaluation budget counts each **noisy** observation. After an optimiser returns a recommendation, the runner computes the known noise-free objective at that point. This diagnostic oracle evaluation is deliberately not charged to the noisy budget; in a real black-box task where the noise-free objective is unavailable, replace it with fresh independent validation evaluations.

Interpret noisy outputs carefully:

- `best_observed_value` is the lowest noisy value used to select the returned recommendation. It may be negative, even though the true benchmark functions are nonnegative. That is a noise realisation, not an actually negative underlying objective.
- `recommendation_true_value` (also `best_value`, for consistent primary summaries) is the noise-free objective value at the selected point. Use this to compare optimisation quality.
- `noise_std_objective_units` records the actual standard deviation used at that function/dimension/level; `noise_reference_scale` records its scale.
- `traces.csv` reports noisy observations and noisy best-so-far values, not the noise-free objective at every candidate.

**Noise-information caveat:** `--bo-noise-aware known` (the default) supplies the known noise variance to the sklearn GP, while CMA-ES and random search use their standard implementations without a replication/noise handler. This is a useful noise-aware BO baseline, but it is not an equal-information comparison. Run a sensitivity condition with `--bo-noise-aware ignore` to see what happens when BO assumes nearly deterministic observations, and state this distinction in the report. A further study could add CMA-ES noise handling or reserve part of the budget for repeated evaluations at promising candidates.

Use at least 20 repeats for a report-quality study; the sample pilot included with this delivery uses only two repeats and is diagnostic, not conclusive. Report medians and spread across repeats, not just the lowest run.

## 6. Output files and interpretation

Every run writes:

- `runs.csv`: one row per repeat/problem/method/worker count. Includes evaluation count, value used for primary evaluation, success flag, seed, budget and wall time. In noisy mode, `best_value`/`recommendation_true_value` are the noise-free score at the returned point and `best_observed_value` is the lowest noisy observation.
- `summary.csv`: per-configuration means/medians, sample standard deviation of best value, mean/median runtime, success rate at `--target`, and success rates across a threshold-sensitivity ladder. In `noisy` mode, the primary value is the noise-free quality of the selected recommendation; noisy best-observed summaries are separate columns.
- `traces.csv`: best-so-far objective value at each completed evaluation, helpful for plotting convergence curves. In `noisy` mode these are noisy observation traces.
- In COCO mode, `coco/`: COCO observer data in separate per-algorithm/per-repeat folders.

For the report, make at least two plots: (1) success rate or median final best value against function and dimension for the multimodality experiment; (2) wall-clock time and median final best value against worker count for the parallelism experiment. Show variability over repeats (for example, standard deviation, interquartile range or confidence intervals), not only averages.

When interpreting speed, `wall_seconds` includes objective-call time, optimiser/acquisition overhead and local thread-pool setup inside the run; it excludes Python process startup and final CSV writing. In the parallelism experiment the artificial delay makes a worker-speedup interpretable. BO's GP fitting/acquisition overhead remains part of its elapsed time, which is important to the project brief.

## 7. What changed from the supplied script

1. Removed `mkl_bugfix`, the nonstandard `fmin_lq_surr2` call, and hard-coded assumptions about a 40-D skip.
2. Replaced the surrogate-CMA entry point with the standard `CMAEvolutionStrategy.ask()` / `tell()` interface from pycma.
3. Added a low-overhead single-worker sklearn GP-EI backend with Sobol startup and candidate points, periodic hyperparameter optimisation, and an explicit Optuna `GPSampler` backend for asynchronous parallel BO. The selected BO backend is written to the CSV results.
4. Removed BFGS from the default comparison because numerical gradient estimates consume extra black-box evaluations and can make a nominal iteration limit misleading. Added random search as a budget-matched reference.
5. The objective budget now includes the initial point, and the script does not make a separate final call to re-evaluate the incumbent.
6. Uses separate COCO suites/observer folders for each method and repeat; records a CSV of raw runs, summary statistics and best-so-far traces.
7. Adds shifted known-optimum test functions for a controlled modality study and worker sweeps for a controlled parallelism study.
8. Uses separate optimizer seeds per method/repeat, while holding the start point fixed across methods for each synthetic repeat.

## 8. Limitations to state in the report

- GP-based BO is generally more comfortable at low-to-moderate dimension and small evaluation budgets than at high dimension or very large budgets. Its training cost grows as the data set grows; a slow GP-based run is not necessarily a code defect.
- The default sklearn BO backend is sequential sampled-EI BO with kernel hyperparameters refitted every five model-based steps, rather than a fully re-optimised GP at each step. It is intended to keep routine low-dimensional comparisons fast; this design choice should be described in the report. The Optuna backend is asynchronous parallel BO, not a jointly optimised batch `qEI`/`qLogEI` design. Do not mix the two BO backends in one aggregate comparison; the backend is recorded in the CSVs.
- `synthetic` mode is deterministic, while `noisy` mode adds controlled random observation noise. In noisy experiments, assess the final recommendation using fresh independent evaluations or the known noise-free function, rather than using the lowest noisy observation.
- For publication-quality comparisons, do tuning on separate pilot problems or a held-out set; do not tune a method repeatedly against the same final benchmark instances.

Optuna labels `GPSampler` as experimental, so pin the Optuna version used for the final report and record it (`python -c "import optuna; print(optuna.__version__)"`). The method/backend, seed, package versions and worker count should all be included in the experiment log.
