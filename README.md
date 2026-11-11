# Q-BIRD

**Quantum-assisted banking intelligence and resource decisions.** Q-BIRD connects customer subscription prediction, constrained campaign allocation, and a six-qubit probability experiment in one reproducible pipeline.

This repository uses the official Atomiq participant workbook in `data/bank_campaign_participant_release.xlsx`. It contains 300 labeled training customers, 100 unlabeled test customers, campaign economics for the 400 customers, and the official constraints. No labels are available for the 100 test customers, and the pipeline does not invent them.

## What is implemented

1. **Stage 1, prediction.** Three classical models are compared using the same seeded five-fold out-of-fold predictions. Preprocessing is part of each scikit-learn pipeline, so imputation, scaling, and one-hot encoding are fitted inside each fold. Model selection uses PR-AUC, with Brier score as a tie-breaker. The selected model is refit on all 300 labeled rows and predicts all 100 test customers. Ranking is probability descending, then `customer_id` ascending.
2. **Stage 2, constrained selection.** The top ten candidates are joined to `Stage2_Data`. For each candidate, expected net value is `predicted_probability * expected_deposit_value - contact_cost`. The optimizer must select exactly six, with total contact cost at most 25 and total capacity units at most 22. The repository includes a seeded p=1 custom diagonal-Hamiltonian statevector QAOA experiment and an exact enumeration of all 210 six-of-ten subsets. The exact classical result is the correctness oracle. The custom Stage-2 statevector implementation is **not** represented as a Qiskit-transpiled circuit; its method and limitations are included in the JSON output.
3. **Stage 3, probability simulation.** The six selected probabilities are encoded using `RY(2 asin(sqrt(p_i)))` on six qubits. The estimated probability of at least three subscriptions is compared with the exact Poisson-binomial result and a seeded Monte Carlo benchmark. If Qiskit Aer is installed, the pipeline runs an Aer circuit and an explicit depolarizing-noise experiment. If Qiskit Aer is unavailable, it uses a seeded NumPy shot simulator and clearly labels the approximate measurement bit-flip fallback. Do not describe fallback results as Qiskit execution.

**No quantum advantage is claimed.** The six-qubit circuit prepares independent Bernoulli outcomes from supplied probabilities; it is a probability-estimation demonstration, not proof that quantum computing improves the prediction or business objective.

## Quick start

Python 3.11 is recommended for compatibility with the pinned quantum packages.

### Windows PowerShell

```powershell
cd q-bird
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python run_pipeline.py
```

### macOS / Linux

```bash
cd q-bird
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python run_pipeline.py
```

The default run uses the included official workbook and the official fixed limits (`budget=25`, `capacity=22`). To supply another copy of the same workbook:

```bash
python run_pipeline.py --workbook /path/to/bank_campaign_participant_release.xlsx
```

The script writes the result files into the repository root. The API can also be started with:

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs`. The React dashboard can be started in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The standalone `Q-BIRD-frontend.html` supplied alongside the project is a **frontend-only presentation**. It has embedded snapshot data and export interactions; it does not run Python, call the backend, retrain models, or recompute the quantum simulation.

## Mandatory result files

All are written at the repository root:

- `stage1_predictions.csv`: 100 rows, `customer_id`, `predicted_probability`, `rank`, `is_top_10`.
- `stage2_selection.csv` and `stage2_selection.json`: selected six, operational inputs, expected net value, constraints, QAOA sample, exact classical optimum, and optimality gap.
- `stage3_simulation.csv` and `stage3_simulation.json`: six customer-to-qubit mapping, input probabilities, shot estimates, exact result, Monte Carlo result, noise details, and circuit metadata.
- `run_metadata.json`: seed, environment versions, validation metrics, selected model, methods, and runtime.

## Mathematical definitions

For customer `i`, let `p_i` be the Stage 1 predicted probability, `v_i` the expected deposit value, `c_i` the contact cost, `h_i` the capacity units, and `x_i` the binary contact decision.

\[
\max_x \sum_{i=1}^{10} x_i(p_i v_i-c_i)
\]

subject to

\[
\sum_i x_i=6,\qquad \sum_i c_i x_i\le25,\qquad \sum_i h_i x_i\le22,\qquad x_i\in\{0,1\}.
\]

There are `choose(10,6) = 210` possible six-customer combinations, so exhaustive enumeration gives an exact classical optimum cheaply. The relative optimality gap is `(classical_optimum - QAOA_objective) / max(abs(classical_optimum), 1e-12)`. A zero gap means the sampled feasible QAOA solution matches the exact objective for this run, not that QAOA is generally superior.

For the six selected independent subscription probabilities, Stage 3 computes `P(X >= 3)` using the Poisson-binomial distribution via dynamic programming. It is also estimated from finite shots and Monte Carlo trials. The exact analytical value is the reference; finite-shot differences are expected.

## Reproducibility and caveats

- Primary seed: `42`, consistent with the workbook's Stage 1 sampling seed.
- The workbook has only 35 positive labels among the 300 training rows. Metrics will have substantial uncertainty and should not be generalized as production performance.
- Stage 1 validation metrics use out-of-fold training predictions. They are not test-set metrics.
- The data has no `duration` field, as specified by the workbook README.
- Stage 2 uses the official `Stage2_Data` values. Do not replace the objective with the sum of probabilities alone.
- The QAOA implementation is an educational statevector experiment using a diagonal cost Hamiltonian and soft constraint penalties. It is not a hardware run and the code does not currently transpile a Stage-2 QAOA circuit through Qiskit.
- The Stage 3 fallback is explicitly labeled if Qiskit Aer cannot import. For a Qiskit circuit/noise run, install the pinned dependencies under a compatible Python version and rerun; inspect `execution_backend` in `stage3_simulation.json` before describing the result.
- The standalone HTML is a static visual showcase with embedded snapshot values. Re-run the pipeline before submission if code or inputs change, then refresh the snapshot.

## Repository layout

```text
q-bird/
├── backend/                 # FastAPI service and scientific pipeline
├── data/                    # Official workbook and schema notes
├── frontend/                # React dashboard source
├── stage1_predictions.csv   # Generated result
├── stage2_selection.csv/.json
├── stage3_simulation.csv/.json
├── run_metadata.json
├── run_pipeline.py
├── requirements.txt         # Backend requirements are in backend/
└── LICENSE
```

## License

MIT. See `LICENSE`.
