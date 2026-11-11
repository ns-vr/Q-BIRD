#  Q-BIRD
### Quantum-Assisted Banking Intelligence & Resource Decisions

**Turning banking data into smarter customer decisions through machine learning, constrained optimisation, and quantum probability experiments.**

<p align="center">
  <strong>Predict. Prioritise. Simulate.</strong>
</p>

Q-BIRD is a banking intelligence prototype designed to explore how machine learning and quantum-inspired computing techniques can support more informed customer selection and resource allocation decisions.

Instead of treating customer prediction, campaign selection, and outcome estimation as isolated tasks, Q-BIRD connects them through a three-stage decision pipeline.

It predicts customer responses, identifies the most valuable group under real-world constraints, and uses a quantum probability experiment to study possible campaign outcomes.

---

##  The Problem

Banks and financial institutions often work with limited campaign budgets, restricted outreach capacity, and large customer datasets.

The challenge is not simply identifying customers who might respond. It is deciding:

- Which customers are most likely to respond?
- Which combination delivers the highest expected value?
- How should limited campaign budgets and contact capacity be allocated?
- How can institutions estimate the uncertainty around possible outcomes?

Traditional prediction models can help identify likely responders, but prediction alone does not solve the resource allocation problem.

**Q-BIRD brings these decisions together in one connected workflow.**

---

##  Our Solution

Q-BIRD uses a three-stage intelligence pipeline.

| Stage | Module | Purpose |
|---|---|---|
| 01 | **PULSE** | Predicts customer response probabilities using machine learning. |
| 02 | **BIRD** | Selects the best customer combination under budget and capacity constraints. |
| 03 | **FAIR** | Simulates and estimates campaign outcomes using quantum probability techniques. |

### 01 — PULSE: Predictive Intelligence

PULSE analyses customer information and estimates the probability of a positive campaign response.

The pipeline evaluates multiple machine learning models:

- Logistic Regression
- Random Forest
- Extra Trees

Models are evaluated using stratified five-fold out-of-fold predictions. PR-AUC is used as the primary selection metric, with Brier score as a tie-breaker.

The selected model is then refitted and used to generate predictions for the unlabelled test records.

**Output:** Customer-level response probabilities and prediction results.

### 02 — BIRD: Intelligent Resource Allocation

A high response probability does not automatically make a customer the best choice.

BIRD considers the expected economic value of each customer alongside campaign costs and operational restrictions.

The objective is to maximise:

`Expected Net Value = (Response Probability × Expected Deposit Value) − Contact Cost`

The selection process considers:

- Campaign budget limits
- Contact capacity limits
- Customer-level expected value
- The best combination of selected customers

The current implementation considers the top ten candidates and selects exactly six, subject to the configured constraints.

It compares the optimisation result against an exact enumeration of all 210 possible six-customer combinations.

The quantum-inspired optimisation experiment uses a seeded, diagonal-Hamiltonian statevector approach. The exact classical enumeration acts as a correctness reference.

**Output:** A constrained customer selection with its expected economic value.

### 03 — FAIR: Quantum Probability Simulation

FAIR explores how quantum probability experiments can be applied to campaign outcome estimation.

The selected six customers are represented using six qubits. Each qubit is configured using the corresponding response probability:

`θᵢ = 2 × asin(√pᵢ)`

The experiment compares:

- Finite-shot probability estimates
- Exact Poisson-binomial probabilities
- Seeded Monte Carlo simulations

When Qiskit Aer is available, the pipeline can use its quantum simulation capabilities with a depolarising noise model. Otherwise, a seeded NumPy shot simulator provides a clearly labelled measurement bit-flip fallback.

The goal is to examine probability estimation and simulation behaviour, not to claim that a quantum computer currently outperforms classical computing.

**Output:** Campaign outcome probability estimates and simulation comparisons.

---

##  What Makes Q-BIRD Different?

Q-BIRD is not just another customer prediction dashboard.

Its key idea is connecting three different decision-making layers:

**1. Prediction + optimisation**

Rather than stopping after calculating customer probabilities, Q-BIRD uses those predictions to make constrained resource allocation decisions.

**2. Economic awareness**

Customer selection accounts for expected value, contact cost, available budget, and operational capacity.

**3. Quantum probability exploration**

The final stage investigates quantum circuit-based probability estimation and compares it against classical probability methods.

**4. Reproducibility**

Seeded experiments and saved outputs make it easier to inspect and reproduce the pipeline's results.

**5. Transparent limitations**

The system distinguishes actual model outputs, optimisation results, and simulated quantum measurements. It does not present a simulation as proof of quantum advantage.

---

##  System Architecture

```text
                Q-BIRD FRONTEND
                       |
           User Controls & Dashboard
                       |
                       v
               FASTAPI BACKEND
                       |
                       v
              THREE-STAGE PIPELINE
                       |
          +------------+------------+
          |            |            |
          v            v            v
        PULSE         BIRD          FAIR
      Prediction   Optimisation   Simulation
          |            |            |
          v            v            v
      Response     Selected      Probability
      Probabilities Customers     Estimates
          |            |            |
          +------------+------------+
                       |
                       v
               RESULTS & EXPORTS
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, React |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Quantum Simulation | Qiskit Aer, when available |
| Numerical Simulation | NumPy |
| Data Processing | Pandas |
| Dataset | Atomiq banking campaign participant dataset |

---

##  Dataset & Experimental Setup

Q-BIRD uses the provided banking campaign participant dataset.

The current dataset contains:

| Dataset component | Records |
|---|---:|
| Labelled training records | 300 |
| Positive training records | 35 |
| Unlabelled test records | 100 |
| Total records | 400 |

The dataset also includes campaign economics used by the optimisation stage.

The relatively small number of labelled records means that model evaluation results should be interpreted cautiously. Cross-validation helps assess performance, but it does not eliminate uncertainty or guarantee generalisation to a larger banking population.

The project uses the supplied dataset rather than presenting generated customer records as real-world evidence.

---

##  Getting Started

There are two ways to explore Q-BIRD.

### Option 1 — Open the Standalone HTML Frontend

<badge color="success">Quickest way to view the interface</badge>

You can access the standalone frontend directly from the repository without installing Python, Node.js, or additional dependencies.

1. Open this GitHub repository.
2. Navigate to `q-bird/Q-BIRD-frontend.html`.
3. Click **Download raw file** or open the file and save it.
4. Locate the downloaded HTML file on your computer.
5. Double-click it to open it in Chrome, Edge, or Firefox.

That's it. You can explore the standalone interface and its embedded result snapshots.

**Important:** This standalone HTML is a presentation-ready frontend. It does not start the Python backend, execute the machine learning pipeline, or make live API requests. For live pipeline execution, follow Option 2.

### Option 2 — Run the Full Backend-Connected Application

<badge color="info">Full application experience</badge>

This option runs the React dashboard and connects it to the FastAPI backend.

#### Requirements

- Python 3.11 recommended
- Node.js and npm
- Git

#### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd q-bird
```

Replace the repository URL with your actual GitHub clone URL.

#### Step 2 — Install backend dependencies

From the repository root:

```bash
pip install -r backend/requirements.txt
```

#### Step 3 — Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

Check whether it is running:

```text
http://localhost:8000/health
```

#### Step 4 — Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

```text
http://localhost:5173
```

The React frontend uses the backend API to execute the pipeline and retrieve generated results.

#### Optional — Configure the backend URL

The frontend defaults to:

```text
http://localhost:8000
```

If your backend is running elsewhere, create a `.env` file inside the `frontend` directory:

```env
VITE_API_URL=http://localhost:8000
```

Restart the frontend after changing the environment variable.

---

##  API Overview

The FastAPI backend currently exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend availability. |
| POST | `/pipeline/run` | Executes the three-stage pipeline. |
| GET | `/results/{name}` | Retrieves an allowed generated result file. |

The pipeline endpoint accepts optional input data and configuration parameters, including:

- Budget — default: 25
- Contact capacity — default: 22

The default dataset is loaded from the repository's data directory when no alternative file is provided.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

##  Project Structure

```text
q-bird/
│
├── backend/
│   ├── main.py
│   ├── pipeline.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   └── style.css
│   └── public/
│       └── assets/
│           └── qbird-hero.png
│
├── data/
│   ├── bank_campaign_participant_release.xlsx
│   └── schema.json
│
├── Q-BIRD-frontend.html
├── run_pipeline.py
├── validate_outputs.py
├── seeds.json
│
├── stage1_predictions.csv
├── stage2_selection.csv
├── stage2_selection.json
├── stage3_simulation.csv
├── stage3_simulation.json
├── run_metadata.json
│
├── README.md
└── LICENSE
```

---

##  Generated Outputs

Each pipeline run produces structured outputs that can be inspected or exported.

| File | Contents |
|---|---|
| `stage1_predictions.csv` | Customer response predictions. |
| `stage2_selection.csv` | Selected customer combination. |
| `stage2_selection.json` | Optimisation summary and constraints. |
| `stage3_simulation.csv` | Probability simulation results. |
| `stage3_simulation.json` | Simulation summary and comparisons. |
| `run_metadata.json` | Run configuration and execution metadata. |

These outputs make the pipeline easier to validate, compare, and reproduce.

---

##  Feasibility & Potential Applications

Q-BIRD explores a workflow that could be useful in banking and financial campaign operations.

Potential applications include:

- Customer outreach prioritisation
- Deposit campaign planning
- Budget-conscious customer selection
- Campaign capacity management
- Probability-based outcome estimation
- Decision-support experimentation

### Potential Market Direction

The project could be developed further as a decision-support layer for:

- Banks
- Financial institutions
- NBFCs
- Fintech companies
- Banking analytics teams
- Financial campaign management providers

A practical commercial direction would be to offer Q-BIRD as an analytics and optimisation layer that integrates with existing banking systems rather than attempting to replace their core infrastructure.

### Possible Business Model

1. **SaaS licensing:** Subscription-based access for financial institutions.
2. **Enterprise integration:** Custom API and workflow integration.
3. **Analytics services:** Campaign optimisation and performance analysis.
4. **Research partnerships:** Joint experimentation with institutions exploring quantum-assisted financial workflows.

These are potential future directions, not existing commercial deployments or validated market demand.

---

##  Validation & Future Development

The current prototype establishes the core three-stage workflow. Further development could focus on:

### Model improvements
- Larger and more diverse banking datasets.
- Stronger out-of-sample validation.
- Calibration and subgroup performance analysis.
- Better handling of missing and imbalanced data.

### Optimisation improvements
- Larger customer-selection problems.
- More complex operational constraints.
- Comparisons with established classical optimisation solvers.
- Scalability and execution-time benchmarking.

### Quantum experimentation
- Hardware-compatible quantum circuit implementations.
- More extensive noise analysis.
- Comparisons across circuit sizes and shot counts.
- Benchmarks against classical alternatives.

### Product development
- Role-based access and secure authentication.
- Audit logs and model-version tracking.
- Production-grade deployment.
- Integration with authorised banking data sources.
- Better monitoring and explainability.

---

## Limitations & Responsible Use

Q-BIRD is an experimental hackathon and research prototype.

A few important distinctions:

- Quantum simulation is not the same as execution on a real quantum computer.
- No quantum advantage over classical methods is claimed.
- Small training datasets can produce uncertain performance estimates.
- Simulation results are not guaranteed predictions of actual campaign performance.
- The system has not been validated for production banking decisions.
- The current backend is not a production-secured financial service.

Do not upload real customer personal or financial information to a publicly accessible deployment.

Before any real financial deployment, the system would require security reviews, privacy safeguards, appropriate authorisation, independent validation, fairness testing, and regulatory assessment.

---


##  License

This project is distributed under the MIT License. See the `LICENSE` file for details.

---

<p align="center">
  <strong>Q-BIRD</strong><br/>
  Predict. Prioritise. Simulate.<br/>
  <em>Exploring the next layer of intelligent banking decisions.</em>
</p>
