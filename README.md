# Stock Opportunity Agent

Quant research & backtesting lab. **No real trading, no broker, no external APIs, no API keys.**
Idea: same data, same timestamp -> many deterministic agents -> ranking -> benchmarked backtest.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest            # run the tests
streamlit run app.py        # open the dashboard (http://localhost:8501)
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
streamlit run app.py
```

Optional config via environment variables (see `.env.example`):
`SOA_SEED`, `SOA_N_STOCKS`, `SOA_N_PERIODS`, `SOA_TOP_N`, `SOA_RANDOM_TRIALS`, `SOA_EXPERIMENT_DIR`.

## Pipeline (one Experiment)

```
Dataset -> Agents -> Decisions -> Simulated Trades -> Returns -> Metrics -> Ranking
```

`engine/experiment.py::run_experiment(params)` runs it end to end and returns an `Experiment`:
`experiment_id`, `dataset_version`, `timestamp`, `code_version`, `agent_versions`, `parameters`, `results`.
The id is a hash of parameters + dataset version + code/agent versions, so the same experiment
always gets the same id and the same results (only `timestamp` differs). `Experiment.save(dir)` writes strict JSON.

## Dashboard

Dark, multi-screen Streamlit app (theme in `.streamlit/config.toml`). Every screen shows a
**SYNTHETIC DATA** banner; the sidebar shows the data-source status (synthetic, no live feed).

| Screen | What it shows |
|---|---|
| Overview | stocks scanned, opportunities found, active agents, system performance, last update, data source |
| Agents Leaderboard | opportunity score, alpha (+t-stat), win rate, Sharpe, max drawdown, trades, status |
| Stock Opportunities | ticker, company, price, market cap, factor scores, opportunity score, recommending agents; search, filter, sort, CSV export. Select a row to open its detail |
| Stock Detail | company info, financial metrics, factor profile, every agent's view, reasons, rule-based risk flags, score & price history (up to the as-of date only) |
| Performance & Backtesting | equity curve, drawdown, agent comparison, vs market / buy & hold / random, random-portfolio distribution, full metrics |
| Experiments | saved run history (seed, dataset/code/agent versions, parameters, results), inspect + JSON download |

Each run is saved to `experiments/` (`SOA_EXPERIMENT_DIR`, git-ignored) and listed on the Experiments screen.

Layers: `engine/` produces an `Experiment` -> `dashboard/analytics.py` (pure pandas view-models,
unit-tested) -> `dashboard/views/*` (layout only) -> `app.py` (settings, caching, navigation).

## Random Agent (control)

`agents/random_agent.py` ranks stocks by a placebo `noise` factor: a hash of ticker + date, uniform
0-100, uncorrelated with every feature and outcome. It runs through the exact same pipeline
(top-N, equal weight, same metrics), so a strategy that cannot beat it shows no edge. It is never
counted as a recommendation, and its "opportunity score" is hidden on the leaderboard.

## Synthetic data: signal is separate from outcome

Each month, features at time T are noisy observations of hidden state and of the price path up to T.
Only afterwards is the outcome after T drawn: market + sector shock + a *small* loading on the state at T
+ large fat-tailed idiosyncratic noise (about 8% monthly). Details:

- every agent concept (value, quality, catalyst, momentum, breakout, squeeze, contrarian) gets the same weak loading
- each loading is multiplied by a random, seeded, per-regime factor (it can even flip sign)
- part of the edge is a hidden alpha no feature observes
- `SignalConfig.null()` removes every relationship: any alpha found there is luck (used as a control, also selectable in the dashboard)

## Benchmarks (`engine/benchmarks.py`)

- **Market (S&P proxy)**: cap-weighted universe, rebalanced monthly (the reference for alpha)
- **Buy & Hold**: equal-weight universe bought at the start, never rebalanced
- **Random Selection**: top-N random stocks per month over many seeded trials; each agent is also ranked against the distribution of random portfolios ("Beats Random Portfolios")

## Metrics (`engine/evaluation.py`)

Monthly periods, risk-free rate 0, a trade = one pick held one month, portfolio = equal-weight top-N.
Total Return, CAGR, Win Rate, Average Win, Average Loss, Expectancy, Profit Factor, Sharpe (annualised),
Max Drawdown, Number of Trades, and **Alpha** per benchmark: annualised regression intercept
(agent = alpha + beta x benchmark), with beta, the intercept t-statistic and excess CAGR.
|t| >= 2 is the usual bar for "probably not luck"; the dashboard says "Not distinguishable from luck" otherwise.

## No look-ahead bias

- Agents receive only a frozen `StockSnapshot` (plain values, no outcome fields, no reference to the dataset).
- Realised returns live in `MarketDataset.outcome()`, called by the backtest after decisions.
- `tests/test_lookahead.py` verifies: all agents receive identical data; agent code cannot reach outcomes
  (static + dynamic checks); changing future snapshots/outcomes never changes a decision; agents are
  stateless, read-only and do not import each other; identical results across runs and across
  processes with different `PYTHONHASHSEED`.

## Layout

```
app.py              Streamlit entry point: settings, caching, navigation
config.py           Environment-variable settings
interfaces.py       Future extension points - interfaces only (real data plugs in here)
agents/             BaseAgent + 8 strategy agents + Random Agent control (weights = configuration)
data/               models.py, sample_data.py (generator + SignalConfig)
engine/             scoring, backtest, benchmarks, evaluation, experiment (+ history), versioning
dashboard/          analytics.py (view-models), components.py (frame builders), context.py,
                    theme.py (CSS/colours), views/ (one module per screen)
.streamlit/         config.toml (dark theme)
tests/              pytest suite
```

## Adding an agent

Subclass `BaseAgent`, set `name`, `description`, `weights` (sum to 1, keys from `engine.scoring.FACTORS`),
`risk_penalty`, optionally `key_facts()`; add it to `default_agents()` in `agents/__init__.py`.
