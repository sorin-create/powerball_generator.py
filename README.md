# powerball_generator.py
# Null Drift — Powerball Signal & Probability Engine

A lightweight Python statistical simulation and number-sequence generator designed for **US Powerball** (`5` unique main numbers from `1–69` + `1` Powerball from `1–26`).

Built as an interactive computational companion to sci-fi storytelling, this engine analyzes historical draw frequencies, exponential recency decay, rolling sum bounds, and substrate parity thresholds (`a = 0.29213`) to generate filtered candidate sets and run walk-forward backtests.

---

## Quick Start in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook (or open the notebook from this repository).
2. Paste the contents of `powerball_generator.py` into a code cell (or clone this repo).
3. Click **Run (`▶`)**.

If `powerball_archive.csv` is not present locally, the script automatically fetches it from this GitHub repository.

---

## Repository Structure

* `powerball_generator.py` — Main engine (archive cleaner, weighted frequency/decay scoring, dynamic sum filter, parity regime evaluation, variant generator, and walk-forward backtest).
* `powerball_archive.csv` — Cleaned historical Powerball draw dataset (`Date, m1, m2, m3, m4, m5, pb`) under the `1–69` + `1–26` format.
* `README.md` — Project overview and usage instructions.

---

## How It Works

1. **Weighted Number Scoring:** Combines short-window frequency, global historical frequency, and exponential recency decay (`decay = 0.98`) to rank main numbers (`1–69`) and Powerball numbers (`1–26`).
2. **Dynamic Sum Corridor:** Computes a rolling mean ± standard deviation sum range from recent draws so generated sets stay within realistic sum distributions.
3. **Parity & Overlap Filtering:** Applies even/odd balance constraints, overlap limits across generated variants, and a parity stability check (`NULL_DRIFT` vs. `PANIC_JAMMING` at threshold `a = 0.29213`).
4. **Walk-Forward Backtest:** Evaluates out-of-sample top-`K` hit rates against a random baseline without look-ahead bias.

---

## Run Locally

```bash
pip install numpy pandas
python powerball_generator.py
