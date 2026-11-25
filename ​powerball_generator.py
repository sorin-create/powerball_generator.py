from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

MAIN_COLS = ["m1", "m2", "m3", "m4", "m5"]
PB_COL = "pb"
ALL_DRAW_COLS = MAIN_COLS + [PB_COL]

MAIN_MIN, MAIN_MAX = 1, 69
MAIN_PICK_COUNT = 5
PB_MIN, PB_MAX = 1, 26

PARITY_A = 0.29213
PARITY_B = round(1.0 - PARITY_A, 5)

# Inlocuieste USER si REPO_NAME cu numele tau de GitHub si numele repository-ului
GITHUB_PB_CSV_RAW_URL = (
    "https://raw.githubusercontent.com/USER/REPO_NAME/main/powerball_archive.csv"
)
DEFAULT_PB_CSV_PATH = Path("powerball_archive.csv")


def normalize_pb_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        c = str(col).strip().lower()
        if c in {"data", "date", "draw_date"}:
            rename_map[col] = "Date"
        elif c in {"m1", "n1", "nr1", "b1", "ball1"}:
            rename_map[col] = "m1"
        elif c in {"m2", "n2", "nr2", "b2", "ball2"}:
            rename_map[col] = "m2"
        elif c in {"m3", "n3", "nr3", "b3", "ball3"}:
            rename_map[col] = "m3"
        elif c in {"m4", "n4", "nr4", "b4", "ball4"}:
            rename_map[col] = "m4"
        elif c in {"m5", "n5", "nr5", "b5", "ball5"}:
            rename_map[col] = "m5"
        elif c in {"pb", "powerball", "bonus"}:
            rename_map[col] = "pb"
    return out.rename(columns=rename_map)


def clean_pb_archive(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date"] + ALL_DRAW_COLS)

    out = normalize_pb_columns(df)
    if "Date" not in out.columns:
        out["Date"] = pd.date_range("2015-10-07", periods=len(out), freq="3D").strftime("%Y-%m-%d")

    for col in ALL_DRAW_COLS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["Date"] = out["Date"].astype(str).str.strip()
    out = out[out["Date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    out = out.dropna(subset=ALL_DRAW_COLS).copy()

    for col in MAIN_COLS:
        out[col] = out[col].astype(int)
        out = out[(out[col] >= MAIN_MIN) & (out[col] <= MAIN_MAX)]

    out[PB_COL] = out[PB_COL].astype(int)
    out = out[(out[PB_COL] >= PB_MIN) & (out[PB_COL] <= PB_MAX)]

    if out.empty:
        return pd.DataFrame(columns=["Date"] + ALL_DRAW_COLS)

    unique_main = out[MAIN_COLS].apply(lambda r: len(set(r)) == MAIN_PICK_COUNT, axis=1)
    out = out[unique_main].copy()
    out = out.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
    return out[["Date"] + ALL_DRAW_COLS]


def generate_sample_pb_archive(n_draws: int = 250, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_draws, freq="3D").strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        mains = sorted(
            int(x) for x in rng.choice(np.arange(MAIN_MIN, MAIN_MAX + 1), size=MAIN_PICK_COUNT, replace=False)
        )
        pb = int(rng.integers(PB_MIN, PB_MAX + 1))
        rows.append([d] + mains + [pb])
    return pd.DataFrame(rows, columns=["Date"] + ALL_DRAW_COLS)


def download_pb_csv_from_github(
    csv_path: Path = DEFAULT_PB_CSV_PATH,
    raw_url: str = GITHUB_PB_CSV_RAW_URL,
    timeout: int = 15,
) -> bool:
    if not raw_url or "USER/REPO_NAME" in raw_url:
        return False
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        csv_path.write_text(content, encoding="utf-8")
        return csv_path.exists() and csv_path.stat().st_size > 0
    except Exception as e:
        print(f"Download warning: {e}")
        return False


def load_pb_archive(
    csv_path: Path = DEFAULT_PB_CSV_PATH,
    raw_url: str = GITHUB_PB_CSV_RAW_URL,
    allow_github_download: bool = True,
    use_sample_if_missing: bool = True,
) -> pd.DataFrame:
    if not csv_path.exists() and allow_github_download:
        download_pb_csv_from_github(csv_path, raw_url=raw_url)

    if csv_path.exists():
        raw_df = pd.read_csv(csv_path)
        cleaned = clean_pb_archive(raw_df)
        if not cleaned.empty:
            return cleaned

    if use_sample_if_missing:
        print("Using synthetic sample archive (set GITHUB_PB_CSV_RAW_URL or place powerball_archive.csv).")
        return generate_sample_pb_archive()

    return pd.DataFrame(columns=["Date"] + ALL_DRAW_COLS)


def get_pb_hot_cold_stats(hist_df: pd.DataFrame, recent_draws: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hist_df.empty:
        return (
            pd.DataFrame(columns=["Number", "Appearances"]),
            pd.DataFrame(columns=["Number", "Draws_Absent"]),
        )

    vals = hist_df[MAIN_COLS].to_numpy(dtype=int)
    n_total = len(vals)

    recent_slice = vals[-min(recent_draws, n_total) :].ravel()
    counts = np.bincount(recent_slice, minlength=MAIN_MAX + 1)
    hot_df = (
        pd.DataFrame(
            [{"Number": num, "Appearances": int(counts[num])} for num in range(MAIN_MIN, MAIN_MAX + 1)]
        )
        .sort_values(by=["Appearances", "Number"], ascending=[False, True])
        .reset_index(drop=True)
    )

    cold_rows = []
    for num in range(MAIN_MIN, MAIN_MAX + 1):
        draws_since = n_total
        for i in range(n_total - 1, -1, -1):
            if num in vals[i]:
                draws_since = (n_total - 1) - i
                break
        cold_rows.append({"Number": num, "Draws_Absent": draws_since})
    cold_df = (
        pd.DataFrame(cold_rows)
        .sort_values(by=["Draws_Absent", "Number"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return hot_df, cold_df


def compute_ranked_main_numbers(
    hist_df: pd.DataFrame,
    window: int = 5,
    decay: float = 0.98,
    weights: tuple[float, float, float] = (-0.25, 0.60, 0.15),
) -> tuple[list[int], dict[int, float]]:
    universe = list(range(MAIN_MIN, MAIN_MAX + 1))
    if hist_df.empty:
        return universe, {x: 0.0 for x in universe}

    w_win, w_glob, w_rec = weights
    vals = hist_df[MAIN_COLS].to_numpy(dtype=int)
    n_rows = len(vals)
    win_vals = vals[-min(window, n_rows) :]

    win_counts = np.bincount(win_vals.ravel(), minlength=MAIN_MAX + 1)
    glob_counts = np.bincount(vals.ravel(), minlength=MAIN_MAX + 1)

    row_weights = decay ** np.arange(n_rows - 1, -1, -1, dtype=float)
    rec_arr = np.bincount(
        vals.ravel(),
        weights=np.repeat(row_weights, MAIN_PICK_COUNT),
        minlength=MAIN_MAX + 1,
    )

    max_w = max(int(win_counts[1:].max()), 1)
    max_g = max(int(glob_counts[1:].max()), 1)
    max_r = max(float(rec_arr[1:].max()), 1e-9)

    scores = {
        x: (
            w_win * (win_counts[x] / max_w)
            + w_glob * (glob_counts[x] / max_g)
            + w_rec * (rec_arr[x] / max_r)
        )
        for x in universe
    }
    ranked = sorted(universe, key=lambda x: (-scores[x], x))
    return ranked, scores


def rank_powerball_numbers(hist_df: pd.DataFrame, recent_draws: int = 30) -> tuple[list[int], dict[int, float]]:
    pb_universe = list(range(PB_MIN, PB_MAX + 1))
    if hist_df.empty:
        return pb_universe, {x: 0.0 for x in pb_universe}

    pb_vals = hist_df[PB_COL].to_numpy(dtype=int)
    recent = pb_vals[-min(recent_draws, len(pb_vals)) :]
    recent_counts = np.bincount(recent, minlength=PB_MAX + 1)
    glob_counts = np.bincount(pb_vals, minlength=PB_MAX + 1)

    max_r = max(int(recent_counts[1:].max()), 1)
    max_g = max(int(glob_counts[1:].max()), 1)

    scores = {
        x: 0.65 * (recent_counts[x] / max_r) + 0.35 * (glob_counts[x] / max_g)
        for x in pb_universe
    }
    ranked = sorted(pb_universe, key=lambda x: (-scores[x], x))
    return ranked, scores


def compute_dynamic_main_sum_bounds(
    hist_df: pd.DataFrame,
    rolling_window: int = 50,
    std_mult: float = 1.0,
    fallback_bounds: tuple[float, float] = (125.0, 225.0),
) -> tuple[float, float]:
    min_possible = sum(range(1, MAIN_PICK_COUNT + 1))  # 15
    max_possible = sum(range(MAIN_MAX - MAIN_PICK_COUNT + 1, MAIN_MAX + 1))  # 325

    if hist_df.empty or len(hist_df) < 10:
        return float(fallback_bounds[0]), float(fallback_bounds[1])

    sums = hist_df[MAIN_COLS].to_numpy(dtype=float).sum(axis=1)
    arr = sums[-max(10, min(rolling_window, len(sums))) :]
    mean_s = float(arr.mean())
    std_s = float(arr.std())
    low = max(float(min_possible), mean_s - std_mult * std_s)
    high = min(float(max_possible), mean_s + std_mult * std_s)
    return low, high


def evaluate_substrate_parity(
    comb: list[int],
    scores: dict[int, float],
    sum_bounds: tuple[float, float],
    threshold_a: float = PARITY_A,
) -> dict[str, float | str | bool]:
    low_s, high_s = sum_bounds
    span = max(high_s - low_s, 1e-6)
    center = 0.5 * (low_s + high_s)
    sum_dev = min(1.0, abs(sum(comb) - center) / (0.5 * span))

    comb_scores = np.array([scores.get(x, 0.0) for x in comb], dtype=float)
    score_spread = min(1.0, float(np.std(comb_scores)) * 2.0)

    friction_a = round(float(PARITY_A * (0.6 * sum_dev + 0.4 * score_spread) * 1.4), 4)
    regime = "PANIC_JAMMING" if friction_a >= threshold_a else "NULL_DRIFT"
    marginal_stability = (friction_a < threshold_a) and (low_s <= sum(comb) <= high_s)

    return {
        "friction_a": friction_a,
        "passive_b": round(max(0.0, 1.0 - friction_a), 4),
        "regime": regime,
        "marginal_stability": marginal_stability,
    }


def generate_powerball_variants(
    df: pd.DataFrame,
    n_variants: int = 5,
    pool_size: int = 24,
    max_overlap: int = 2,
    max_num_freq: int = 3,
    seed: int | None = None,
    max_attempts: int = 10000,
) -> list[dict]:
    ranked_main, main_scores = compute_ranked_main_numbers(df)
    ranked_pb, pb_scores = rank_powerball_numbers(df)
    low_s, high_s = compute_dynamic_main_sum_bounds(df)

    pool = ranked_main[: max(MAIN_PICK_COUNT, min(pool_size, len(ranked_main)))]
    pool_arr = np.array(pool, dtype=int)
    pb_arr = np.array(ranked_pb[: min(12, len(ranked_pb))], dtype=int)

    raw_m = np.array([main_scores.get(int(x), 0.0) for x in pool_arr], dtype=float)
    m_probs = (raw_m - raw_m.min()) + 0.15
    m_probs = m_probs / m_probs.sum()

    raw_pb = np.array([pb_scores.get(int(x), 0.0) for x in pb_arr], dtype=float)
    pb_probs = (raw_pb - raw_pb.min()) + 0.20
    pb_probs = pb_probs / pb_probs.sum()

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    selected_sets: list[set[int]] = []
    freq: dict[int, int] = {int(x): 0 for x in pool_arr}

    stages = [
        (True, max_overlap, max_num_freq),
        (True, max_overlap + 1, max_num_freq + 1),
        (False, 4, n_variants),
    ]

    for strict_filter, ov_limit, freq_limit in stages:
        attempts = 0
        while len(selected) < n_variants and attempts < max_attempts:
            attempts += 1
            pick = rng.choice(pool_arr, size=MAIN_PICK_COUNT, replace=False, p=m_probs)
            comb = sorted(int(x) for x in pick)
            c_set = set(comb)

            if any(c_set == prev for prev in selected_sets):
                continue

            evens = sum(1 for x in comb if x % 2 == 0)
            s = sum(comb)

            if strict_filter:
                if not (2 <= evens <= 3 and low_s <= s <= high_s):
                    continue
            else:
                if not (1 <= evens <= 4 and 80 <= s <= 270):
                    continue

            if any(freq[x] >= freq_limit for x in comb):
                continue
            if not all(len(c_set & prev_set) <= ov_limit for prev_set in selected_sets):
                continue

            pb_pick = int(rng.choice(pb_arr, p=pb_probs))
            avg_sc = float(np.mean([main_scores.get(x, 0.0) for x in comb]))
            parity = evaluate_substrate_parity(comb, main_scores, (low_s, high_s))

            selected.append(
                {
                    "main": comb,
                    "pb": pb_pick,
                    "score": avg_sc,
                    "evens": evens,
                    "sum": s,
                    "parity": parity,
                }
            )
            selected_sets.append(c_set)
            for x in comb:
                freq[x] += 1

        if len(selected) >= n_variants:
            break

    selected.sort(
        key=lambda item: (
            0 if bool(item["parity"]["marginal_stability"]) else 1,
            float(item["parity"]["friction_a"]),
            -float(item["score"]),
            item["main"],
        )
    )
    return selected


def walk_forward_pb_backtest(
    df: pd.DataFrame,
    window: int = 5,
    decay: float = 0.98,
    weights: tuple[float, float, float] = (-0.25, 0.60, 0.15),
    k_list: tuple[int, ...] = (5, 10, 15),
    start_date: str | None = "2026-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    if len(df) <= window:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    universe = np.arange(MAIN_MIN, MAIN_MAX + 1)
    start_dt = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise") if start_date else None
    row_dts = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

    main_hits = {k: [] for k in k_list}
    rand_main_hits = {k: [] for k in k_list}
    pb_hits = []

    for t in range(window, len(df)):
        if start_dt is not None and (pd.isna(row_dts.iloc[t]) or row_dts.iloc[t] < start_dt):
            continue

        hist = df.iloc[:t]
        actual_main = set(int(v) for v in df.iloc[t][MAIN_COLS].values)
        actual_pb = int(df.iloc[t][PB_COL])

        ranked_main, _ = compute_ranked_main_numbers(hist, window=window, decay=decay, weights=weights)
        ranked_pb, _ = rank_powerball_numbers(hist)

        pb_hits.append(1 if ranked_pb[0] == actual_pb else 0)

        for k in k_list:
            safe_k = min(k, MAIN_MAX)
            h = len(set(ranked_main[:safe_k]) & actual_main)
            rh = len(set(int(x) for x in rng.choice(universe, size=safe_k, replace=False)) & actual_main)
            main_hits[k].append(h)
            rand_main_hits[k].append(rh)

    if not pb_hits:
        return pd.DataFrame()

    pb_hit_rate = round(float(np.mean(pb_hits) * 100), 2)
    rows = []
    for k in k_list:
        arr = np.array(main_hits[k], dtype=float)
        rand_arr = np.array(rand_main_hits[k], dtype=float)
        rows.append(
            {
                "Top_K_Main": k,
                "Draws": len(arr),
                "Avg_Main_Hits": round(float(arr.mean()), 4),
                "Random_Avg": round(float(rand_arr.mean()), 4),
                "Delta": round(float(arr.mean() - rand_arr.mean()), 4),
                "Hit>=1_%": round(float((arr >= 1).mean() * 100), 2),
                "Top1_PB_Hit_%": pb_hit_rate,
            }
        )
    return pd.DataFrame(rows)


def run_powerball_generator(
    n_variants: int = 5,
    seed: int | None = None,
    backtest_start: str | None = "2026-01-01",
    run_backtest: bool = True,
):
    df = load_pb_archive()
    low_s, high_s = compute_dynamic_main_sum_bounds(df)
    hot_df, cold_df = get_pb_hot_cold_stats(df, recent_draws=20)
    variants = generate_powerball_variants(df, n_variants=n_variants, seed=seed)

    print(
        f"Draws: {len(df)} ({df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}) | "
        f"Main sum band: {low_s:.1f}-{high_s:.1f} | Threshold a={PARITY_A}\n"
    )
    print("Top 5 Hot (last 20 draws):", hot_df.head(5)["Number"].tolist())
    print("Top 5 Cold (draws absent):", cold_df.head(5)["Number"].tolist(), "\n")

    for i, v in enumerate(variants, 1):
        p = v["parity"]
        print(
            f"V{i}: Main {v['main']} + PB [{v['pb']}] | "
            f"Sc:{v['score']:.3f} Evens:{v['evens']} Sum:{v['sum']} | "
            f"a(t)={p['friction_a']:.4f} [{p['regime']}]"
        )

    if run_backtest:
        bt = walk_forward_pb_backtest(df, start_date=backtest_start)
        if not bt.empty:
            print(f"\nWalk-Forward Backtest (from {backtest_start or 'start'}):")
            print(bt.to_string(index=False))

    return variants


run_powerball_generator(n_variants=5, seed=None, backtest_start="2026-01-01", run_backtest=True)
