"""
Streamlit App – Momentum Rebalancing Demo
========================================
Run with ⇒  
    streamlit run momentum_app.py

Assumes an **already‑downloaded price CSV** (wide format, Date column + tickers)
covering 2014‑01‑01 through today.  Point the *CSV path* widget to the file.

Play with parameters in the sidebar and see how different rebalancing schemes
(CUSUM‑driven vs fixed vs buy‑and‑hold) perform.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --- Project imports --------------------------------------------------------
try:
    from pre_whiten_ts import Prewhiten  # type: ignore
    from run_grid import *
    from resettable_cusum import ResettableCUSUM  # type: ignore
except ImportError as e:
    st.error("Could not import project modules pre_whiten_ts / resettable_cusum.")
    raise e

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # Offline mode only

##############################################################################
# USER PARAMETERS – edit these directly in the notebook                      #
##############################################################################

TICKERS: List[str] = [
    "AAPL", "GOOG", "AMZN", "INTC", "ORCL",
    "XOM", "CVX", "COP", "HES", "OXY",
]
START_DATE: str | dt.date = "2014-01-01"  # YYYY‑MM‑DD or datetime.date
END_DATE: str | dt.date | None = None      # None => today
CSV_PATH: Path | None = None               # Optional local CSV with prices

# Momentum & portfolio
LOOKBACK: int = 126  # trading days for momentum ranking
TOP_N: int = 2       # number of assets to hold
FIX_K: int = 21      # fixed rebalancing interval (days)

# Detector tuning
ALARM_TARGETS: List[str] = ["mean_down", "var_up"]  # choose any subset
DELTA: float = 0.4   # design shift (σ₀ units)
H: float = 8.0       # CUSUM threshold
WARMUP: int = 60     # baseline estimation window
PREWHITEN: str = "garch"  # "none", "ar1", "garch"
TRANSACTION_COST = 0.00  # proportional cost per full turnover (e.g. 0.001 = 10 bps)

APP_DIR = Path(__file__).resolve().parent  # directory of this script

##############################################################################
# Helper functions                                                           #
##############################################################################

def _resolve_csv_path(csv_path: str | Path) -> Path:
    """Return an absolute Path, resolving relative paths to the app directory."""
    p = Path(csv_path).expanduser()
    if not p.is_absolute():
        p = (APP_DIR / p).resolve()
    return p

def load_prices(csv_path: Path, tickers: List[str]) -> pd.DataFrame:
    csv_path = _resolve_csv_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        st.warning(f"Missing tickers in CSV skipped: {', '.join(missing)}")
    return df.loc[:, [c for c in tickers if c in df.columns]].sort_index()

def momentum_weights(prices: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    """Equal‑weight the *top_n* performers over the past *lookback* days."""
    rel = prices / prices.shift(lookback)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for t in rel.index[lookback:]:
        sel = rel.loc[t].rank(ascending=False, method="first") <= top_n
        if sel.any():
            w.loc[t, sel] = 1 / sel.sum()
    return w

def build_detector(
    series: pd.Series,
    target: str,
    delta: float,
    h: float,
    warmup: int,
    prewhiten: str,
):
    """Return a ResettableCUSUM (optionally pre-whitened) using explicit params."""
    mapping = {
        "mean_down": dict(targets={"mu_down"}, delta_mu=delta, h_mu_dn=h),
        "mean_up":   dict(targets={"mu_up"},   delta_mu=delta, h_mu_up=h),
        "var_up":    dict(targets={"sig_up"},  delta_sig=delta, h_sig_up=h),
        "var_down":  dict(targets={"sig_down"},delta_sig=delta, h_sig_dn=h),
    }
    if target not in mapping:
        raise ValueError(target)

    if prewhiten == "none":
        base = series.iloc[:warmup]
        det = ResettableCUSUM(base.mean(), base.std(ddof=1), warmup=warmup, **mapping[target])
    else:
        det = Prewhiten(ResettableCUSUM, W=warmup, warmup=warmup, **mapping[target])
        det.method = prewhiten
    return det

def detect_alarms(series: pd.Series, detector) -> List[pd.Timestamp]:
    alarms: List[pd.Timestamp] = []
    for i, (ts, x) in enumerate(series.items()):
        if _update(detector, i, x):
            alarms.append(ts)
    return alarms


def _update(det, i, x):
    """Try detector.update with flexible signatures (x,i | x | i,x)."""
    for call in ((x, i), (x,), (i, x)):
        try:
            return det.update(*call)
        except TypeError:
            continue
    raise


def run_backtest(prices: pd.DataFrame, dates: Iterable[pd.Timestamp], lookback: int, top_n: int, trans_cost: float) -> pd.Series:
    """Momentum back‑test with proportional transaction cost."""
    w_sparse = momentum_weights(prices, lookback, top_n).loc[dates]
    w_sparse = w_sparse.reindex(sorted(set(dates)))
    w_daily = (
        w_sparse.replace(0.0, np.nan)
        .ffill()
        .reindex(prices.index)
        .fillna(method="ffill")
    )
    ret = prices.pct_change().fillna(0.0)
    prev_w   = w_daily.shift(1).fillna(0.0)   
    gross = (w_daily * ret).sum(axis=1)

    # Step 4: turnover & costs (only on rebalance days)
    turnover = (w_daily - prev_w).abs().sum(axis=1)
    net = gross - turnover * trans_cost
    wealth = (1 + net).cumprod()
    wealth.iloc[0] = 1.0
    return wealth

def buy_hold_wealth(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight *all* tickers from day 0 and hold to the end."""
    ret = prices.pct_change().fillna(0.0)
    eq_w = pd.DataFrame(1 / prices.shape[1], index=prices.index, columns=prices.columns)
    wealth = (1 + (eq_w * ret).sum(axis=1)).cumprod()
    wealth.iloc[0] = 1.0
    return wealth

def buy_hold_wealth(prices: pd.DataFrame, trans_cost: float) -> pd.Series:
    ret = prices.pct_change().fillna(0.0)
    eq_w = pd.DataFrame(1 / prices.shape[1], index=prices.index, columns=prices.columns)
    port = (eq_w * ret).sum(axis=1)
    # one‑off initial cost to build the portfolio
    port.iloc[0] -= trans_cost
    wealth = (1 + port).cumprod()
    wealth.iloc[0] = 1.0
    return wealth

##############################################################################
# Simulation tab helpe                                                       #
##############################################################################
def parse_nums(txt: str, *, as_int=False, allow_none=True) -> tuple:
    """
    Return a tuple of parsed numbers (or None tokens).
    • Empty string  -> (None,)  if allow_none else ()
    • 'None' token  -> (None,)
    """
    if isinstance(txt, (tuple, list)):                       # already parsed
        return tuple(txt)

    toks = [t.strip() for t in txt.replace(";", ",").split(",") if t.strip()]
    if not toks:                                             # empty box
        return (None,) if allow_none else tuple()

    out = []
    for t in toks:
        if t.lower() == "none":
            out.append(None)
        else:
            try:
                out.append(int(t) if as_int else float(t))
            except ValueError:
                st.warning(f"Could not parse '{t}'")
    return tuple(out)

##############################################################################
# Streamlit UI                                                               #
##############################################################################
# st.sidebar.header("Connect with me")

# st.sidebar.link_button("GitHub",   "https://github.com/ang-zhao",
#                        icon=":material/folder_code:")
# st.sidebar.link_button("LinkedIn", "https://linkedin.com/in/angela0zhao",
#                        icon=":material/link:")
# st.sidebar.link_button("Personal Website", "https://ang-zhao.github.io/angzhao.github.io/",
#                        icon=":material/person:")
# st.sidebar.link_button("Google Scholar", "https://scholar.google.com/citations?hl=en&user=Vc9Pn4YAAAAJ",
#                        icon=":material/fingerprint:")

st.sidebar.title("Parameters")
st.sidebar.markdown("## ⓵ Portfolio tab parameters")
# --- File + tickers ---------------------------------------------------------
csv_path = st.sidebar.text_input("CSV path", value= Path("data")/"prices.csv")
all_tickers_default = [
    "AAPL", "GOOG", "AMZN", "INTC", "ORCL",
    "XOM", "CVX", "COP", "HES", "OXY",
]
selected_tickers = st.sidebar.multiselect("Select tickers", options=all_tickers_default, default=all_tickers_default)

# --- Dates ------------------------------------------------------------------
start_date = st.sidebar.date_input("Start date", dt.date(2014, 1, 1))
end_date = st.sidebar.date_input("End date", dt.date.today())

# --- Momentum ---------------------------------------------------------------
lookback = st.sidebar.number_input("Momentum look-back (days)", 20, 252, 126, step=5)
top_n = st.sidebar.slider("Number of assets held (Top N)", 1, len(selected_tickers), 2)
fix_k = st.sidebar.number_input("Fixed rebalance interval (days)", 5, 252, 21, step=1)

# --- Detector params --------------------------------------------------------
st.sidebar.subheader("CUSUM settings")
alarm_targets = st.sidebar.multiselect("Targets", ["mean_down", "mean_up", "var_up", "var_down"], default=["mean_down", "var_up"])
_delta = st.sidebar.number_input("Design shift Δ (σ units)", 0.0, 20.0, 0.4, step=0.1)
_h = st.sidebar.number_input("Threshold h", 0.1, 20.0, 8.0, step=0.5)
warmup = st.sidebar.number_input("Warm-up window", 1, 252, 60, step=5)
prewhiten = st.sidebar.selectbox("Pre-whitening", ["garch"], index=0) #"none", "ar1" removed for now

# --- Costs ------------------------------------------------------------------
trans_cost = st.sidebar.number_input("Transaction cost (proportional)", 0.0, 1.0, 0.0, step=0.0005, format="%f")

intro_tab, portfolio_tab, simulation_tab = st.tabs(["📖 Introduction", "📈 Portfolio", "📈 Simulations"])

with intro_tab:
    st.markdown(
        """
        ## 📈 Adaptive-Trading Dashboard  
        ### Interactive demo (2025).

        The app has two independent workspaces:

        | Tab | What it shows | How to interact |
        |-----|---------------|-----------------|
        | **Portfolio** | Back-tests a *Method-A momentum* strategy on your local price file. Compares four rebalancing schemes: CUSUM μ↓ alarms, CUSUM σ↑ alarms, fixed-interval (*K* days), buy-and-hold | Widgets live in the **left sidebar**. Changes re-draw instantly. |
        | **Simulations** | Monte-Carlo study of the **Resettable CUSUM** detector. Explore how *p*, *h*, Δ, warm-up *W*, and optional pre-whitening affect hit-rate, LADD & TAFAR. | Controls sit in a **right-hand panel**; nothing runs until you press **“Run simulation ▶”**. Results appear in 3-D scatter + frontier plots. |

        ---

        ### ① Portfolio tab — quick steps
        1. **Load prices**  
        *Point the “CSV path” box to a wide file with a `Date` column plus one column per ticker (adj-close prices).*  
        2. **Pick tickers & date window**.  
        3. **Tune momentum** — look-back window, Top-*N*, fixed interval.  
        4. **Set CUSUM parameters** (Δ, *h*, warm-up, pre-whitening).  
        5. **Transaction cost** (proportional, per turnover).  
        The plot updates live; hover the legend to isolate a curve.

        ---

        ### ② Simulations tab — how to run
        1. **Choose detector target**  
        *μ↓ (mean-drop) **or** σ↑ (variance-up)*  
        2. **Enter comma-separated grids**  
        *p, warm-up *W*, thresholds *h*, design-shift (Δμ or Δσ), μ/σ in-control & post-shift.*  
        3. *(Optional)* change Monte-Carlo reps *M* or pre-whitening model.  
        4. **Press “Run simulation ▶”**  
        The grid is simulated once; a metric shows **max hit-rate**, a 3-D scatter displays hit-rate surfaces (colour-coded by μ₁ or σ₁), and an LADD-vs-TAFAR frontier appears below. Expand *Raw results* for the full DataFrame.

        *Tip:* parameter grids are cached; re-running with unchanged values is instant.

        ---

        ### ③ Interpretation cheatsheet
        | Acronym | Meaning |
        |---------|---------|
        | **Hit-rate** | fraction of shifts the detector catches within 1 × shift length |
        | **LADD** | mean detection delay (once per alarm) |
        | **TAFAR** | time-averaged false-alarm rate |

        *Bigger hit-rate, lower LADD & TAFAR = better.*

        ---

        ### ④ Notes & disclaimers
        * All computations are offline — the app never fetches data from the internet.  
        * Examples use daily prices 2014-present; swap in your own CSV for other assets. yfinance implementation in progress. 
        * Educational tool only; **not** investment advice.
        ---

        ### ⑤ Connnect with me

        [GitHub](https://github.com/ang-zhao)
        [LinkedIn](https://linkedin.com/in/angela0zhao)  
        [Personal Website](https://ang-zhao.github.io/angzhao.github.io/)  
        [Google Scholar](https://scholar.google.com/citations?hl=en&user=Vc9Pn4YAAAAJ)
        """
    )

# --- Load data --------------------------------------------------------------
with portfolio_tab:
    try:
        prices = load_prices(Path(csv_path), selected_tickers)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    prices = prices.loc[str(start_date): str(end_date)].dropna()
    if prices.empty:
        st.error("No overlapping data for given dates.")
        st.stop()

    signal = prices.pct_change().mean(axis=1).dropna()

    # --- Build schedules --------------------------------------------------------
    first_day = prices.index[0:1]

    schedules = {
        "CUSUM μ↓": [],
        "CUSUM σ↑": [],
        "Fixed": list(prices.index[::fix_k]),
    }

    if "mean_down" in alarm_targets:
        det_mu = build_detector(signal, "mean_down", _delta, _h, warmup, prewhiten)
        schedules["CUSUM μ↓"] = detect_alarms(signal, det_mu)
    if "var_up" in alarm_targets:
        det_var = build_detector(signal, "var_up", _delta, _h, warmup, prewhiten)
        schedules["CUSUM σ↑"] = detect_alarms(signal, det_var)
    if "mean_up" in alarm_targets:
        det_mu_up = build_detector(signal, "mean_up", _delta, _h, warmup, prewhiten)
        schedules["CUSUM μ↑"] = detect_alarms(signal, det_mu_up)
    if "var_down" in alarm_targets:
        det_var_dn = build_detector(signal, "var_down", _delta, _h, warmup, prewhiten)
        schedules["CUSUM σ↓"] = detect_alarms(signal, det_var_dn)

    # Ensure first day present
    schedules = {k: sorted(set(v).union(first_day)) for k, v in schedules.items()}

    # --- Back-tests -------------------------------------------------------------
    wealth = {k: run_backtest(prices, v, lookback, top_n, trans_cost) for k, v in schedules.items()}
    wealth["Buy & Hold"] = buy_hold_wealth(prices, trans_cost)

    # NEW check-boxes ---------------------------------------------------------
    show_mu_lines  = st.sidebar.checkbox("Show μ↓ alarm lines",  value=False)
    show_sig_lines = st.sidebar.checkbox("Show σ↑ alarm lines", value=False)
    show_fix_lines = st.sidebar.checkbox("Show fixed rebalance lines", value=False)

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, s in wealth.items():
        ax.plot(s.index, s.values, label=name)

    # ╭─  vertical lines at alarm  ───────────────────────────────────────────────╮
    if show_mu_lines:
        for t in schedules["CUSUM μ↓"]:
            ax.axvline(t, color="tab:blue",  ls="--", lw=0.8, alpha=0.6)
    if show_sig_lines:
        for t in schedules["CUSUM σ↑"]:
            ax.axvline(t, color="tab:orange", ls="--", lw=0.8, alpha=0.6)
    if show_fix_lines:
        for t in schedules["Fixed"]:
            ax.axvline(t, color="tab:green", ls="--", lw=0.8, alpha=0.6)
    # ╰───────────────────────────────────────────────────────────────────────╯

    ax.set_ylabel("Cumulative wealth (initial = 1)")
    ax.set_title("Method-A Momentum – Rebalancing Schedules")
    ax.legend()
    ax.grid(False) 
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    st.pyplot(fig)

    st.markdown(
        """
        ## 📈 Momentum Rebalancing Demo — User Guide
        Welcome! This mini-app lets you explore a **portfolio momentum strategy** with different rebalancing
        schedules, detector settings and trading frictions — all on *locally-cached* price data
        (no internet calls).

        ---

        ### 1 · Load data
        1. Put your wide CSV (columns = tickers, first column = `Date`) in the project folder  
        &nbsp;  — e.g. `data/prices.csv` provided in the repo.  
        2. In the **sidebar** → *CSV path* field, enter the relative or absolute path.  
        • A green check means the file was found; a red error stops the app.

        ---

        ### 2 · Choose the universe & horizon
        | Control | Meaning |
        |---------|---------|
        | **Select tickers** | pick the assets to trade. Missing symbols in the CSV are skipped. |
        | **Start / End date** | window the back-test; defaults to full file span. |

        ---

        ### 3 · Set momentum strategy
        | Control | Default | Notes |
        |---------|---------|-------|
        | Look-back (days) | 126 | trailing window used to rank performance |
        | Top *N* assets | 2 | equal-weights the best *N* tickers |
        | Fixed interval | 21 days | baseline schedule (cash held between trades) |

        ---

        ### 4 · Configure CUSUM alarms
        Under **“CUSUM settings”** pick:
        * **Targets** μ↓ (mean-drop), μ↑, σ↑ (variance-up), σ↓  
        * **Δ (σ units)** design shift the detector is tuned for  
        * **Threshold *h*** higher = fewer alarms  
        * **Warm-up window** observations used to estimate in-control mean/σ  
        * **Pre-whitening** *none*, AR(1) or GARCH — reduces autocorrelation before detection.

        ---

        ### 5 · Trading friction
        Set a **transaction-cost** (proportional): e.g.  
        * `0.001 = 10 bps` per unit of portfolio turnover on each rebalance.  
        Buy-and-hold pays it once, at inception.

        ---

        ### 6 · What the plot shows
        * **CUSUM μ↓ / σ↑ / …** alarm-driven schedules (one curve per target you enabled)  
        * **Fixed** rebalance every *K* days regardless of alarms  
        * **Buy & Hold** build equal-weight portfolio on day 0 and never trade again

        All curves start at 1.0; y-axis is the **cumulative wealth multiplier** after costs.
        Hover the legend to isolate a strategy, or zoom with the toolbar.

        ---

        ### 7 · Tips & gotchas
        * Extreme parameter values (very low *h* or high Δ) can fire thousands of alarms and slow the UI.  
        * Pre-whitening adds a short delay (warm-up) before alarms can appear.

        ---

        ### 8 · Disclaimer
        This tool is **for educational purposes only** and does not constitute investment advice.
        """
    )

# ---------------------------- Simulations tab ----------------------------
# Place this block under `with tab_sim:` in streamlit_app.py
with simulation_tab:
    # ----------------- Simulations tab ---------------------------------
    st.header("Simulation tab explorer")

    # ❶ Layout: big plot area (col_main) + slim control column (col_ctrl)
    # col_main, col_ctrl = st.columns([3, 1])

    # ── Sidebar – common grids ------------------------------------------
    st.sidebar.markdown("## ⓶ Simulation grid")
    with st.sidebar.form(key="sim_form"): # Form prevents the app from reloading after every parameter change
        target = st.selectbox(
            "CUSUM target",
            options={"mu_down": "μ↓ (mean-drop)", "sig_up": "σ↑ (variance-up)"},
            format_func=lambda k: {"mu_down": "μ↓ (mean-drop)",
                                "sig_up": "σ↑ (variance-up)"}[k],
        )

        p_vals_txt   = st.text_input("p values",              "0.001667")
        W_vals_txt   = st.text_input("warm-up W",             "50,100,150")
        dmu_vals_txt = "0.4"
        dsig_vals_txt = "0.4"
        # ── Target-specific threshold grid ----------------------------------
        if target == "mu_down":
            h_mu_txt  = st.text_input("h μ↓ grid",            "1,3,6,10")
            h_mu_vals  = parse_nums(h_mu_txt,  as_int=True, allow_none=False)
            h_sig_vals = (None,)                                       # placeholder
            h_label    = "h_μ↓"
            axis_h     = "h_mu_dn"
            dmu_vals_txt = st.text_input("Δμ design shift",       "0.3,0.4")
        else:  # σ↑
            h_sig_txt = st.text_input("h σ↑ grid",           "3,6")
            h_sig_vals = parse_nums(h_sig_txt, as_int=True, allow_none=False)
            h_mu_vals  = (None,)
            h_label    = "h_σ↑"
            axis_h     = "h_sig_up"
            dsig_vals_txt = st.text_input("Δσ design shift",       "0.3,0.4")

        # ── Baseline & post-shift values ------------------------------------
        mu0_vals = parse_nums(st.text_input("μ₀ (pre)",  "0.1"))
        mu1_vals = parse_nums(st.text_input("μ₁ (post)", "-1,-10"))
        sig0_vals = parse_nums(st.text_input("σ₀ (pre)", "0.1"))
        sig1_vals = parse_nums(st.text_input("σ₁ (post)", "0.3"))

        pre_model = st.selectbox("Pre-whitening", ["none", "ar1", "garch"], 0)
        M_trials  = st.number_input("Monte-Carlo reps (M)", 1, 500, 20, step=5)

        # ── 3-D axis pickers -------------------------------------------------
        axis_map = {"p": "p", h_label: axis_h, "W": "W", "Δμ": "delta_mu", "μ₁": "mu1"}
        if target == "var_up":
            axis_map["σ₁"] = "sig1"                            # extra axis option

        x_axis = st.selectbox("X-axis", list(axis_map), 0)
        y_axis = st.selectbox("Y-axis", list(axis_map), 1)

        run_btn = st.form_submit_button("Run simulation ▶")

    if run_btn:
        st.spinner("Running grid …")

        df = run_grid(
            p_grid         = (p_vals := parse_nums(p_vals_txt)),
            T_grid         = (12000,),
            h_mu_dn_grid   = h_mu_vals,
            h_sig_up_grid  = h_sig_vals,
            W_grid         = parse_nums(W_vals_txt, as_int=True, allow_none=False),
            delta_mu_grid  = parse_nums(dmu_vals_txt),
            delta_sig_grid = parse_nums(dsig_vals_txt),
            mu0_grid       = mu0_vals,
            mu1_grid       = mu1_vals,
            sig0_grid      = sig0_vals,
            sig1_grid      = sig1_vals,
            targets        = {target},
            M              = int(M_trials),
            prewhiten_method = pre_model,
            prewhiten_W      = 50,
        )

        st.success(f"Grid complete → {len(df)} rows")
        st.metric("Max hit-rate", f"{df['hit_rate'].max():.2%}")

        # ── 3-D scatter — colour by μ₁ or σ₁ ──────────────────────────────
        plot_df = df.dropna(subset=["hit_rate"])
        if plot_df.empty:
            st.warning("All rows have NaN hit-rate — nothing to plot.")
        else:
            colour_col = "mu1" if target == "mu_down" else "sig1"
            legend_lab = "μ₁"  if target == "mu_down" else "σ₁"

            import plotly.express as px
            fig3d = px.scatter_3d(
                plot_df,
                x = axis_map[x_axis],
                y = axis_map[y_axis],
                z = "hit_rate",
                color = plot_df[colour_col].astype(str),
                size  = "hit_rate",
                symbol = "W",
                hover_data = ["LADD", "TAFAR", "delta_mu", "delta_sig", colour_col],
                width = 1100,    
                height = 800, 
            )
            x_vals = np.sort(plot_df[axis_map[x_axis]].unique())
            y_vals = np.sort(plot_df[axis_map[y_axis]].unique())
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(tickmode="array",
                            tickvals=x_vals,
                            ticktext=[f"{v:g}" for v in x_vals]),
                    yaxis=dict(tickmode="array",
                            tickvals=y_vals,
                            ticktext=[f"{v:g}" for v in y_vals]),
                ),
                title=f"Hit-rate surface — target = {target}",
                legend_title=f"{legend_lab}, W",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig3d, use_container_width=False)

        st.subheader("Operating frontier (LADD vs TAFAR)")
        tradeoff_spider(df, targets={target})
        st.pyplot(plt.gcf())

        with st.expander("Show raw results table"):
            st.dataframe(df)

    else:
        st.write("⬅️ Enter parameter grids and press **Run simulation ▶**")

    st.markdown(
        """
## 📝 Simulation-Tab Cheat-Sheet  

### 1 · Inputs you can tweak
| Control | Symbol(s) | What it means | Typical range |
|---------|-----------|---------------|---------------|
| **Target** | `μ↓` or `σ↑` | Which statistic the detector watches: mean-down or variance-up. | — |
| **Event rate `p`** | `p` | *Hazard rate* of a change-point in the Monte-Carlo generator (≈ 1/average run-length). Smaller *p* = rarer shifts. | `10⁻⁴ – 10⁻²` |
| **Warm-up `W`** | `W` | Obs. used to estimate in-control μ₀, σ₀ after each alarm. Longer *W* → better estimates, slower “re-arming”. | 20 – 200 |
| **Threshold grid** | `h_μ↓` **or** `h_σ↑` | CUSUM decision threshold. Higher *h* = fewer (but later) alarms. | 2 – 10 (scaled σ₀ units) |
| **Design shift** | `Δμ` **or** `Δσ/σ₀` | The magnitude the detector is *tuned for*. Sets the drift in the CUSUM update. | 0.1 – 1.0 |
| **Baseline μ₀, σ₀** | `μ₀`, `σ₀` | In-control mean & st.dev. used by the synthetic generator. | any float |
| **Post-shift μ₁, σ₁** | `μ₁`, `σ₁` | Mean or st.dev. *after* a change-point. Larger gap from baseline ⇒ easier detection. | any float |
| **Pre-whitening** | *none / ar1 / garch* | Removes AR(1) or GARCH volatility before CUSUM. Lowers false-alarm rate in autocorrelated data. | drop-down |
| **Monte-Carlo reps** | `M` | Number of independent simulated streams per grid cell. Higher *M* = smoother estimates, slower runtime. | 10 – 500 |

---

### 2 · Outputs shown
| Metric | Definition | Shown as |
|--------|------------|----------|
| **Hit-rate** | % of true change-points detected within 1× shift length. | **Z-axis & bubble size** in 3-D scatter; text metric. |
| **LADD** | *Lagged alarm detection delay* – mean delay (in obs.) after each alarm. Lower = faster. | Y-axis in frontier plot. |
| **TAFAR** | *Time-averaged false-alarm rate* – proportion of time in false-alarm state (log-scaled). Lower = fewer false alarms. | X-axis (log) in frontier plot. |

---

### 3 · What patterns to expect
| Dial this… | …and you’ll usually see | Why |
|------------|-------------------------|-----|
| **↑ `h` threshold** | ↓ hit-rate · ↓ TAFAR · ↑ LADD | Need a bigger cumulative drift to trigger → fewer & later alarms. |
| **↑ Δ (design shift)** | ↓ false alarms on small shifts · slight ↑ delay on the exact Δ you tuned away from | Detector becomes *selective* for larger moves. |
| **↑ Warm-up `W`** | ↓ false alarms after resets · ↑ delay just after a real shift | Longer window stabilises μ₀/σ₀ estimates but keeps detector “blind” longer. |
| **↑ Gap \|μ₁ − μ₀\| or \|σ₁/σ₀\|** | ↑ hit-rate · ↓ LADD · ↓ TAFAR | Bigger post-shift jump ⇒ drift accumulates faster ⇒ easier detection. |
| **↑ `p` (more frequent shifts)** | Frontier bends up/right (harder to keep low TAFAR & LADD simultaneously) | Less time to “cool-off” → detector often re-arms during a shift. |
| **Turn on *AR(1) / GARCH* pre-whitening** | TAFAR drops; LADD may rise slightly | Removes serial correlation so noise looks i.i.d.; alarms less likely to be spurious. |
| **↓ `M` reps** | Noisy clouds & unstable frontier | Each metric is Monte-Carlo-estimated; too few paths adds variance. |

*Use the 3-D scatter to see how hit-rate responds to two knobs at once, then zoom in with the frontier to weigh speed (LADD) against cleanliness (TAFAR).*  
Higher bubbles near the bottom-left corner are your sweet-spot parameter sets.
    """
    )
