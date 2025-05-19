import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from resettable_cusum import ResettableCUSUM, stream_with_types
from pre_whiten_ts import Prewhiten

# ────────────────────────────────────────────────────────────────────
# 6. universal grid runner
# ────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# universal grid runner – v2, compatible with 4-target detector - target aware thresholds
# ------------------------------------------------------------------
def run_grid(
    # grids for *all* knobs; omit / keep singleton tuples for unused ones
    p_grid        =(1/600,),
    T_grid        =(12_000,),
    W_grid        =(50,),

    h_mu_dn_grid  =(5,),
    h_mu_up_grid  =(None,),
    h_sig_up_grid =(8,),
    h_sig_dn_grid =(None,),

    delta_mu_grid =(0.4,),
    delta_sig_grid=(0.4,),

    mu0_grid      =(0.5,),
    mu1_grid      =(-0.5,),
    sig0_grid     =(1.0,),
    sig1_grid     =(1.4,),

    targets       =frozenset({"mu_down","sig_up"}),
    M=10, seed=0,

    prewhiten_method='none',    # 'none'|'ar1'|'garch'
    prewhiten_W=50,
):
    rng_master = np.random.default_rng(seed)
    rows = []

    grid = itertools.product(
        p_grid, T_grid, W_grid,
        h_mu_dn_grid, h_mu_up_grid, h_sig_up_grid, h_sig_dn_grid,
        delta_mu_grid, delta_sig_grid,
        mu0_grid, mu1_grid, sig0_grid, sig1_grid
    )

    for (p, T, W,
         h_dn, h_up, h_su, h_sd,
         d_mu, d_sig,
         mu0, mu1, sig0, sig1) in grid:

        # skip thresholds that are None in the grid but branch in targets
        h_dn = h_dn if "mu_down" in targets else None
        h_up = h_up if "mu_up"   in targets else None
        h_su = h_su if "sig_up"  in targets else None
        h_sd = h_sd if "sig_down" in targets else None

        delays, hits, misses, fa = [], 0, 0, 0
        rng0 = np.random.default_rng(rng_master.integers(1e9))

        for _ in range(M):
            rng  = np.random.default_rng(rng0.integers(1e9))
            x, cps = stream_with_types(T, p, mu0, mu1, sig0, sig1, rng)

            det = Prewhiten(
                    ResettableCUSUM,
                    method = prewhiten_method,
                    W      = prewhiten_W,
                    targets=targets,
                    h_mu_dn=h_dn, h_mu_up=h_up,
                    h_sig_up=h_su, h_sig_dn=h_sd,
                    delta_mu=d_mu, delta_sig=d_sig,
                    warmup=W          # pass-through for underlying CUSUM
                  )

            cp_iter = iter(cps)
            next_tau, next_type = next(cp_iter, (None, None))

            for t, xi in enumerate(x, 1):
                if det.update(xi, t):
                    if next_tau is not None and t >= next_tau:
                        if next_type in targets:
                            delays.append(t - next_tau); hits += 1
                        next_tau, next_type = next(cp_iter, (None, None))
                    else:
                        fa += 1

            while next_tau is not None:
                if next_type in targets: misses += 1
                next_tau, next_type = next(cp_iter, (None, None))

        rows.append(dict(
            p=p, T=T, W=W,
            h_mu_dn=h_dn, h_mu_up=h_up,
            h_sig_up=h_su, h_sig_dn=h_sd,
            delta_mu=d_mu, delta_sig=d_sig,
            mu0=mu0, mu1=mu1, sig0=sig0, sig1=sig1,
            LADD=np.mean(delays) if delays else np.nan,
            hit_rate=hits/(hits+misses) if hits+misses else np.nan,
            hits = hits,
            misses = misses,
            TAFAR=fa/(M*T)
        ))

    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────
#  Helper: decide which threshold column to use
# ───────────────────────────────────────────────────────────────
def _pick_threshold_col(df, targets, thresh_col=None):
    """Return a column name to represent 'h' on the x-axis / colour."""
    if thresh_col is not None:
        return thresh_col
    # preference order
    prefs = ["mu_down", "mu_up", "sig_up", "sig_down"]
    mapping = {"mu_down": "h_mu_dn",
               "mu_up"  : "h_mu_up",
               "sig_up" : "h_sig_up",
               "sig_down":"h_sig_dn"}
    for t in prefs:
        if t in targets and mapping[t] in df.columns:
            return mapping[t]
    # fallback: first numeric threshold column present
    for c in ["h_mu_dn","h_mu_up","h_sig_up","h_sig_dn"]:
        if c in df.columns:
            return c
    raise ValueError("No threshold column found in dataframe")

# ───────────────────────────────────────────────────────────────
def tradeoff_spider(df, targets, thresh_col=None):
    hcol = _pick_threshold_col(df, targets, thresh_col)
    fig, ax = plt.subplots(figsize=(7,5))
    uniq = sorted(df[hcol].dropna().unique())
    cmap = plt.cm.get_cmap('viridis', len(uniq))
    for (h,W), sub in df.groupby([hcol,'W']):
        ax.scatter(sub.TAFAR, sub.LADD,
                   s=120*sub.hit_rate,
                   color=cmap(uniq.index(h)),
                   marker={30:'o',50:'s',100:'^'}.get(W,'o'),
                   edgecolors='k', linewidth=.3, alpha=.75,
                   label=f"h={h}, W={W}")
    ax.set_xscale('log'); ax.grid(True, ls='--', alpha=.3)
    ax.set_xlabel("TAFAR (log)"); ax.set_ylabel("LADD")
    ax.set_title("Operating frontier – bubble size ∝ hit-rate")
    ax.legend(ncol=2, fontsize=7)
    plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────────
def facet_heatmap(df, metric, targets, thresh_col=None):
    hcol = _pick_threshold_col(df, targets, thresh_col)
    p_vals, d_vals = sorted(df.p.unique()), sorted(df.delta_mu.unique())
    n_rows, n_cols = len(p_vals), len(d_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3*n_cols,2.5*n_rows),
                             sharex=True, sharey=True)
    for i,p in enumerate(p_vals):
        for j,d in enumerate(d_vals):
            ax   = axes[i,j] if n_rows>1 else axes[j]
            slab = df[(df.p==p)&(df.delta_mu==d)]
            pivot= slab.pivot(index='W', columns=hcol, values=metric)
            im   = ax.imshow(pivot, origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(f"p={p:.4f}, δμ={d:.2f}", fontsize=8)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns)
            ax.set_yticks(range(len(pivot.index)),  pivot.index)
            if i==n_rows-1: ax.set_xlabel(hcol)
            if j==0:        ax.set_ylabel("W")
    fig.colorbar(im, ax=axes.ravel().tolist(), label=metric)
    plt.show()

# ───────────────────────────────────────────────────────────────
def sweep_lines(df, p_val, W_val, targets, thresh_col=None):
    hcol = _pick_threshold_col(df, targets, thresh_col)
    fig, axs = plt.subplots(3,1, figsize=(6,7), sharex=True)
    sub = df[(df.p==p_val)&(df.W==W_val)]
    for d, grp in sub.groupby('delta_mu'):
        axs[0].plot(grp[hcol], grp.hit_rate,'-o', label=f"δμ={d}")
        axs[1].plot(grp[hcol], grp.LADD,    '-o')
        axs[2].plot(grp[hcol], grp.TAFAR,   '-o')
    axs[0].set_ylabel("Hit-rate"); axs[1].set_ylabel("LADD")
    axs[2].set_ylabel("TAFAR");     axs[2].set_xlabel(hcol)
    for ax in axs: ax.grid(True, ls='--', alpha=.3)
    axs[0].legend(title="design shift")
    plt.suptitle(f"Sweep at p={p_val:.4f}, W={W_val}")
    plt.tight_layout(); plt.show()


# Example usage: res = run_grid(targets={"mu_down"}, h_mu_dn_grid=(3,4,5), ...)
# tradeoff_plot(res, warmup_val=50, targets={"mu_down"})
# facet_heatmap(res, metric='hit_rate', targets={"mu_down"})
# tradeoff_spider(res, targets={"mu_down"})


# # ───────────────────────────────────────────────────────────────
# 3 ── For AR/GARCH pre-whitening: “spider” trade-off plot with colour = method ------------------
def spider_multi(df, *, warmup_val=50, targets={'mu_down'},
                 thresh_col=None, palette=('tab:blue','tab:orange','tab:green')):
    hcol = _pick_threshold_col(df, targets, thresh_col)
    fig, ax = plt.subplots(figsize=(7,5))
    meths = df['method'].unique()
    colour = {m:c for m,c in zip(meths, palette)}
    for (m,h), sub in df[df.W == warmup_val].groupby(['method', hcol]):
        ax.scatter(sub.TAFAR, sub.LADD,
                   s=140*sub.hit_rate,
                   color=colour[m], marker='o',
                   edgecolors='k', linewidth=.3, alpha=.75,
                   label=f"{m}  h={h}")
    ax.set_xscale('log');  ax.grid(True, ls='--', alpha=.3)
    ax.set_xlabel("TAFAR (log)"); ax.set_ylabel("LADD")
    ax.set_title(f"Frontier @ W={warmup_val}  – bubble ∝ hit-rate")
    # one legend entry per method
    handles, labels = ax.get_legend_handles_labels()
    by_m = {}
    for h,l in zip(handles,labels):
        m = l.split()[0]
        by_m[m] = h
    ax.legend(by_m.values(), by_m.keys(), title="Pre-whiten", fontsize=8)
    plt.tight_layout(); plt.show()

# 4 ── stacked line plot (hit-rate vs threshold) ---------------------
def sweep_multi(df, p_val, W_val, targets, metric='hit_rate'):
    hcol = _pick_threshold_col(df, targets)
    plt.figure(figsize=(6,4))
    sub = df[(df.p==p_val)&(df.W==W_val)]
    for m,g in sub.groupby('method'):
        plt.plot(g[hcol], g[metric], '-o', label=m)
    plt.xlabel(hcol); plt.ylabel(metric); plt.grid(True, ls='--', alpha=.3)
    plt.title(f'{metric} @ p={p_val:.4f}, W={W_val}')
    plt.legend(); plt.tight_layout(); plt.show()

def spider_all(df, *, targets, thresh_col=None):
    hcol   = _pick_threshold_col(df, targets, thresh_col)
    cmap   = plt.cm.get_cmap('tab10')
    meths  = df['method'].unique()
    shapes = {w:m for w,m in zip(sorted(df.W.unique()),
                                 ['o','s','^','v','D','P','X'])}
    plt.figure(figsize=(7,5))
    for i,m in enumerate(meths):
        sub = df[df.method==m]
        plt.scatter(sub.TAFAR, sub.LADD,
                    s = 80*sub.hit_rate,          # bubble = hit
                    c = [cmap(i)], marker='o',    # color by method
                    alpha=.25, edgecolors='none', label=m)
    plt.xscale('log'); plt.grid(True, ls='--', alpha=.3)
    plt.xlabel("TAFAR (log)"); plt.ylabel("LADD")
    plt.title("All parameter cells – colour: method, size: hit-rate")
    plt.legend(title="Pre-whiten", fontsize=8)
    plt.tight_layout(); plt.show()
