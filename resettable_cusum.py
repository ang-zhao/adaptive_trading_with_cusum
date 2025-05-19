import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# ------------------------------------------------------------
# 1.  Detector (mean‑down + variance‑up)
# ------------------------------------------------------------
class ResettableCUSUM:
    """
    Resettable quickest-detection monitor that can track
      • mean-down     (μ↓)
      • mean-up       (μ↑)
      • variance-up   (σ↑)
      • variance-down (σ↓)

    Parameters
    ----------
    mu0, sigma0 : float
        Initial in-control mean and st.dev.
    delta_mu, delta_sig : float
        Design shifts |Δμ| and |Δσ| expressed in units of σ₀.
    targets : set[str]
        Any subset of {"mu_down","mu_up","sig_up","sig_down"}.
    h_* : float or None
        Thresholds.  If None, the corresponding statistic is skipped.
    warmup : int
        Re-estimation window length after each alarm.
    """
    def __init__(self,
                 mu0, sigma0,
                 delta_mu=0.4, delta_sig=0.4,
                 targets=frozenset({"mu_down","sig_up"}),
                 h_mu_dn=8.0, h_mu_up=None,
                 h_sig_up=8.0, h_sig_dn=None,
                 warmup=50):
        self.mu0, self.s0 = mu0, sigma0
        self.dm, self.ds = delta_mu, delta_sig
        self.targets     = frozenset(targets)
        self.h_dn  = h_mu_dn
        self.h_up  = h_mu_up
        self.h_su  = h_sig_up
        self.h_sd  = h_sig_dn
        self.W     = warmup

        # initialise statistics only for requested targets
        self.S_dn = self.S_up = self.S_su = self.S_sd = 0.0
        self.buffer = []
        self.change_points = []

    # ------- one-step log-likelihood increments -----------------
    def _inc_mu_down(self, x):
        return ( self.dm / self.s0**2) * (x - self.mu0 - 0.5*self.dm)
    def _inc_mu_up(self, x):
        return (-self.dm / self.s0**2) * (x - self.mu0 + 0.5*self.dm)
    def _inc_sig_up(self, x):
        r = (x - self.mu0)**2 / self.s0**2
        return -0.5*np.log(1+self.ds) + 0.5*self.ds/(1+self.ds)*r
    def _inc_sig_down(self, x):
        r = (x - self.mu0)**2 / self.s0**2
        return -0.5*np.log(1-self.ds) + 0.5*self.ds/(1-self.ds)*r

    # ------- main update ---------------------------------------
    def update(self, x, t):
        # warm-up blackout
        if self.buffer:
            self.buffer.append(x)
            if len(self.buffer) == self.W:
                d = np.asarray(self.buffer)
                self.mu0, self.s0 = d.mean(), d.std(ddof=1)
                self.buffer = []
            return False

        alarm = False
        # mean down
        if "mu_down" in self.targets and self.h_dn is not None:
            self.S_dn = max(0.0, self.S_dn + self._inc_mu_down(x))
            alarm |= (self.S_dn >= self.h_dn)
        # mean up
        if "mu_up" in self.targets and self.h_up is not None:
            self.S_up = max(0.0, self.S_up + self._inc_mu_up(x))
            alarm |= (self.S_up >= self.h_up)
        # var up
        if "sig_up" in self.targets and self.h_su is not None:
            self.S_su = max(0.0, self.S_su + self._inc_sig_up(x))
            alarm |= (self.S_su >= self.h_su)
        # var down
        if "sig_down" in self.targets and self.h_sd is not None:
            self.S_sd = max(0.0, self.S_sd + self._inc_sig_down(x))
            alarm |= (self.S_sd >= self.h_sd)

        if alarm:
            self.change_points.append(t)
            # reset all active stats
            self.S_dn = self.S_up = self.S_su = self.S_sd = 0.0
            self.buffer = [x]
            return True
        return False

# ------------------------------------------------------------
# 2.  Recurrent stream with labelled change types
# ------------------------------------------------------------
def stream_with_types(T, p, mu0, mu1, sig0, sig1, rng):
    """
    Returns:
        data  – np.ndarray length T
        breaks – list of tuples (tau, type) where type in
                 {'mu_down','mu_up','sig_up','sig_down'}
    """
    data = []
    breaks = []
    t = 0
    cur_mu, cur_sig = mu0, sig0
    while t < T:
        gap = int(min(rng.geometric(p), T - t))
        data.extend(rng.normal(cur_mu, cur_sig, gap))
        t += gap
        if t >= T:
            break
        # flip mean or variance with equal probability
        if rng.random() < 0.5:
            new_mu = mu1 if cur_mu == mu0 else mu0
            breaks.append((t, "mu_down" if new_mu < cur_mu else "mu_up"))
            cur_mu = new_mu
        else:
            new_sig = sig1 if cur_sig == sig0 else sig0
            breaks.append((t, "sig_up" if new_sig > cur_sig else "sig_down"))
            cur_sig = new_sig
    return np.array(data), breaks

# ------------------------------------------------------------
# 3.  Monte‑Carlo wrapper
# ------------------------------------------------------------
def evaluate_grid(h_vals, W_vals, targets,
                  M=40, T=12000, p=1/600,
                  mu0=0.5, mu1=-0.5, sig0=1.0, sig1=1.4,
                  **detector_kwargs):          # <── NEW
    """
    detector_kwargs  -- any extra named args forwarded to ResettableCUSUM,       
    e.g. delta_mu=0.7, h_sig=4, etc.
    """
    
    rows = []
    rng0 = np.random.default_rng(0)
    for h in h_vals:
        for W in W_vals:
            eligible, detected = 0, 0
            delays, hits, misses, fa = [], 0, 0, 0
            for m in range(M):
                rng = np.random.default_rng(rng0.integers(1e9))
                x, cps = stream_with_types(T, p, mu0, mu1, sig0, sig1, rng)

                det = ResettableCUSUM(mu0, sig0, 0.4, 0.4, h, h, W)
                cp_iter = iter(cps)
                next_tau, next_type = next(cp_iter, (None, None))
                for t, xi in enumerate(x, 1):
                    if det.update(xi, t):
                        if next_tau is not None and t >= next_tau:
                            if next_type in targets:
                                delays.append(t - next_tau)
                                hits += 1
                            # step to next break
                            next_tau, next_type = next(cp_iter, (None, None))
                        else:
                            fa += 1

                # -------- Sanity check: eligible share ≈ 0.50; prob alarm before next flip ≈ 0.65-0.75
                # det.change_points now holds every alarm time in that path
                # Example: compute delays for all target breaks
                for i, (tau, typ) in enumerate(cps):
                    if typ not in targets:
                        continue
                    eligible += 1
                    end_of_reg = cps[i+1][0] if i+1 < len(cps) else T + 1
                    alarm = next((a for a in det.change_points if a >= tau), None)
                    if alarm is not None and alarm < end_of_reg:
                        detected += 1

                # eligible_share = eligible / len(cps)
                # p_alarm        = detected / eligible if eligible else np.nan
                # print(f"[h={h}, W={W}] eligible={eligible_share:.2f}, "
                #     f"p_alarming_in_time={p_alarm:.2f}")
                # print("alarms:", det.change_points[:10])
                # print("first few breaks:", cps[:5])
                # --------------------------------------------------------------------------------------
                
                # count remaining target breaks as misses
                while next_tau is not None:
                    if next_type in targets:
                        misses += 1
                    next_tau, next_type = next(cp_iter, (None, None))
            hit_rate = hits / (hits + misses) if (hits + misses) else np.nan
            elig_share = eligible / len(cps)/M    # for info
            p_alarm    = detected / eligible if eligible else np.nan
            rows.append(dict(h_mu=h, W=W,
                             LADD=np.mean(delays) if delays else np.nan,
                             hit_rate=hit_rate,
                             TAFAR=fa / (M*T),
                             elig_share=elig_share,
                             p_alarm_in_time=p_alarm))
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4.  Plots
# ------------------------------------------------------------

# ------------------------------------------------------------------
# 4.1 helper to build dataframe of many paths
def simulate_paths(N=100, T=3000, p=1/600,
                   mu0=0.1, mu1=-0.1, sig0=0.1, sig1=0.3, seed=42):
    rng0 = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        rng = np.random.default_rng(rng0.integers(1e9))
        x, br = stream_with_types(T, p, mu0, mu1, sig0, sig1, rng)
        rows.append(dict(path=i, returns=x, breaks=br))
    return pd.DataFrame(rows)

# Plotting utility
def plot_paths(df, targets, show_breaks=False, max_paths=200, title="Recurrent scenarios"):
    T = len(df.iloc[0]['returns'])
    x = np.arange(T)
    plt.figure(figsize=(10,4))
    for _, row in df.head(max_paths).iterrows():
        plt.plot(x, row['returns'], linewidth=0.5, alpha=0.12)
        if show_breaks:
            for tau, typ in row['breaks']:
                if typ in targets:
                    plt.axvline(tau, color='red', alpha=0.3, linestyle='--')
    plt.xlabel("Time"); plt.ylabel("Return")
    plt.title(f"{title} – {min(max_paths,len(df))} paths")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# demo_df = simulate_paths(N=100, T=1200)
# plot_paths(demo_df, TARGETS, show_breaks=True, title="Sample recurrent stream")

# ------------------------------------------------------------------
# 4.2 combined trade‑off plot: each W as its own line ----
# plt.figure(figsize=(6.5, 4.5))

# for W in W_vals:
#     sub = results[results['W'] == W].sort_values('TAFAR')
#     plt.plot(sub['TAFAR'], sub['LADD'], marker='o', label=f"W = {W}")
#     for _, r in sub.iterrows():
#         plt.text(r['TAFAR'], r['LADD'], f"h={int(r['h_mu'])}", fontsize=7)

# plt.xscale('log')
# plt.xlabel("TAFAR (log)")
# plt.ylabel("LADD")
# plt.title("TAFAR–LADD trade‑off for each warm‑up W")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Warm‑up")
# plt.tight_layout()
# plt.show()

# # heat‑map of hit‑rate
# pivot = results.pivot(index="W", columns="h_mu", values="hit_rate")
# plt.figure(figsize=(6,3.6))
# plt.imshow(pivot, origin='lower', aspect='auto', cmap='viridis')
# plt.colorbar(label="Hit rate (targets only)")
# plt.xticks(range(len(h_vals)), h_vals)
# plt.yticks(range(len(W_vals)), W_vals)
# plt.xlabel("Threshold $h_{\\mu}$")
# plt.ylabel("Warm‑up $W$")
# plt.title("Hit‑rate heat‑map (targets only)")
# plt.tight_layout()
# plt.show()

# print("making plot 2")
# # ---- compact line plot (hit‑rate vs h) ----
# plt.figure(figsize=(6.5, 4.5))

# for W in W_vals:
#     sub = results[results['W'] == W].sort_values('TAFAR')
#     plt.plot(sub["h_mu"], sub["hit_rate"], marker='o', label=f"W = {W}")
#     for _, r in sub.iterrows():
#         plt.text(r['TAFAR'], r['LADD'], f"h={int(r['h_mu'])}", fontsize=7)

# plt.xscale('log')
# plt.xlabel("TAFAR (log)")
# plt.ylabel("LADD")
# plt.title("TAFAR–LADD trade‑off for each warm‑up W")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Warm‑up")
# plt.tight_layout()
# plt.show()


# # Lower thresholds and warm-up
# TARGETS  = {"mu_down", "mu_up", "sig_up"}  
# h_vals = [3, 4, 5]
# W_vals = [20, 100]
# det_args = dict(delta_mu=0.4, delta_sig=0.4, h_sig=4, p=1/600)  # lower variance threshold too

# results = evaluate_grid(h_vals, W_vals, TARGETS, M=1, T=12000,
#                         mu0=0.0, mu1=-0.5, sig0=1.0, sig1=1.4,
#                         **det_args)

# print("Simulation summary (hit‑rate only for targets):")
# print(results.to_string(index=False))





