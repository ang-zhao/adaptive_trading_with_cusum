import numpy as np
from arch import arch_model  
# ------------------------------------------------------------------
# Pre-whitening decorator for any detector class
# ------------------------------------------------------------------
class Prewhiten:
    """
    method = 'none' | 'ar1' | 'garch'
    W      = window length used to refit the pre-whitening model
    base_cls_kwargs -> forwarded to the underlying detector
    """
    def __init__(self, base_cls, *, method='none', W=50, **base_cls_kwargs):
        self.base_cls      = base_cls
        self.method        = method
        self.W             = W
        self.kw            = base_cls_kwargs
        self.win           = []             # raw data window
        self.phi           = 0.0            # AR(1)
        self.garch_mod     = None           # lazy import
        self.det           = None           # underlying CUSUM
        self.prev_x        = None

    # ---------- helpers -------------------------------------------------
    def _fit_ar1(self):
        x = np.asarray(self.win)
        num = np.dot(x[1:], x[:-1]); den = np.dot(x[:-1], x[:-1])
        self.phi = num/den if den else 0.0

    from arch import arch_model                 # top of file (once)

    # ---------- fit on current window ---------------------------------
    def _fit_garch(self):
        data = np.asarray(self.win)
        gmod = arch_model(data, mean="AR", lags=1,
                        vol="GARCH", p=1, q=1, dist="normal")
        res  = gmod.fit(disp="off")

        # AR(1) coefficient may be absent → default 0
        self.phi   = res.params.get('ar.L1', 0.0)
        self.omega = res.params['omega']
        self.alpha = res.params['alpha[1]']
        self.beta  = res.params['beta[1]']

        # works for both old and new arch versions
        cv = getattr(res, "conditional_volatility",
                    getattr(res, "conditional_vol", None))
        self.last_var = cv[-1]**2



    # ---------- innovation generator ----------------------------------
    def _innovation(self, x):
        if self.method == 'none':
            return x

        if self.method == 'ar1':
            e          = x - self.phi * self.prev_x
            self.prev_x = x
            return e

        if self.method == 'garch':
            var = (self.omega +
                self.alpha * (self.prev_x - self.phi*self.prev_prev_x)**2 +
                self.beta  * self.last_var)
            e   = (x - self.phi*self.prev_x) / np.sqrt(var)
            # shift state
            self.prev_prev_x, self.prev_x, self.last_var = self.prev_x, x, var
            return e
        raise ValueError("Unknown method")


    # ---------- main update --------------------------------------------
    def update(self, x, t):
        # accumulate window until ready
        if self.det is None:
            self.win.append(x)
            if len(self.win) < self.W:
                self.prev_x = x
                self.prev_prev_x = x
                self.prev_var = np.var(self.win) or 1.0
                return False            # not enough data yet
            # fit model & spawn detector
            if self.method == 'ar1':
                self._fit_ar1()
            elif self.method == 'garch':
                self._fit_garch()
            mu0 = np.mean(self.win)
            sig0 = np.std(self.win, ddof=1)
            scale = np.sqrt(1 - self.phi**2) if self.method != 'none' else 1.0
            self.det = self.base_cls(mu0, sig0*scale, **self.kw)
            return False

        # feed innovation to detector
        e = self._innovation(x)
        alarm = self.det.update(e, t)

        # refit model every W points
        self.win.append(x)
        if len(self.win) > self.W:
            self.win.pop(0)
        if t % self.W == 0:
            if self.method == 'ar1':
                self._fit_ar1()
            elif self.method == 'garch':
                self._fit_garch()

        if alarm:
            self.win = []           # clear window → detector will rebuild
            self.det = None
        return alarm
