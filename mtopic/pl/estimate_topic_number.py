import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from ._utils import savefig


_RATIONAL_FN = lambda x, p: (p[0] + p[1] * x) / (1 + p[2] * x + p[3] * x**2)
_RATIONAL_P0 = [1.0, 0.1, 0.1, 0.01]


def _norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi != lo else np.zeros_like(arr)


def _fit_rational(x_norm, y_norm):
    def loss(p):
        try:
            denom = 1 + p[2] * x_norm + p[3] * x_norm**2
            if np.any(denom <= 0):
                return 1e10
            pred = (p[0] + p[1] * x_norm) / denom
            return np.sum((pred - y_norm) ** 2) if np.all(np.isfinite(pred)) else 1e10
        except Exception:
            return 1e10

    best_res, best_loss = None, np.inf
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        res = minimize(
            loss, [v * scale for v in _RATIONAL_P0], method="Nelder-Mead",
            options={"maxiter": 100_000, "xatol": 1e-10, "fatol": 1e-10},
        )
        if res.fun < best_loss:
            best_loss, best_res = res.fun, res

    return best_res.x


def _find_plateau(x_fit, y_fit, threshold_fraction):
    dy = np.gradient(y_fit, x_fit)
    below = np.where(dy <= threshold_fraction * dy.max())[0]
    return x_fit[below[0]] if len(below) else x_fit[-1]


def estimate_topic_number(
    mdata,
    *,
    plateau_threshold=0.05,
    figsize=(6, 5),
    transparent=False,
    save=None,
):
    """
    Fit a rational function to the mean held-out log-likelihood curve and
    identify the optimal number of topics.

    Requires ``mtopic.tl.estimate_topic_number`` to have been run first,
    which stores the CV results in ``mdata.uns["CV_results"]``.

    :param mdata:
        MuData object containing ``mdata.uns["CV_results"]`` from
        ``mtopic.tl.estimate_topic_number``.
    :type mdata: muon.MuData
    :param plateau_threshold:
        Fraction of max slope used to define the plateau onset. Default is ``0.05``.
    :type plateau_threshold: float, optional
    :param figsize:
        Figure size in inches (width, height). Default is ``(6, 5)``.
    :type figsize: tuple, optional
    :param transparent:
        Whether to save the figure with a transparent background. Default is ``False``.
    :type transparent: bool, optional
    :param save:
        Path to save the figure. If ``None``, the figure is shown but not saved. Default is ``None``.
    :type save: str, optional

    :returns:
        ``None``. Fit info is stored in ``mdata.uns["CV_fit_info"]``.
    :rtype: None
    """

    results_df = pd.DataFrame(mdata.uns["CV_results"])

    Ks = list(np.unique(results_df["K"]))
    assert len(Ks) >= 6
    df = results_df[results_df["CV"] != "total"]

    x      = np.asarray(Ks, dtype=float)
    y_mean = df.groupby("K")["total"].mean().reindex(Ks).values

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y_mean.min(), y_mean.max()
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y_mean - y_min) / (y_max - y_min) if y_max != y_min else np.zeros_like(y_mean)

    params = _fit_rational(x_norm, y_norm)

    y_pred_norm = _RATIONAL_FN(x_norm, params)
    ss_res = np.sum((y_norm - y_pred_norm) ** 2)
    ss_tot = np.sum((y_norm - y_norm.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot)

    x_fit_norm = np.linspace(1e-6, 1, 300)
    x_fit      = x_fit_norm * (x_max - x_min) + x_min
    y_fit_raw  = _RATIONAL_FN(x_fit_norm, params) * (y_max - y_min) + y_min
    y_fit      = _norm01(y_fit_raw)
    y_data     = _norm01(y_mean)

    plateau_x = _find_plateau(x_fit, y_fit, plateau_threshold)
    plateau_y = float(np.interp(plateau_x, x_fit, y_fit))
    best_K    = int(round(plateau_x))

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y_data, color="steelblue", s=25, zorder=5, label="Data (mean)")
    ax.plot(x_fit, y_fit, color="deeppink", linewidth=2, label=f"Rational fit  r²={r2:.3f}")
    ax.axvline(plateau_x, color="black", linestyle=":", linewidth=1.4, zorder=4)
    ax.axvspan(plateau_x - 10, plateau_x + 10, alpha=0.12, color="black", zorder=3)
    ax.scatter([plateau_x], [plateau_y], color="black", s=60, zorder=6)
    ax.text(plateau_x + 1.5, plateau_y, f"K={best_K}", fontsize=9, va="center")
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Normalized log-likelihood")
    ax.legend(fontsize=8)

    fig.tight_layout()
    mdata.uns["CV_fit_info"] = {
        "params":  params.tolist(),
        "r2":      r2,
        "best_K":  best_K,
        "x_min":   float(x_min), "x_max":   float(x_max),
        "y_min":   float(y_min), "y_range":  float(y_max - y_min),
    }
    savefig(save, transparent)
    