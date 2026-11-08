"""
هسته‌ی پیاده‌سازی DMD و EDMD از صفر برای سری‌های زمانی مالی.
این ماژول توسط dmd_indicator.py و نوت‌بوک import می‌شود.
"""

import numpy as np


def hankel_embed(series: np.ndarray, delay: int) -> np.ndarray:
    """
    series: آرایه‌ی یک‌بعدی (مثلا بازده لگاریتمی)
    delay : طول پنجره‌ی تأخیر (بعد فضای حالت)
    خروجی: ماتریس (delay, n-delay+1) - ردیف آخر هر ستون = جدیدترین نمونه‌ی آن پنجره
    """
    n = len(series)
    if delay >= n:
        raise ValueError("delay باید کوچک‌تر از طول سری باشد")
    cols = n - delay + 1
    H = np.empty((delay, cols))
    for i in range(delay):
        H[i, :] = series[i:i + cols]
    return H


class DMD:
    """DMD دقیق (Exact DMD)."""

    def __init__(self, rank=None):
        self.rank = rank
        self.eigvals = None
        self.modes = None

    def fit(self, X: np.ndarray):
        """X: (n_features, n_snapshots)"""
        X1, X2 = X[:, :-1], X[:, 1:]
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
        r = min(self.rank or len(S), len(S))
        U_r, S_r, V_r = U[:, :r], S[:r], Vh[:r, :].conj().T
        A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1.0 / S_r)
        eigvals, W = np.linalg.eig(A_tilde)
        Phi = X2 @ V_r @ np.diag(1.0 / S_r) @ W
        self.eigvals, self.modes = eigvals, Phi
        return self

    def forecast(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """پیش‌بینی n_steps گام جدید آینده، شروع از حالت شناخته‌شده‌ی x0."""
        b = np.linalg.lstsq(self.modes, x0, rcond=None)[0]
        time_dyn = np.array([self.eigvals ** k for k in range(1, n_steps + 1)]).T
        X_pred = self.modes @ (b[:, None] * time_dyn)
        return np.real(X_pred[-1, :])


def poly_rbf_dictionary(X: np.ndarray, n_rbf_centers: int = 10, rbf_width: float = 1.0) -> np.ndarray:
    """دیکشنری ویژگی غیرخطی برای EDMD: [حالت خام, درجه۲, RBF]"""
    n_features, n_snap = X.shape
    feats = [X, X ** 2]
    rng = np.random.default_rng(0)
    idx = rng.choice(n_snap, size=min(n_rbf_centers, n_snap), replace=False)
    centers = X[:, idx]
    diffs = X[:, :, None] - centers[:, None, :]
    dist2 = np.sum(diffs ** 2, axis=0)
    rbf = np.exp(-dist2 / (2 * rbf_width ** 2)).T
    feats.append(rbf)
    return np.vstack(feats)


class EDMD:
    """EDMD برای تحلیل مودال غیرخطی (فرکانس‌ها و نرخ رشد/میرایی)."""

    def __init__(self, dictionary_fn=poly_rbf_dictionary, rank=None):
        self.dictionary_fn = dictionary_fn
        self.rank = rank
        self.eigvals = None
        self.K = None

    def fit(self, X: np.ndarray):
        Phi_full = self.dictionary_fn(X)
        Phi1, Phi2 = Phi_full[:, :-1], Phi_full[:, 1:]
        U, S, Vh = np.linalg.svd(Phi1, full_matrices=False)
        r = min(self.rank or len(S), len(S))
        U_r, S_r, V_r = U[:, :r], S[:r], Vh[:r, :].conj().T
        K_tilde = U_r.conj().T @ Phi2 @ V_r @ np.diag(1.0 / S_r)
        eigvals, _ = np.linalg.eig(K_tilde)
        self.eigvals = eigvals
        self.K = K_tilde
        return self

    def dominant_frequencies(self, dt: float = 1.0):
        growth_rate = np.log(np.abs(self.eigvals)) / dt
        freq = np.angle(self.eigvals) / (2 * np.pi * dt)
        return growth_rate, freq


if __name__ == "__main__":
    t = np.linspace(0, 20, 400)
    signal = np.sin(2 * np.pi * 0.3 * t) + 0.3 * np.sin(2 * np.pi * 1.1 * t)
    signal += 0.02 * np.random.default_rng(0).standard_normal(len(t))
    X = hankel_embed(signal, delay=15)
    dmd = DMD(rank=6).fit(X)
    print("DMD eigenvalues (magnitude):", np.abs(dmd.eigvals))
    edmd = EDMD(rank=10).fit(X)
    growth, freq = edmd.dominant_frequencies(dt=t[1] - t[0])
    print("EDMD dominant freqs:", np.sort(freq)[:5])
