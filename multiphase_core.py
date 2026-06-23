"""
multiphase_core.py — parameterized compute for the multi-phase polymorph competition app.

Pure compute + matplotlib figures (no Streamlit here, so it can be unit-tested).
Everything is driven by a `PP` parameter object so the UI can pass user-edited values.

Layers (same backbone as prototype v3):
  - K_ab : 1-D master-equation steady-state flux (Becker-Doring) + Turnbull-Fisher attachment.
  - graph / path bottleneck analysis.
  - Layer 3 : mean-field fraction-competition ODE  dphi/dt = T(phi,T) phi (non-isothermal).
  - phase field : multi-OP Allen-Cahn + K-driven Poisson seeding; time calibrated to TF kinetics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import solve_ivp

kB = 8.617333262e-5
eV_J = 1.602176634e-19
SHAPE = (36.0 * np.pi) ** (1.0 / 3.0)
_PALETTE = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728",
            "#17becf", "#8c564b", "#e377c2", "#bcbd22"]
# Allen-Cahn numerical constants (shared by the run and the time calibration)
AC_DT, AC_LMOB, AC_KAPPA, AC_GAMMA, AC_DELTA = 0.06, 1.0, 0.5, 1.0, 1.2


class PP:
    """Parameter bundle for N phases."""
    def __init__(self, phases, G, VOL, DIFF, SIGMA, cG, Qd, T_REF=160.0):
        self.phases = list(phases); self.N = len(phases)
        self.G, self.VOL, self.DIFF, self.SIGMA = G, VOL, DIFF, SIGMA
        self.cG, self.Qd, self.T_REF = cG, Qd, T_REF
        self.LAM = {p: 2.5e-10 for p in phases}
        self.idx = {p: i for i, p in enumerate(phases)}
        self.color = {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(phases)}


def default_params(n=6):
    """Defaults for n phases. n==6 reproduces the validated prototype-v3 showcase."""
    if n == 6:
        phases = ["A", "B", "C", "D", "E", "F"]
        G = {"A": 0.0, "B": -0.030, "C": -0.050, "D": -0.070, "E": -0.085, "F": -0.100}
        VOL = {"A": 11.5, "B": 11.8, "C": 12.0, "D": 11.7, "E": 12.1, "F": 11.6}
        DIFF = {"A": 1.3e-15, "B": 1.6e-13, "C": 1.0e-13, "D": 8.0e-14, "E": 5.0e-14, "F": 3.0e-15}
        cG = {"A": 0.0, "B": 1e-4, "C": 2e-4, "D": 1.5e-4, "E": 2.5e-4, "F": 6e-4}
        Qd = {"A": 0.10, "B": 0.08, "C": 0.09, "D": 0.10, "E": 0.12, "F": 0.15}
        pairs = {("A","B"):0.050,("B","C"):0.055,("C","D"):0.060,("D","E"):0.058,("E","F"):0.052,
                 ("A","C"):0.090,("B","D"):0.085,("C","E"):0.095,("D","F"):0.088,
                 ("A","D"):0.130,("B","E"):0.125,("C","F"):0.135,
                 ("A","E"):0.165,("B","F"):0.170,("A","F"):0.210}
        SIGMA = {}
        for (x, y), v in pairs.items():
            SIGMA[(x, y)] = v; SIGMA[(y, x)] = v
        return PP(phases, G, VOL, DIFF, SIGMA, cG, Qd)

    phases = [chr(ord("A") + i) for i in range(n)]
    G = {p: round(-0.10 * i / (n - 1), 4) if n > 1 else 0.0 for i, p in enumerate(phases)}
    VOL = {p: round(11.5 + 0.1 * i, 2) for i, p in enumerate(phases)}
    base = np.linspace(0, 1, n)
    DIFF = {p: float(10 ** (-15 + 2.5 * np.sin(np.pi * base[i]))) for i, p in enumerate(phases)}
    _cg = [0.0, 1e-4, 2e-4, 1.5e-4, 2.5e-4, 6e-4, 3e-4, 4e-4, 5e-4, 5e-4]
    _qd = [0.10, 0.08, 0.09, 0.10, 0.12, 0.15, 0.11, 0.12, 0.13, 0.13]
    cG = {p: _cg[i] for i, p in enumerate(phases)}
    Qd = {p: _qd[i] for i, p in enumerate(phases)}
    SIGMA = {}
    for i, a in enumerate(phases):
        for j, b in enumerate(phases):
            if i < j:
                s = round(0.050 + 0.040 * (j - i - 1), 3)
                SIGMA[(a, b)] = s; SIGMA[(b, a)] = s
    return PP(phases, G, VOL, DIFF, SIGMA, cG, Qd)


# ----------------------------------------------------------------- rates / graph
def G_at(pp, T):
    return {p: pp.G[p] + pp.cG[p] * (T - pp.T_REF) for p in pp.phases}

def D_at(pp, T):
    return {p: pp.DIFF[p] * np.exp(-(pp.Qd[p] / kB) * (1.0 / T - 1.0 / pp.T_REF)) for p in pp.phases}


def transition_rate(pp, a, b, T, Gf=None, Df=None):
    Gf = Gf or pp.G; Df = Df or pp.DIFF
    dmu = Gf[a] - Gf[b]
    if dmu <= 0:
        return dict(J=0.0, Wstar=np.inf, dGstar=np.inf, nstar=np.inf, s=0.0, gamma=0.0, dmu=dmu)
    kT = kB * T; s = dmu / kT
    vb = pp.VOL[b] * 1e-30
    gamma = pp.SIGMA[(a, b)] * SHAPE * vb ** (2.0 / 3.0) / (kT * eV_J)
    nstar = (2.0 * gamma / (3.0 * s)) ** 3
    Wstar = 4.0 * gamma ** 3 / (27.0 * s ** 2); w1 = gamma - s; dGstar = Wstar - w1
    nmax = int(min(8000, max(400, 4.0 * nstar)))
    n = np.arange(1, nmax + 1, dtype=float)
    w = -s * n + gamma * n ** (2.0 / 3.0)
    c1 = 1.0 / (pp.VOL[a] * 1e-30); Omega = 24.0 * Df[a] / pp.LAM[a] ** 2
    f = Omega * n ** (2.0 / 3.0)
    logterms = (w[:-1] - w[0]) - np.log(f[:-1]) - np.log(c1)
    m = logterms.max(); J = float(np.exp(-(m + np.log(np.sum(np.exp(logterms - m))))))
    return dict(J=J, Wstar=Wstar, dGstar=dGstar, w1=w1, nstar=nstar, s=s, gamma=gamma, dmu=dmu)


def build_K(pp, T, Gf=None, Df=None):
    K = np.zeros((pp.N, pp.N))
    for a in pp.phases:
        for b in pp.phases:
            if a == b:
                continue
            r = transition_rate(pp, a, b, T, Gf, Df)
            if r["J"] > 0:
                K[pp.idx[a], pp.idx[b]] = r["J"]
    return K


def all_simple_paths(start, target, adj):
    out = []
    def dfs(node, path):
        if node == target:
            out.append(list(path)); return
        for nb in adj.get(node, []):
            if nb not in path:
                path.append(nb); dfs(nb, path); path.pop()
    dfs(start, [start]); return out


def path_bottleneck(path, K, idx):
    rates = [K[idx[path[i]], idx[path[i + 1]]] for i in range(len(path) - 1)]
    if not rates or min(rates) <= 0:
        return 0.0, None
    bn = min(rates); pos = int(np.argmin(rates))
    return bn, (path[pos], path[pos + 1])


def analyze(pp, K, start):
    adj = {}
    for a in pp.phases:
        for b in pp.phases:
            if a != b and K[pp.idx[a], pp.idx[b]] > 0:
                adj.setdefault(a, []).append(b)
    results = {}
    for target in pp.phases:
        if target == start:
            continue
        scored = []
        for path in all_simple_paths(start, target, adj):
            bn, edge = path_bottleneck(path, K, pp.idx)
            if bn > 0:
                scored.append(dict(path=path, bottleneck=bn, bn_edge=edge))
        scored.sort(key=lambda d: d["bottleneck"], reverse=True)
        results[target] = scored
    return results


def _layout(pp):
    order = sorted(pp.phases, key=lambda p: pp.G[p], reverse=True)   # by free energy
    pos = {}
    for r, p in enumerate(order):
        x = (-1) ** r * (0.4 + 0.18 * (r // 2))
        pos[p] = (x, pp.G[p])
    return pos


def plot_graph(pp, K, best_path, T):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    pos = _layout(pp)
    edges = [(a, b, K[pp.idx[a], pp.idx[b]]) for a in pp.phases for b in pp.phases
             if a != b and K[pp.idx[a], pp.idx[b]] > 0]
    if not edges:
        ax.text(0.5, 0.5, "no downhill edges at this T", ha="center"); return fig
    logs = [np.log10(j) for *_, j in edges]; lo, hi = min(logs), max(logs)
    best_edges = set()
    if best_path:
        p = best_path["path"]; best_edges = {(p[i], p[i + 1]) for i in range(len(p) - 1)}
    for a, b, j in edges:
        lj = np.log10(j); x1, y1 = pos[a]; x2, y2 = pos[b]
        frac = (lj - lo) / (hi - lo + 1e-9); on = (a, b) in best_edges
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8 + 4.5 * frac,
                                    color="#d62728" if on else "#999",
                                    alpha=0.95 if on else 0.3 + 0.4 * frac, shrinkA=16, shrinkB=16))
    for p in pp.phases:
        x, y = pos[p]
        ax.scatter([x], [y], s=1500, c=pp.color[p], edgecolor="k", lw=1.6, zorder=5)
        ax.text(x, y, p, ha="center", va="center", fontsize=14, fontweight="bold", color="w", zorder=6)
    ax.set_title(f"Transition graph — all downhill edges (T={T:.0f} K)\nwidth=log10 K; red=best path", fontsize=12)
    ax.set_ylabel("free energy G (eV/atom)"); ax.set_xticks([]); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); return fig


# ----------------------------------------------------------------- Layer 3 ODE
def _coeffs(pp, T, Gf, Df, nu0, g0, KMAX, DMAX):
    K = build_K(pp, T, Gf=Gf, Df=Df); kT = kB * T
    nu = np.zeros((pp.N, pp.N)); gc = np.zeros((pp.N, pp.N))
    for ia, a in enumerate(pp.phases):
        for ib, b in enumerate(pp.phases):
            if Gf[a] > Gf[b]:
                Kab = K[pp.idx[a], pp.idx[b]]
                nu[ib, ia] = nu0 * (Kab / KMAX) if Kab > 0 else 0.0
                gc[ib, ia] = g0 * ((Gf[a] - Gf[b]) / kT) * (Df[a] / DMAX)
    return nu, gc


def _rhs(phi, nu, gc, N):
    phi = np.clip(phi, 0.0, None)
    Tm = nu + gc * phi[:, None]; np.fill_diagonal(Tm, 0.0)
    Tm[np.diag_indices(N)] = -Tm.sum(axis=0)
    return Tm @ phi


def integrate_fractions(pp, T_of_t, t_total, n_seg, nu0, g0, start, n_sub=8):
    KMAX = build_K(pp, pp.T_REF).max(); DMAX = max(pp.DIFF.values())
    phi = np.zeros(pp.N); phi[pp.idx[start]] = 1.0
    t_edges = np.linspace(0.0, t_total, n_seg + 1)
    ts, traj = [0.0], [phi.copy()]
    for k in range(n_seg):
        Tseg = float(T_of_t(0.5 * (t_edges[k] + t_edges[k + 1])))
        nu, gc = _coeffs(pp, Tseg, G_at(pp, Tseg), D_at(pp, Tseg), nu0, g0, KMAX, DMAX)
        sol = solve_ivp(lambda t, y: _rhs(y, nu, gc, pp.N), (t_edges[k], t_edges[k + 1]), phi,
                        method="BDF", t_eval=np.linspace(t_edges[k], t_edges[k + 1], n_sub),
                        rtol=1e-7, atol=1e-12)
        for j in range(1, sol.y.shape[1]):
            ts.append(sol.t[j]); traj.append(np.clip(sol.y[:, j], 0, None))
        phi = np.clip(sol.y[:, -1], 0, None); phi = phi / phi.sum()
    return np.array(ts), np.array(traj)


def plot_fractions_ode(pp, ts, traj, title, T_vals=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for ip, p in enumerate(pp.phases):
        ax.plot(ts, traj[:, ip], lw=2.2, color=pp.color[p], label=p)
    ax.set_xlabel("elapsed time (arb. units)"); ax.set_ylabel(r"phase fraction $\phi$")
    ax.set_title(title); ax.set_ylim(-0.02, 1.02); ax.legend(ncol=min(pp.N, 8), fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    if T_vals is not None:
        axT = ax.twinx(); axT.plot(ts, T_vals, "k--", lw=1.3, alpha=0.7); axT.set_ylabel("T (K)")
    fig.tight_layout(); return fig


# ----------------------------------------------------------------- phase field + calibration
def _lap(a):
    return np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a


def run_phase_field(pp, T_hot, T_cold, start, nx=96, steps=300, rebuild_every=12,
                    seed_every=12, seed_C=6e-30, disk_r=3, rng_seed=1, record_every=8):
    dt, L_mob, kappa, gamma_int, Delta = AC_DT, AC_LMOB, AC_KAPPA, AC_GAMMA, AC_DELTA
    rng = np.random.default_rng(rng_seed); N = pp.N
    eta = np.zeros((N, nx, nx)); eta[pp.idx[start]] = 1.0
    yy, xx = np.mgrid[-disk_r:disk_r + 1, -disk_r:disk_r + 1]; disk = (xx ** 2 + yy ** 2) <= disk_r ** 2
    rec = list(range(0, steps, record_every)); frames, frac_t, t_axis, T_axis = [], [], [], []
    K = drive = None
    for step in range(steps):
        T = T_hot + (T_cold - T_hot) * (step / max(1, steps - 1))
        if step % rebuild_every == 0:
            Gf = G_at(pp, T); Df = D_at(pp, T); K = build_K(pp, T, Gf=Gf, Df=Df)
            gv = np.array([Gf[p] for p in pp.phases]); drive = (gv - gv.max()) / (gv.max() - gv.min() + 1e-12)
        ss = (eta ** 2).sum(0); new = np.empty_like(eta)
        for p in range(N):
            df = (-eta[p] + eta[p] ** 3 + 2 * gamma_int * eta[p] * (ss - eta[p] ** 2)
                  + Delta * drive[p] * 6.0 * eta[p] * (1.0 - eta[p]))
            new[p] = eta[p] + dt * L_mob * (kappa * _lap(eta[p]) - df)
        eta = np.clip(new, 0.0, 1.0)
        if step % seed_every == 0 and step > 0:
            dom = np.argmax(eta, 0); dv = eta.max(0); Gf = G_at(pp, T)
            for ia, a in enumerate(pp.phases):
                dn = [(ib, K[pp.idx[a], pp.idx[b]]) for ib, b in enumerate(pp.phases)
                      if Gf[a] > Gf[b] and K[pp.idx[a], pp.idx[b]] > 0]
                if not dn:
                    continue
                ktot = sum(k for _, k in dn); p_nuc = 1.0 - np.exp(-seed_C * ktot * seed_every * dt)
                hits = np.argwhere((dom == ia) & (dv > 0.85) & (rng.random((nx, nx)) < p_nuc))
                if len(hits) > 12:
                    hits = hits[rng.choice(len(hits), 12, replace=False)]
                bs = [ib for ib, _ in dn]; ws = np.array([k for _, k in dn]); ws = ws / ws.sum()
                for (yi, xi) in hits:
                    ib = rng.choice(bs, p=ws); ys = (yi + yy) % nx; xs = (xi + xx) % nx
                    for q in range(N):
                        eta[q][ys[disk], xs[disk]] = 0.0
                    eta[ib][ys[disk], xs[disk]] = 1.0
        if step in rec:
            frames.append(np.argmax(eta, 0).copy())
        frac_t.append([(np.argmax(eta, 0) == p).mean() for p in range(N)]); t_axis.append(step * dt); T_axis.append(T)
    return frames, rec, np.array(frac_t), np.array(t_axis), np.array(T_axis)


def _measure_v_sim(pp, a, b, T, nx=240, n_steps=200, window=(40, 180)):
    dt, L_mob, kappa, gamma_int, Delta = AC_DT, AC_LMOB, AC_KAPPA, AC_GAMMA, AC_DELTA
    Gf = G_at(pp, T); gv = np.array([Gf[p] for p in pp.phases])
    drive = (gv - gv.max()) / (gv.max() - gv.min() + 1e-12)
    ia, ib = pp.idx[a], pp.idx[b]; x = np.arange(nx); N = pp.N
    eta = np.zeros((N, nx)); prof = 0.5 * (1 - np.tanh((x - nx * 0.25) / 2.0))
    eta[ib] = prof; eta[ia] = 1 - prof
    lap1 = lambda u: np.roll(u, 1) + np.roll(u, -1) - 2.0 * u
    def ipos():
        d = eta[ib] - 0.5; k = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
        if len(k) == 0:
            return np.nan
        i0 = k[0]; return i0 + d[i0] / (d[i0] - d[i0 + 1])
    pos = []
    for s in range(n_steps):
        sq = (eta ** 2).sum(0); new = np.empty_like(eta)
        for p in range(N):
            df = (-eta[p] + eta[p] ** 3 + 2 * gamma_int * eta[p] * (sq - eta[p] ** 2)
                  + Delta * drive[p] * 6.0 * eta[p] * (1 - eta[p]))
            new[p] = eta[p] + dt * L_mob * (kappa * lap1(eta[p]) - df)
        eta = np.clip(new, 0, 1); pos.append(ipos())
    pos = np.array(pos); s0, s1 = window
    return abs(np.polyfit(np.arange(s0, s1), pos[s0:s1], 1)[0])


def v_phys(pp, a, b, T):
    dmu = (pp.G[a] - pp.G[b]) + (pp.cG[a] - pp.cG[b]) * (T - pp.T_REF)
    return (D_at(pp, T)[a] / pp.LAM[a]) * (1.0 - np.exp(-dmu / (kB * T)))


def calibrate_time(pp, a_ref, b_ref, T_cal, dx=1.0e-9):
    """Return (C_t [s per sim-time], seed_C_phys, diagnostics)."""
    v_sim = _measure_v_sim(pp, a_ref, b_ref, T_cal)
    vp = v_phys(pp, a_ref, b_ref, T_cal)
    C_t = (v_sim / AC_DT) * dx / vp if vp > 0 else np.nan
    seed_C_phys = (dx ** 3) * C_t
    return C_t, seed_C_phys, dict(v_sim=v_sim, v_phys=vp, dx=dx, us_per_step=C_t * AC_DT * 1e6)


def _flabel(t_axis, T_axis, s, C_t):
    return f"t = {t_axis[s] * C_t * 1e3:.2f} ms   T = {T_axis[s]:.0f} K"


def plot_montage(pp, frames, rec, t_axis, T_axis, C_t, title, ncols=3):
    cl = [pp.color[p] for p in pp.phases]; cmap = ListedColormap(cl)
    npick = min(6, len(frames))
    pick = [rec[int(round(x))] for x in np.linspace(0, len(rec) - 1, npick)]
    idxs = [int(np.argmin(np.abs(np.array(rec) - s))) for s in pick]
    nrow = int(np.ceil(npick / ncols))
    fig, axes = plt.subplots(nrow, ncols, figsize=(4 * ncols, 4 * nrow)); axes = np.atleast_1d(axes).ravel()
    for k, j in enumerate(idxs):
        axes[k].imshow(frames[j], cmap=cmap, vmin=0, vmax=pp.N - 1, interpolation="nearest")
        axes[k].set_title(_flabel(t_axis, T_axis, rec[j], C_t), fontsize=11)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    for k in range(npick, len(axes)):
        axes[k].axis("off")
    handles = [plt.Line2D([0], [0], marker="s", ls="", mfc=cl[i], mec="k", ms=11) for i in range(pp.N)]
    fig.legend(handles, pp.phases, loc="lower center", ncol=min(pp.N, 8), fontsize=10)
    fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0.05, 1, 0.97]); return fig


def plot_pf_fractions(pp, frac_t, t_axis, T_axis, C_t):
    tm = t_axis * C_t * 1e3
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for p in range(pp.N):
        ax.plot(tm, frac_t[:, p], lw=2.2, color=pp.color[pp.phases[p]], label=pp.phases[p])
    ax.set_xlabel("elapsed time (ms)"); ax.set_ylabel("area fraction (argmax)")
    ax.set_ylim(-0.02, 1.02); ax.legend(ncol=min(pp.N, 8), fontsize=9, loc="center left"); ax.grid(alpha=0.3)
    axT = ax.twinx(); axT.plot(tm, T_axis, "k--", lw=1.4, alpha=0.7); axT.set_ylabel("temperature (K)")
    ax.set_title("Phase fractions under cooling (dashed = temperature)"); fig.tight_layout(); return fig


def montage_gif_bytes(pp, frames, rec, t_axis, T_axis, C_t, fps=8):
    import matplotlib.animation as animation
    import tempfile, os
    cl = [pp.color[p] for p in pp.phases]; cmap = ListedColormap(cl)
    fig, ax = plt.subplots(figsize=(4.8, 5.0)); ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=pp.N - 1, interpolation="nearest")
    ttl = ax.set_title(_flabel(t_axis, T_axis, rec[0], C_t), fontsize=12)
    def upd(i):
        im.set_data(frames[i]); ttl.set_text(_flabel(t_axis, T_axis, rec[i], C_t)); return [im, ttl]
    anim = animation.FuncAnimation(fig, upd, frames=len(frames), interval=1000 // fps, blit=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False); tmp.close()
    anim.save(tmp.name, writer=animation.PillowWriter(fps=fps)); plt.close(fig)
    with open(tmp.name, "rb") as fh:
        data = fh.read()
    os.unlink(tmp.name)
    return data
