"""
multiphase_app.py — Streamlit UI for the multi-phase polymorph competition prototype (v3).

Run locally:   streamlit run multiphase_app.py
Deploy:        push to GitHub, then share.streamlit.io -> pick repo -> main file multiphase_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import multiphase_core as C

st.set_page_config(page_title="Polymorph competition", page_icon="🧊", layout="wide")
st.title("🧊 다중상 polymorph 경쟁 — two-step nucleation (prototype v3)")
st.caption("각 전이의 1D master-equation 핵생성률 → 경로 그래프 → 분율 경쟁 ODE → phase-field 미세조직. "
           "파라미터는 Fe 스케일 예시이며 모두 편집 가능합니다.")

# ----------------------------------------------------------------- sidebar: structure
with st.sidebar:
    st.header("구성")
    n = st.slider("상의 개수 N", 2, 8, 6)
    # (re)initialize editable tables when N changes
    if st.session_state.get("_n") != n:
        pp0 = C.default_params(n)
        st.session_state["_n"] = n
        st.session_state["df_phase"] = pd.DataFrame(
            {"G (eV/atom)": [pp0.G[p] for p in pp0.phases],
             "V (Å³)": [pp0.VOL[p] for p in pp0.phases],
             "D (m²/s)": [pp0.DIFF[p] for p in pp0.phases],
             "cG (eV/atom/K)": [pp0.cG[p] for p in pp0.phases],
             "Qd (eV)": [pp0.Qd[p] for p in pp0.phases]},
            index=pp0.phases)
        sig = pd.DataFrame(0.0, index=pp0.phases, columns=pp0.phases)
        for i, a in enumerate(pp0.phases):
            for j, b in enumerate(pp0.phases):
                if i < j:
                    sig.iloc[i, j] = pp0.SIGMA[(a, b)]
        st.session_state["df_sigma"] = sig

    phases = list(st.session_state["df_phase"].index)
    start = st.selectbox("시작 상 (모상)", phases, index=0)
    T_ref = st.number_input("기준 온도 T_ref (K)", 50.0, 2000.0, 160.0, step=10.0)
    st.markdown("---")
    st.caption("아래 탭에서 각 결과를 **버튼**으로 계산합니다.")

# ----------------------------------------------------------------- parameter editors
st.subheader("입력 파라미터")
c1, c2 = st.columns([1.05, 1.0])
with c1:
    st.markdown("**상별 파라미터** — 자유에너지 $G$, 원자부피 $V$, 확산도 $D$, "
                "온도계수 $cG$ ($G(T)=G+cG\\,(T-T_{ref})$), 확산 활성화 $Q_d$.")
    df_phase = st.data_editor(st.session_state["df_phase"], width='stretch', key="ed_phase",
                              column_config={c: st.column_config.NumberColumn(format="%.4g") for c in st.session_state["df_phase"].columns})
with c2:
    st.markdown("**쌍별 계면에너지** $\\sigma_{ab}$ (J/m²) — 상삼각만 사용(대칭). "
                "인접 rank는 낮게, 멀리 건너뛰면 높게 → two-step을 만든다.")
    df_sigma = st.data_editor(st.session_state["df_sigma"], width='stretch', key="ed_sigma",
                              column_config={c: st.column_config.NumberColumn(format="%.3f") for c in st.session_state["df_sigma"].columns})


def build_pp():
    """Reconstruct a PP parameter object from the current editor tables."""
    G = {p: float(df_phase.loc[p, "G (eV/atom)"]) for p in phases}
    VOL = {p: float(df_phase.loc[p, "V (Å³)"]) for p in phases}
    DIFF = {p: float(df_phase.loc[p, "D (m²/s)"]) for p in phases}
    cG = {p: float(df_phase.loc[p, "cG (eV/atom/K)"]) for p in phases}
    Qd = {p: float(df_phase.loc[p, "Qd (eV)"]) for p in phases}
    SIGMA = {}
    for i, a in enumerate(phases):
        for j, b in enumerate(phases):
            if i < j:
                v = float(df_sigma.iloc[i, j])
                SIGMA[(a, b)] = v; SIGMA[(b, a)] = v
    return C.PP(phases, G, VOL, DIFF, SIGMA, cG, Qd, T_REF=T_ref)


pp = build_pp()
most_stable = min(phases, key=lambda p: pp.G[p])

tab1, tab2, tab3, tab4 = st.tabs(["① 전이율 · 그래프", "② 분율 경쟁 (ODE)", "③ 미세조직 (phase field)", "ℹ️ 설명"])

# ----------------------------------------------------------------- tab 1: rates / graph
with tab1:
    st.markdown("각 내리막 전이 $a\\to b$의 핵생성률 "
                "$K_{ab}=\\big[\\sum_n 1/(f_n c_n^{eq})\\big]^{-1}$ (1D master-equation 정상상태, TF 부착). "
                "경로의 **bottleneck**(가장 느린 간선)이 그 경로의 율속이다.")
    Tg = st.number_input("그래프/행렬 온도 (K)", 50.0, 2000.0, float(T_ref), step=10.0, key="Tg")
    if st.button("계산", key="b1"):
        K = C.build_K(pp, Tg)
        if K.max() <= 0:
            st.warning("이 온도에서 내리막 전이가 없습니다 (모든 Δμ ≤ 0). 온도나 G를 조정하세요.")
        else:
            res = C.analyze(pp, K, start)
            dfK = pd.DataFrame(K, index=phases, columns=phases).replace(0, np.nan)
            st.markdown(f"**$K_{{ab}}$ (m⁻³ s⁻¹), T = {Tg:.0f} K**")
            st.dataframe(dfK.style.format("{:.2e}", na_rep="·"), width='stretch')
            rows = []
            for tgt in phases:
                if tgt == start or not res.get(tgt):
                    continue
                b = res[tgt][0]
                rows.append({"target": tgt, "best path": "-".join(b["path"]),
                             "bottleneck (m⁻³s⁻¹)": f"{b['bottleneck']:.2e}",
                             "limiting edge": f"{b['bn_edge'][0]}→{b['bn_edge'][1]}",
                             "# routes": len(res[tgt])})
            if rows:
                st.markdown(f"**시작 {start} → 각 상, 최적(가장 빠른 bottleneck) 경로**")
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            best = res.get(most_stable, [{}])
            best = best[0] if best else None
            st.pyplot(C.plot_graph(pp, K, best, Tg))

# ----------------------------------------------------------------- tab 2: fraction ODE
with tab2:
    st.markdown("평균장 분율 경쟁 $\\dot{\\boldsymbol\\phi}=\\mathbf T(\\boldsymbol\\phi,T)\\boldsymbol\\phi$ "
                "(핵생성 $\\nu_{ab}\\!\\propto\\!K_{ab}$, 성장 $g_{ab}\\!\\propto\\!\\Delta\\mu\\,D_a$, "
                "더 안정한 상에게 잡아먹히는 손실 = 용해). **열처리 경로가 우세상을 바꾼다.**")
    cc1, cc2, cc3 = st.columns(3)
    mode = cc1.radio("열처리", ["등온", "냉각 (ramp)"], key="ode_mode")
    nu0 = cc2.number_input("ν₀ (핵생성 prefactor)", 0.0, 1.0, 2e-3, format="%.4g", key="nu0")
    g0 = cc3.number_input("g₀ (성장 prefactor)", 0.0, 50.0, 2.5, format="%.3g", key="g0")
    if mode == "등온":
        Tiso = cc1.number_input("온도 (K)", 50.0, 2000.0, 180.0, step=10.0, key="Tiso")
        ttot = cc2.number_input("시간 (arb.)", 1.0, 500.0, 40.0, step=5.0, key="tt_iso")
        proto = (lambda t: Tiso, ttot, 1)
        Tvals_fun = None
    else:
        Th = cc1.number_input("T_hot (K)", 50.0, 2000.0, 260.0, step=10.0, key="Th_ode")
        Tc = cc2.number_input("T_cold (K)", 50.0, 2000.0, 120.0, step=10.0, key="Tc_ode")
        ttot = cc3.number_input("총 냉각시간 (arb.)", 1.0, 500.0, 60.0, step=5.0, key="tt_cool")
        ramp = (lambda Th, Tc, tt: (lambda t: Th + (Tc - Th) * (t / tt)))(Th, Tc, ttot)
        proto = (ramp, ttot, 30); Tvals_fun = ramp
    if st.button("ODE 풀기", key="b2"):
        with st.spinner("적분 중..."):
            T_of_t, t_total, n_seg = proto
            ts, tr = C.integrate_fractions(pp, T_of_t, t_total, n_seg, nu0, g0, start,
                                           n_sub=(120 if mode == "등온" else 8))
        dom = phases[int(np.argmax(tr[-1]))]
        ms = min(phases, key=lambda p: C.G_at(pp, float(T_of_t(t_total)))[p])
        st.success(f"최종 우세상: **{dom}**  (그 온도의 최안정상: {ms})  "
                   + ("→ 최안정상과 다르면 *kinetic 선택*" if dom != ms else ""))
        Tv = Tvals_fun(ts) if Tvals_fun is not None else None
        ttl = "isothermal" if mode == "등온" else "cooling"
        st.pyplot(C.plot_fractions_ode(pp, ts, tr, ttl, T_vals=Tv))
        st.dataframe(pd.DataFrame({"phase": phases,
                                   "final fraction": [round(float(tr[-1, i]), 4) for i in range(pp.N)]}),
                     width='stretch', hide_index=True)

# ----------------------------------------------------------------- tab 3: phase field
with tab3:
    st.markdown("다상 Allen–Cahn + $K_{ab}$ 구동 Poisson 시딩. 시간축은 **Turnbull–Fisher 계면 속도**에 "
                "맞춰 보정해 **물리 시간(ms)**으로 표시한다. 냉각 중 미세조직이 핵생성→성장→경쟁→후기변태로 진화한다.")
    pc1, pc2, pc3, pc4 = st.columns(4)
    Th = pc1.number_input("T_hot (K)", 50.0, 2000.0, 240.0, step=10.0, key="Th_pf")
    Tc = pc2.number_input("T_cold (K)", 50.0, 2000.0, 140.0, step=10.0, key="Tc_pf")
    nx = pc3.select_slider("격자 nx", [64, 80, 96, 110, 128], value=96, key="nx_pf")
    steps = pc4.select_slider("스텝 수", [150, 240, 300, 360, 450], value=300, key="steps_pf")
    pe1, pe2 = st.columns(2)
    dx_nm = pe1.number_input("길이 척도 Δx (nm/cell)", 0.1, 10.0, 1.0, step=0.1, key="dx")
    make_gif = pe2.checkbox("애니메이션(gif)도 생성", value=True, key="mkgif")
    st.caption(f"시간 보정 기준 전이: {start}→{most_stable} (가장 안정한 상), T_cal = 중점 온도.")
    if st.button("phase field 실행", key="b3"):
        with st.spinner("계면 속도 보정 + 시뮬레이션 중... (수십 초 소요)"):
            T_cal = 0.5 * (Th + Tc)
            C_t, seedC, diag = C.calibrate_time(pp, start, most_stable, T_cal, dx=dx_nm * 1e-9)
            fr, rec, frac, ta, Ta = C.run_phase_field(pp, Th, Tc, start, nx=nx, steps=steps, seed_C=seedC)
        st.info(f"보정: v_sim={diag['v_sim']:.3f} cells/step, v_phys={diag['v_phys']:.2e} m/s, "
                f"Δx={dx_nm:.1f} nm → **{diag['us_per_step']:.2f} µs/step**, 총 {ta[-1]*C_t*1e3:.2f} ms.")
        dom = phases[int(np.argmax(frac[-1]))]
        st.success(f"최종 우세상: **{dom}**   "
                   f"({', '.join(f'{phases[i]}:{frac[-1,i]:.2f}' for i in range(pp.N) if frac[-1,i] > 0.03)})")
        ttl = f"cooling {Th:.0f}→{Tc:.0f} K — time calibrated to Turnbull–Fisher kinetics"
        st.pyplot(C.plot_montage(pp, fr, rec, ta, Ta, C_t, ttl))
        st.pyplot(C.plot_pf_fractions(pp, frac, ta, Ta, C_t))
        if make_gif:
            with st.spinner("gif 만드는 중..."):
                gif = C.montage_gif_bytes(pp, fr, rec, ta, Ta, C_t)
            st.image(gif, caption="microstructure (physical time + temperature)")
            st.download_button("gif 내려받기", gif, file_name="microstructure.gif", mime="image/gif")

# ----------------------------------------------------------------- tab 4: about
with tab4:
    st.markdown(r"""
**무엇을 계산하나.** 모상에서 시작해, 각 내리막 전이를 1D 핵생성으로 풀어 율 $K_{ab}$를 얻고
(monomer 기준 detailed balance + Turnbull–Fisher 부착; Zeldovich 인자는 flux 합에서 자동으로 나옴),
이 $K$ 행렬을 (i) 경로 그래프, (ii) 분율 경쟁 ODE, (iii) phase field로 확장한다.

**정직한 한계.**
- 파라미터는 Fe 스케일 *예시*다(모두 편집 가능).
- 분율 ODE는 평균장·예시 prefactor($\nu_0,g_0$); phase field 계면 파라미터도 예시.
- phase field **시간축**은 한 대표 전이의 TF 계면 속도와 길이 척도 $\Delta x$로 보정한 **차수 수준**의 물리 시간이다(상대 inter-phase 운동학은 예시).
- 핵심 간선은 2D two-step 엔진(`model_core.py`)으로 정밀화할 수 있다(노트북 prototype v3 참고).

**출처/이론.** prototype v3 노트북과 동일한 backbone. 자세한 수식은 그 노트북의 markdown을 참조.
""")
    if st.button("2D two-step 엔진 사용 가능 여부 확인", key="b4"):
        try:
            import model_core  # noqa: F401
            st.success("model_core.py 가 import 됩니다 — 핵심 간선의 2D 정밀화를 추가할 수 있습니다.")
        except Exception as e:
            st.warning(f"model_core.py 없음: {e}. (선택사항) repo에 추가하면 2D 정밀화 탭을 켤 수 있습니다.")
