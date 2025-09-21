import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("A model for chemical–mechanical polishing of a material surface based on contact mechanics\n J.J. Vlassak")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("パラメーター設定")
    line_width = st.number_input("ライン幅 [µm]", value=20.0, format="%.3f")
    period     = st.number_input("周期長 L [µm]", value=100.0, format="%.3f")
    sigma      = st.number_input("粗さ σ [µm]", value=0.005, format="%.6f")
    Eeff       = st.number_input("E/(1-ν²) [MPa]", value=45.0, format="%.3f")
    p_app_kPa  = st.number_input("外部圧 [kPa]", value=15.0, format="%.3f")
    sel        = st.number_input("選択比 metal:dielectric", value=3.0, format="%.2f")

    times_str  = st.text_input("解析時刻 [s, カンマ区切り]", "0.0,1.2,3.6,6.0,9.6,24.0")
    dt         = st.number_input("時間刻み Δt [s]", value=0.01, format="%.6f")
    N          = int(st.number_input("空間分割数 N（偶数推奨）", value=1024, step=1, min_value=2))

    st.markdown("---")
    st.subheader("パターン設定")
    pattern_mode = st.selectbox(
        "材料配置モード",
        ["Single line (centered)", "Multi lines (uniform)", "Custom intervals"],
        index=0
    )

    n_lines = w_um = pitch_um = x_offset = None
    custom_intervals_str = ""
    if pattern_mode == "Multi lines (uniform)":
        n_lines   = int(st.number_input("本数 n", value=3, step=1))
        w_um      = st.number_input("各ライン幅 [µm]", value=float(line_width), format="%.3f")
        pitch_um  = st.number_input("ピッチ [µm]（中心間隔）", value=40.0, format="%.3f")
        x_offset  = st.number_input("中心オフセット [µm]", value=0.0, format="%.3f")
    elif pattern_mode == "Custom intervals":
        st.caption("金属区間をカンマ区切りで列挙。例: -40:-30, -5:5, 25:35（端をまたぐなら 30:-30 など）")
        custom_intervals_str = st.text_input("金属区間 [µm]", value="")

    with st.expander("数値安定化（必要に応じて調整）", expanded=False):
        max_iter = int(st.number_input("固定点反復の上限", value=300, step=10, min_value=1))
        tol_rel  = st.number_input("収束相対閾値", value=1e-6, format="%.6g", key="tol_rel")
        exp_clip = st.number_input("exp の指数クリップ幅", value=80.0, format="%.3f", key="exp_clip")
        p_cap    = st.number_input("圧力上限 [kPa]（0で無効）", value=0.0, format="%.3f", key="p_cap")
        beta     = st.number_input("反復緩和 β（0<β≤1）", value=0.3, format="%.3f", key="beta")

    # 除去レート（Fig.2: 金属 = 5 nm/s）から α を自動算出
    metal_rate_nms = 5.0
    alpha = (metal_rate_nms * 1e-3) / (sel * p_app_kPa)

    run = st.button("計算開始", type="primary", use_container_width=True)

# ---------- helpers ----------
def parse_times(s):
    ts = []
    for t in s.split(","):
        t = t.strip()
        if t:
            try:
                ts.append(float(t))
            except:
                pass
    ts.append(0.0)
    ts = sorted(set([round(v, 12) for v in ts]))
    return ts

def fft_modes(N):
    return np.fft.fftfreq(N) * N

def pad_deflection_fft(p_kPa, period, Eeff, m, spec_mult):
    p = p_kPa * 1e-3  # MPa
    P = np.fft.fft(p)
    W = spec_mult * P
    w = np.fft.ifft(W).real
    return w - w.mean()

def _interval_mask_on_period(x, L, a, b):
    return (x >= a) & (x <= b)

def _wrap_intervals(a, b, L):
    if a <= b:
        return [(a, b)]
    else:
        return [(a,  L/2.0), (-L/2.0, b)]

def build_k_array(x, period, sel, pattern_mode, line_width,
                  n_lines=None, w_um=None, pitch_um=None, x_offset=0.0,
                  custom_intervals_str=""):
    k = np.ones_like(x, dtype=float)
    if pattern_mode == "Single line (centered)":
        mask = np.abs(x) < (line_width / 2.0)
        k[mask] = sel
        return k
    if pattern_mode == "Multi lines (uniform)":
        if n_lines is None or w_um is None or pitch_um is None:
            mask = np.abs(x) < (line_width / 2.0)
            k[mask] = sel
            return k
        idxs = np.arange(int(n_lines)) - (int(n_lines) - 1) / 2.0
        centers = x_offset + idxs * pitch_um
        half = w_um / 2.0
        for c in centers:
            a, b = c - half, c + half
            for aa, bb in _wrap_intervals(a, b, period):
                k[_interval_mask_on_period(x, period, aa, bb)] = sel
        return k
    if pattern_mode == "Custom intervals":
        txt = custom_intervals_str.strip()
        if txt:
            parts = [p.strip() for p in txt.split(",")]
            for p in parts:
                if ":" in p:
                    a_str, b_str = p.split(":")[:2]
                    try:
                        a = float(a_str)
                        b = float(b_str)
                    except:
                        continue
                    if a <= b:
                        for aa, bb in [(a, b)]:
                            k[_interval_mask_on_period(x, period, aa, bb)] = sel
                    else:
                        for aa, bb in _wrap_intervals(a, b, period):
                            k[_interval_mask_on_period(x, period, aa, bb)] = sel
        return k
    mask = np.abs(x) < (line_width / 2.0)
    k[mask] = sel
    return k

# ---------- main ----------
if run:
    x0 = -period / 2.0
    x  = np.linspace(x0, x0 + period, N, endpoint=False)
    dx = period / N

    if pattern_mode == "Multi lines (uniform)":
        k = build_k_array(
            x, period, sel,
            pattern_mode=pattern_mode,
            line_width=line_width,
            n_lines=int(n_lines),
            w_um=float(w_um),
            pitch_um=float(pitch_um),
            x_offset=float(x_offset),
        )
    elif pattern_mode == "Custom intervals":
        k = build_k_array(
            x, period, sel,
            pattern_mode=pattern_mode,
            line_width=line_width,
            custom_intervals_str=custom_intervals_str,
        )
    else:
        k = build_k_array(
            x, period, sel,
            pattern_mode="Single line (centered)",
            line_width=line_width,
        )

    m = fft_modes(N)
    abs_m = np.abs(m)
    spec_mult = np.zeros_like(abs_m, dtype=float)
    nonzero = abs_m > 0
    # Fig.2 スケール合わせの経験則：先頭 2 を付けない
    spec_mult[nonzero] = (period / (np.pi * Eeff)) * (1.0 / abs_m[nonzero])

    S = np.zeros_like(x)
    p_kPa = np.full_like(x, p_app_kPa)
    times_req = parse_times(times_str)
    tcur, tmax = 0.0, float(times_req[-1])
    idx_save = 0
    saved = {}

    while tcur <= tmax + 1e-12:
        p = p_kPa.copy()
        target_mean = p_app_kPa

        for _ in range(int(max_iter)):
            w = pad_deflection_fft(p, period, Eeff, m, spec_mult)
            d = w - S
            inv_sigma = 1.0 / max(sigma, 1e-12)
            a = -np.maximum(d, 0.0) * inv_sigma
            a = np.where(d > 0.0, a, -1e6)
            am = np.max(a)
            a_shift = np.clip(a - am, -exp_clip, exp_clip)
            p_scaled = np.exp(a_shift)
            mean_scaled = p_scaled.mean()
            if mean_scaled < 1e-300 or not np.isfinite(mean_scaled):
                p_new = np.full_like(p, target_mean)
            else:
                p_new = (target_mean / mean_scaled) * p_scaled
            if p_cap > 0.0:
                p_new = np.minimum(p_new, p_cap)
                mtmp = p_new.mean()
                if mtmp > 0:
                    p_new *= target_mean / mtmp
                else:
                    p_new = np.full_like(p, target_mean)
            p_next = (1.0 - beta) * p + beta * p_new
            r = np.linalg.norm(p_next - p) / (np.linalg.norm(p) + 1e-12)
            p = p_next
            if r < tol_rel:
                break

        pm = p.mean()
        if pm > 0:
            p *= target_mean / pm
        p_kPa = p
        w = pad_deflection_fft(p_kPa, period, Eeff, m, spec_mult)
        d = w - S

        while idx_save < len(times_req) and (abs(tcur - times_req[idx_save]) < 1e-9 or tcur > times_req[idx_save]):
            tkey = times_req[idx_save]
            saved[tkey] = {"x_um": x.copy(), "S_um": S.copy(), "w_um": w.copy(), "d_um": d.copy(), "p_kPa": p_kPa.copy()}
            idx_save += 1

        S -= dt * alpha * k * p_kPa
        tcur += dt

    # ---- 可視化（Pad→Gap） ----
    st.header("結果表示")
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True)
    for t in times_req:
        v = saved.get(t)
        if v is None:
            continue
        axes[0].plot(v["x_um"], v["S_um"], label=f"{t:.3f}s")
        axes[1].plot(v["x_um"], v["p_kPa"], label=f"{t:.3f}s")
        axes[2].plot(v["x_um"], v["w_um"], label=f"{t:.3f}s")
    axes[0].set_ylabel("Wafer S(x) [µm]")
    axes[1].set_ylabel("Pressure p(x) [kPa]")
    axes[2].set_ylabel("Pad w(x) [µm]")
    axes[1].set_xlabel("x [µm]")
    for a in axes:
        a.grid(True)
        a.legend(ncol=2, fontsize=9)
    st.pyplot(fig)

    # ---- CSV 出力（d を含む） ----
    st.subheader("データ保存（CSV）")
    cols = st.columns(3)
    i = 0
    for t in times_req:
        v = saved.get(t)
        if v is None:
            continue
        df = pd.DataFrame(v)
        csv = df.to_csv(index=False)
        with cols[i % 3]:
            st.download_button(
                label=f"t={t:.3f}s のCSVをダウンロード",
                data=csv,
                file_name=f"profiles_t{t:.3f}s.csv",
                mime="text/csv",
                use_container_width=True
            )
        i += 1
