"""
Interactive Combined Rectangular Footing Design — Streamlit Web Application
Reinforced Concrete Design per ACI 318 (textbook Example 16.3 workflow)
"""

import streamlit as st
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ============================================================
# 1. BACKEND — Combined Rectangular Footing Design Engine
# ============================================================

@dataclass
class FootingInputs:
    # Materials
    fc_prime: float = 3000.0          # psi
    fy: float = 60000.0               # psi
    # Soil / site
    q_allowable: float = 6000.0       # psf
    depth_grade: float = 6.0          # ft
    w_soil_concrete: float = 125.0    # pcf
    surcharge: float = 100.0          # psf
    # Geometry
    L_cc: float = 18.0                # ft
    col_ext_long: float = 18.0        # in
    col_ext_trans: float = 24.0       # in
    col_int_long: float = 24.0        # in
    col_int_trans: float = 24.0       # in
    # Loads (kips)
    D_ext: float = 170.0
    L_ext: float = 130.0
    D_int: float = 250.0
    L_int: float = 200.0
    # Design parameters
    clear_cover: float = 3.5          # in to bar centroid
    step: float = 0.5                 # in (iteration increment)
    max_depth: float = 100.0          # in


@dataclass
class ShearIteration:
    d: float
    V_u_oneway: float
    phi_V_c_oneway: float
    V_u_punching: float
    phi_V_c_punching: float
    oneway_ok: bool
    punching_ok: bool


@dataclass
class FootingResults:
    P_ext_service: float
    P_int_service: float
    P_total_service: float
    P_ext_factored: float
    P_int_factored: float
    P_total_factored: float
    q_e: float
    Area_req: float
    x_bar: float
    L_required: float
    L_actual: float
    B_req: float
    B_actual: float
    q_u: float
    w_u: float
    x_zero_shear: float
    M_u_neg: float            # in-lb
    iterations: List[ShearIteration]
    success: bool
    d_final: float
    total_thickness: float
    int_col_left_face: float
    As_long_neg: float
    As_min_long: float
    As_min_trans_int: float
    As_min_trans_ext: float
    B_eff_int: float
    B_eff_ext: float
    rho: float


def design_footing(inp: FootingInputs) -> FootingResults:
    # --- Service loads ---
    P_ext_service = inp.D_ext + inp.L_ext
    P_int_service = inp.D_int + inp.L_int
    P_total_service = P_ext_service + P_int_service

    # --- Factored loads (ACI 1.2D + 1.6L) ---
    P_ext_factored = 1.2 * inp.D_ext + 1.6 * inp.L_ext
    P_int_factored = 1.2 * inp.D_int + 1.6 * inp.L_int
    P_total_factored = P_ext_factored + P_int_factored

    # --- Effective soil pressure ---
    q_e = inp.q_allowable - (inp.depth_grade * inp.w_soil_concrete + inp.surcharge)
    Area_req = (P_total_service * 1000.0) / q_e

    # --- Resultant / length / width ---
    x_bar = (P_int_service * inp.L_cc) / P_total_service
    ext_face_to_center = (inp.col_ext_long / 2.0) / 12.0
    L_required = 2 * (x_bar + ext_face_to_center)
    L_actual = math.ceil(L_required * 4.0) / 4.0
    B_req = Area_req / L_actual
    B_actual = math.ceil(B_req * 4.0) / 4.0

    # --- Ultimate soil pressure ---
    q_u = P_total_factored / (L_actual * B_actual)
    w_u = q_u * B_actual
    x_zero_shear = P_ext_factored / w_u
    M_u_neg = (w_u * (x_zero_shear**2) / 2.0) - P_ext_factored * (x_zero_shear - ext_face_to_center)
    M_u_neg_in_lb = abs(M_u_neg) * 1000.0 * 12.0

    # --- Shear iteration ---
    phi_shear = 0.75
    int_col_left_face = ext_face_to_center + inp.L_cc - (inp.col_int_long / 2.0) / 12.0
    iterations: List[ShearIteration] = []
    success = False
    d_final = 0.0

    d_guess = 12.0
    while d_guess < inp.max_depth:
        d_ft = d_guess / 12.0
        # one-way shear at d from interior column face
        x_crit = int_col_left_face - d_ft
        V_u_oneway = abs((w_u * x_crit) - P_ext_factored) * 1000.0
        V_c_oneway = 2.0 * math.sqrt(inp.fc_prime) * (B_actual * 12.0) * d_guess
        phi_V_c_oneway = phi_shear * V_c_oneway
        # two-way punching at exterior column
        side_long = (inp.col_ext_long / 12.0) + (d_ft / 2.0)
        side_trans = (inp.col_ext_trans / 12.0) + d_ft
        b_o = (2 * side_long + side_trans) * 12.0
        area_punching = side_long * side_trans
        V_u_punching = (P_ext_factored - q_u * area_punching) * 1000.0
        V_c_punching = 4.0 * math.sqrt(inp.fc_prime) * b_o * d_guess
        phi_V_c_punching = phi_shear * V_c_punching

        oneway_ok = phi_V_c_oneway >= V_u_oneway
        punching_ok = phi_V_c_punching >= V_u_punching
        iterations.append(ShearIteration(
            d_guess, V_u_oneway, phi_V_c_oneway,
            V_u_punching, phi_V_c_punching,
            oneway_ok, punching_ok
        ))
        if oneway_ok and punching_ok:
            success = True
            d_final = d_guess
            break
        d_guess += inp.step

    total_thickness = (d_final + inp.clear_cover) if success else 0.0

    # --- Flexural reinforcement ---
    phi_flexure = 0.9
    b_in = B_actual * 12.0
    R_n = M_u_neg_in_lb / (phi_flexure * b_in * (d_final**2)) if success else 0.0
    rho = (0.85 * inp.fc_prime / inp.fy) * (1.0 - math.sqrt(1.0 - (2.0 * R_n) / (0.85 * inp.fc_prime))) if success else 0.0
    rho = max(rho, 0.0035)
    As_long_neg = rho * b_in * d_final if success else 0.0
    As_min_1 = (3.0 * math.sqrt(inp.fc_prime) / inp.fy) * b_in * d_final if success else 0.0
    As_min_2 = (200.0 / inp.fy) * b_in * d_final if success else 0.0
    As_min_long = max(As_min_1, As_min_2)

    d_trans = (d_final - 1.0) if success else 0.0
    B_eff_int = inp.col_int_long + 2.0 * (d_final / 2.0)
    As_min_trans_int = (200.0 / inp.fy) * B_eff_int * d_trans if success else 0.0
    B_eff_ext = inp.col_ext_long + (d_final / 2.0)
    As_min_trans_ext = (200.0 / inp.fy) * B_eff_ext * d_trans if success else 0.0

    return FootingResults(
        P_ext_service=P_ext_service, P_int_service=P_int_service,
        P_total_service=P_total_service,
        P_ext_factored=P_ext_factored, P_int_factored=P_int_factored,
        P_total_factored=P_total_factored,
        q_e=q_e, Area_req=Area_req, x_bar=x_bar,
        L_required=L_required, L_actual=L_actual,
        B_req=B_req, B_actual=B_actual,
        q_u=q_u, w_u=w_u, x_zero_shear=x_zero_shear,
        M_u_neg=M_u_neg_in_lb,
        iterations=iterations, success=success, d_final=d_final,
        total_thickness=total_thickness,
        int_col_left_face=int_col_left_face,
        As_long_neg=As_long_neg, As_min_long=As_min_long,
        As_min_trans_int=As_min_trans_int, As_min_trans_ext=As_min_trans_ext,
        B_eff_int=B_eff_int, B_eff_ext=B_eff_ext, rho=rho,
    )


# ============================================================
# 2. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Combined Footing Designer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Professional theme ----------
THEME = {
    "navy":   "#1B365D",
    "steel":  "#2E5A88",
    "amber":  "#F59E0B",
    "green":  "#10B981",
    "red":    "#EF4444",
    "slate":  "#475569",
    "bg":     "#F8FAFC",
    "card":   "#FFFFFF",
    "grid":   "#E2E8F0",
}

st.markdown(f"""
<style>
    .reportview-container, .main {{ background-color: {THEME["bg"]}; }}
    h1, h2, h3, h4 {{ color: {THEME["navy"]}; font-weight: 700; }}
    .stMetric label {{ color: {THEME["slate"]}; font-size: 0.85rem; }}
    .stMetric value {{ color: {THEME["navy"]}; font-weight: 700; }}
    .pill {{
        display:inline-block; padding:.25rem .7rem; border-radius:999px;
        font-weight:700; font-size:.8rem; color:white;
    }}
    .card {{
        background:{THEME["card"]}; border-radius:10px; padding:1rem 1.25rem;
        box-shadow:0 1px 3px rgba(0,0,0,0.08); border-left:4px solid {THEME["steel"]};
        margin-bottom:.75rem;
    }}
    .card-title {{ color:{THEME["navy"]}; font-weight:700; font-size:.95rem;
                  text-transform:uppercase; letter-spacing:.04em; margin-bottom:.5rem; }}
    .mono {{ font-family:'Courier New', monospace; font-weight:600; }}
    .section-divider {{
        height:2px; background:linear-gradient(90deg,{THEME["navy"]},{THEME["amber"]});
        border:none; margin:1.5rem 0;
    }}
    .stAlert {{ border-radius:8px; }}
</style>
""", unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'


# ============================================================
# 3. SIDEBAR INPUTS
# ============================================================

with st.sidebar:
    st.markdown(f"## 🏗️ Combined Footing Designer")
    st.caption("ACI 318 — Rectangular Combined Footing")

    with st.expander("📋 Project Information", expanded=True):
        project_name = st.text_input("Project Name", "Basundhara Residential Area")
        engineer     = st.text_input("Engineer of Record", "Md. Rakibul Hasan Mridha")
        date_str     = st.text_input("Date", "2026-01-15")
        job_no       = st.text_input("Job No.", "CE-2025-014")

    with st.expander("🧱 Material Properties", expanded=True):
        fc_prime = st.number_input("f'c — concrete (psi)", 1000, 10000, 3000, 500, key="fc")
        fy       = st.number_input("fy — steel (psi)",   40000, 100000, 60000, 5000, key="fy")

    with st.expander("🌍 Soil & Site Data", expanded=True):
        q_allowable     = st.number_input("Allowable soil pressure qa (psf)", 1000, 20000, 6000, 250)
        depth_grade     = st.number_input("Depth below grade (ft)", 1.0, 20.0, 6.0, 0.5)
        w_soil_concrete = st.number_input("Avg unit wt soil+conc (pcf)", 100, 160, 125, 1)
        surcharge       = st.number_input("Surface surcharge (psf)", 0, 1000, 100, 10)

    with st.expander("📏 Geometry", expanded=True):
        L_cc          = st.number_input("Center-to-center column spacing (ft)", 5.0, 60.0, 18.0, 0.25)
        col_ext_long  = st.number_input("Exterior column — long dim (in)", 8, 48, 18, 1)
        col_ext_trans = st.number_input("Exterior column — trans dim (in)", 8, 48, 24, 1)
        col_int_long  = st.number_input("Interior column — long dim (in)", 8, 48, 24, 1)
        col_int_trans = st.number_input("Interior column — trans dim (in)", 8, 48, 24, 1)

    with st.expander("⬇️ Loading (kips)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Exterior Column**")
            D_ext = st.number_input("Dead (k)", 0, 2000, 170, 5, key="Dext")
            L_ext = st.number_input("Live (k)", 0, 2000, 130, 5, key="Lext")
        with c2:
            st.markdown("**Interior Column**")
            D_int = st.number_input("Dead (k)", 0, 2000, 250, 5, key="Dint")
            L_int = st.number_input("Live (k)", 0, 2000, 200, 5, key="Lint")

    with st.expander("⚙️ Design Parameters", expanded=False):
        clear_cover = st.number_input("Clear cover to bar centroid (in)", 1.0, 6.0, 3.5, 0.25)
        step        = st.number_input("Depth iteration step (in)", 0.25, 2.0, 0.5, 0.25)
        max_depth   = st.number_input("Max search depth (in)", 36, 200, 100, 5)

    run = st.button("▶  Run Design", use_container_width=True, type="primary")

    if st.button("↺ Load Textbook Defaults", use_container_width=True):
        st.rerun()


# ============================================================
# 4. RUN ANALYSIS
# ============================================================

inputs = FootingInputs(
    fc_prime=fc_prime, fy=fy,
    q_allowable=q_allowable, depth_grade=depth_grade,
    w_soil_concrete=w_soil_concrete, surcharge=surcharge,
    L_cc=L_cc,
    col_ext_long=col_ext_long, col_ext_trans=col_ext_trans,
    col_int_long=col_int_long, col_int_trans=col_int_trans,
    D_ext=D_ext, L_ext=L_ext, D_int=D_int, L_int=L_int,
    clear_cover=clear_cover, step=step, max_depth=max_depth,
)

# Run on first load OR when button pressed
if "results" not in st.session_state or run:
    with st.spinner("Solving combined footing..."):
        st.session_state["results"] = design_footing(inputs)
        st.session_state["inputs"]  = inputs

R: FootingResults = st.session_state["results"]
I: FootingInputs  = st.session_state["inputs"]


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(f"""
<div style="background:linear-gradient(135deg,{THEME['navy']},{THEME['steel']});
            padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.25rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:0.85rem; letter-spacing:.15em; opacity:.85;">{job_no} · {date_str}</div>
            <h1 style="color:white; margin:0; font-size:1.6rem;">{project_name}</h1>
            <div style="opacity:.9; font-size:.9rem;">Combined Rectangular Footing Design · {engineer}</div>
        </div>
        <div style="text-align:right;">
            {pill("PASS" if R.success else "FAIL", THEME["green"] if R.success else THEME["red"])}
            <div style="margin-top:.5rem; font-size:.85rem; opacity:.85;">
                h = <b>{R.total_thickness:.1f} in</b><br>
                L × B = <b>{R.L_actual:.2f} × {R.B_actual:.2f} ft</b>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. TABS
# ============================================================

tab_sum, tab_plan, tab_analysis, tab_shear, tab_reinf, tab_report = st.tabs([
    "📊 Summary", "📐 Plan View", "📈 Analysis",
    "✂️ Shear Design", "🧷 Reinforcement", "📑 Calculation Report"
])

# ---------- TAB 1: SUMMARY ----------
with tab_sum:
    if not R.success:
        st.error("⚠️ No feasible depth found within the search bounds. Increase max depth or reduce loads.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Service Load", f"{R.P_total_service:.1f} k")
        st.metric("Total Factored Load", f"{R.P_total_factored:.1f} k")
    with c2:
        st.metric("Effective Soil Pressure qₑ", f"{R.q_e:.0f} psf")
        st.metric("Ultimate Pressure qᵤ", f"{R.q_u:.2f} ksf")
    with c3:
        st.metric("Footing Length L", f"{R.L_actual:.2f} ft")
        st.metric("Footing Width B", f"{R.B_actual:.2f} ft")
    with c4:
        st.metric("Effective Depth d", f"{R.d_final:.1f} in")
        st.metric("Total Thickness h", f"{R.total_thickness:.1f} in")

    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><div class="card-title">📐 Geometry</div>'
                    f'<div class="mono">Required area = {R.Area_req:.2f} ft²</div>'
                    f'<div class="mono">Furnished area = {R.L_actual*R.B_actual:.2f} ft²</div>'
                    f'<div class="mono">Resultant from ext. col = {R.x_bar:.2f} ft</div>'
                    f'<div class="mono">L_required = {R.L_required:.2f} ft → L = {R.L_actual:.2f} ft</div>'
                    f'<div class="mono">B_required = {R.B_req:.2f} ft → B = {R.B_actual:.2f} ft</div>'
                    '</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="card-title">⚖️ Load Summary</div>'
                    f'<div class="mono">P_ext service = {R.P_ext_service:.1f} k · factored = {R.P_ext_factored:.1f} k</div>'
                    f'<div class="mono">P_int service = {R.P_int_service:.1f} k · factored = {R.P_int_factored:.1f} k</div>'
                    f'<div class="mono">P_total service = {R.P_total_service:.1f} k</div>'
                    f'<div class="mono">P_total factored = {R.P_total_factored:.1f} k (1.2D+1.6L)</div>'
                    f'<div class="mono">wᵤ (line load) = {R.w_u:.2f} k/ft</div>'
                    '</div>', unsafe_allow_html=True)


# ---------- TAB 2: PLAN VIEW ----------
def draw_plan(R, I):
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Footing outline
    L_ft, B_ft = R.L_actual, R.B_actual
    ax.add_patch(patches.Rectangle((0, 0), L_ft, B_ft,
                                    linewidth=2.2, edgecolor=THEME["navy"],
                                    facecolor="#EAF2FB", zorder=1))

    # Column positions (centers)
    ext_cx = (I.col_ext_long/2)/12.0
    int_cx = ext_cx + I.L_cc
    col_w_ext = I.col_ext_long/12.0
    col_d_ext = I.col_ext_trans/12.0
    col_w_int = I.col_int_long/12.0
    col_d_int = I.col_int_trans/12.0

    for cx, w, d, label, P_f in [
        (ext_cx, col_w_ext, col_d_ext, "EXT", R.P_ext_factored),
        (int_cx, col_w_int, col_d_int, "INT", R.P_int_factored),
    ]:
        rx = cx - w/2
        ry = B_ft/2 - d/2
        ax.add_patch(patches.Rectangle((rx, ry), w, d,
                                        linewidth=1.8, edgecolor=THEME["amber"],
                                        facecolor=THEME["amber"], alpha=0.85, zorder=3))
        ax.text(cx, B_ft/2, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=THEME["navy"], zorder=4)
        # load arrow
        ax.annotate(f"{P_f:.0f} k",
                    xy=(cx, B_ft + 0.6), ha="center", fontsize=9, color=THEME["navy"],
                    fontweight="bold")
        ax.annotate("", xy=(cx, B_ft + 0.15), xytext=(cx, B_ft + 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=THEME["red"], lw=1.8))

    # Soil pressure arrows (upward)
    for x in np.linspace(0.3, L_ft-0.3, 14):
        ax.annotate("", xy=(x, -0.15), xytext=(x, -0.55),
                    arrowprops=dict(arrowstyle="-|>", color=THEME["green"], lw=1.4))
    ax.text(L_ft/2, -0.9, f"qᵤ = {R.q_u:.2f} ksf (upward)",
            ha="center", color=THEME["green"], fontsize=9, fontweight="bold")

    # Dimensions
    def dim(x1, x2, y, text):
        ax.annotate("", xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
        ax.text((x1+x2)/2, y+0.15, text, ha="center", fontsize=8.5, color=THEME["slate"])

    dim(0, L_ft, -1.6, f"L = {L_ft:.2f} ft")
    dim(ext_cx, int_cx, B_ft + 1.2, f"L_cc = {I.L_cc:.2f} ft")

    # Resultant line
    res_x = ext_cx + R.x_bar
    ax.plot([res_x, res_x], [-0.5, B_ft+0.5], "--", color=THEME["steel"], lw=1.2, zorder=2)
    ax.text(res_x, B_ft+0.95, f"Resultant\nx̄={R.x_bar:.2f} ft",
            ha="center", fontsize=8, color=THEME["steel"])

    # Width dim on right
    ax.annotate("", xy=(L_ft+0.4, 0), xytext=(L_ft+0.4, B_ft),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
    ax.text(L_ft+0.6, B_ft/2, f"B = {B_ft:.2f} ft", rotation=90,
            va="center", fontsize=8.5, color=THEME["slate"])

    ax.set_xlim(-1.5, L_ft+1.8)
    ax.set_ylim(-2.0, B_ft+1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Combined Footing — Plan View (looking down)",
                 fontsize=12, color=THEME["navy"], fontweight="bold", loc="left")
    plt.tight_layout()
    return fig

with tab_plan:
    st.pyplot(draw_plan(R, I))
    st.caption("Orange rectangles = column footprints · red arrows = factored column loads "
               "· green arrows = upward soil reaction · dashed line = resultant location.")


# ---------- TAB 3: ANALYSIS (SFD & BMD) ----------
def draw_sfd_bmd(R, I):
    ext_cx = (I.col_ext_long/2)/12.0
    int_cx = ext_cx + I.L_cc
    L = R.L_actual
    w = R.w_u
    P1, P2 = R.P_ext_factored, R.P_int_factored

    x = np.linspace(0, L, 1000)
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    for i, xi in enumerate(x):
        v = w*xi
        if xi >= ext_cx: v -= P1
        if xi >= int_cx: v -= P2
        V[i] = v
        m = w*xi**2/2
        if xi >= ext_cx: m -= P1*(xi - ext_cx)
        if xi >= int_cx: m -= P2*(xi - int_cx)
        M[i] = m

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    # SFD
    ax1.fill_between(x, V, 0, where=V>=0, color=THEME["steel"], alpha=0.5, label="+V")
    ax1.fill_between(x, V, 0, where=V<0,  color=THEME["amber"], alpha=0.6, label="−V")
    ax1.plot(x, V, color=THEME["navy"], lw=1.8)
    ax1.axhline(0, color="black", lw=0.8)
    for cx, lbl in [(ext_cx, "Ext"), (int_cx, "Int")]:
        ax1.axvline(cx, color=THEME["red"], ls="--", lw=1)
    ax1.axvline(R.x_zero_shear, color=THEME["green"], ls=":", lw=1.6)
    ax1.text(R.x_zero_shear, max(V)*0.85,
             f"  V=0 @ {R.x_zero_shear:.2f} ft", color=THEME["green"], fontsize=9, fontweight="bold")
    ax1.set_ylabel("Shear V (kips)", fontweight="bold", color=THEME["navy"])
    ax1.set_title("Shear Force Diagram (SFD)", color=THEME["navy"], loc="left", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # BMD (engineer convention: tension on top drawn downward)
    ax2.fill_between(x, M, 0, where=M>=0, color=THEME["steel"], alpha=0.5)
    ax2.fill_between(x, M, 0, where=M<0,  color=THEME["amber"], alpha=0.6)
    ax2.plot(x, M, color=THEME["navy"], lw=1.8)
    ax2.axhline(0, color="black", lw=0.8)
    for cx, lbl in [(ext_cx, "Ext"), (int_cx, "Int")]:
        ax2.axvline(cx, color=THEME["red"], ls="--", lw=1)
    M_min = min(M)
    ax2.axvline(R.x_zero_shear, color=THEME["green"], ls=":", lw=1.6)
    ax2.annotate(f"M_max(neg) = {abs(M_min*12):.0f} in-k\n= {R.M_u_neg:.0f} in-lb",
                 xy=(R.x_zero_shear, M_min),
                 xytext=(R.x_zero_shear+1.5, M_min*0.7),
                 arrowprops=dict(arrowstyle="->", color=THEME["navy"]),
                 fontsize=9, color=THEME["navy"], fontweight="bold")
    ax2.set_xlabel("Distance from left edge (ft)", fontweight="bold", color=THEME["navy"])
    ax2.set_ylabel("Moment M (kip-ft)", fontweight="bold", color=THEME["navy"])
    ax2.set_title("Bending Moment Diagram (BMD)", color=THEME["navy"], loc="left", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig

with tab_analysis:
    st.pyplot(draw_sfd_bmd(R, I))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Zero-shear location", f"{R.x_zero_shear:.2f} ft from left edge")
    with c2:
        st.metric("Max negative moment Mᵤ⁻", f"{R.M_u_neg:,.0f} in-lb",
                  delta=f"{R.M_u_neg/12000:.1f} ft-kip")
    with c3:
        st.metric("Ultimate line load wᵤ", f"{R.w_u:.2f} k/ft")


# ---------- TAB 4: SHEAR DESIGN ----------
def draw_shear_iteration(R):
    ds = [it.d for it in R.iterations]
    v_u_ow  = [it.V_u_oneway/1000 for it in R.iterations]
    pv_c_ow = [it.phi_V_c_oneway/1000 for it in R.iterations]
    v_u_pn  = [it.V_u_punching/1000 for it in R.iterations]
    pv_c_pn = [it.phi_V_c_punching/1000 for it in R.iterations]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ds, v_u_ow,  "-o", color=THEME["red"],   lw=1.8, label="Vᵤ (demand)")
    ax1.plot(ds, pv_c_ow, "-s", color=THEME["green"], lw=1.8, label="φVc (capacity)")
    ax1.axvline(R.d_final, color=THEME["navy"], ls=":", lw=1.5)
    ax1.set_xlabel("Effective depth d (in)")
    ax1.set_ylabel("Shear (kips)")
    ax1.set_title("One-Way Shear", color=THEME["navy"], fontweight="bold", loc="left")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.plot(ds, v_u_pn,  "-o", color=THEME["red"],   lw=1.8, label="Vᵤ (demand)")
    ax2.plot(ds, pv_c_pn, "-s", color=THEME["green"], lw=1.8, label="φVc (capacity)")
    ax2.axvline(R.d_final, color=THEME["navy"], ls=":", lw=1.5)
    ax2.set_xlabel("Effective depth d (in)")
    ax2.set_ylabel("Shear (kips)")
    ax2.set_title("Two-Way Punching Shear (ext. col.)", color=THEME["navy"], fontweight="bold", loc="left")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig

with tab_shear:
    st.pyplot(draw_shear_iteration(R))
    st.markdown("#### Iteration History")
    rows = []
    for it in R.iterations:
        rows.append({
            "d (in)": f"{it.d:.2f}",
            "Vᵤ 1-way (k)": f"{it.V_u_oneway/1000:.1f}",
            "φVc 1-way (k)": f"{it.phi_V_c_oneway/1000:.1f}",
            "1-way OK": "✅" if it.oneway_ok else "❌",
            "Vᵤ punch (k)": f"{it.V_u_punching/1000:.1f}",
            "φVc punch (k)": f"{it.phi_V_c_punching/1000:.1f}",
            "Punch OK": "✅" if it.punching_ok else "❌",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if R.success:
        last = R.iterations[-1]
        st.success(
            f"**Design depth d = {R.d_final:.2f} in**  →  "
            f"1-way DCR = {last.V_u_oneway/last.phi_V_c_oneway:.2f}, "
            f"punching DCR = {last.V_u_punching/last.phi_V_c_punching:.2f}. "
            f"Total thickness h = {R.total_thickness:.1f} in (incl. {I.clear_cover:.1f} in cover)."
        )


# ---------- TAB 5: REINFORCEMENT ----------
with tab_reinf:
    st.markdown("#### Required Steel Areas")
    reinf_data = [
        ("Longitudinal — Top (negative moment)",  R.As_long_neg, "in²", f"ρ = {R.rho:.4f}"),
        ("Longitudinal — Bottom (min. steel)",    R.As_min_long, "in²", "max(3√f'c/fy, 200/fy)"),
        ("Transverse — under Interior Column",    R.As_min_trans_int, "in²",
            f"Strip width = {R.B_eff_int:.1f} in"),
        ("Transverse — under Exterior Column",    R.As_min_trans_ext, "in²",
            f"Strip width = {R.B_eff_ext:.1f} in"),
    ]
    for name, val, unit, note in reinf_data:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">{name}</div>'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span class="mono" style="font-size:1.4rem; color:{THEME["navy"]};">{val:.2f} {unit}</span>'
            f'<span style="color:{THEME["slate"]}; font-size:.85rem;">{note}</span>'
            f'</div></div>', unsafe_allow_html=True)

    st.markdown("#### Bar Selection Suggestion (FY = 60 ksi)")
    bar_db = {3:0.375, 4:0.500, 5:0.625, 6:0.750, 7:0.875, 8:1.000,
              9:1.128, 10:1.270, 11:1.410, 14:1.693, 18:2.257}
    bar_A  = {n: math.pi*(d/2)**2 for n, d in bar_db.items()}

    def suggest(As_req, prefer=(7,8,9,10,11)):
        best = None
        for n in prefer:
            A = bar_A[n]
            count = math.ceil(As_req / A + 1e-9)
            if best is None or count*A < best[2]*bar_A[best[0]]:
                best = (n, count, count*A)
        return best

    rows = []
    for label, As_req, length in [
        ("Long. Top",     R.As_long_neg,      R.L_actual),
        ("Long. Bottom",  R.As_min_long,      R.L_actual),
        ("Trans. Int.",   R.As_min_trans_int, R.B_actual),
        ("Trans. Ext.",   R.As_min_trans_ext, R.B_actual),
    ]:
        if As_req <= 0:
            rows.append({"Member": label, "As_req (in²)": "—",
                         "Suggested Bar": "—", "Count": "—", "As_prov (in²)": "—",
                         "Spacing (in)": "—"})
            continue
        n, count, A_prov = suggest(As_req)
        spacing = (length*12 - 6) / (count - 1) if count > 1 else 0
        rows.append({
            "Member": label,
            "As_req (in²)": f"{As_req:.2f}",
            "Suggested Bar": f"#{n}",
            "Count": str(count),
            "As_prov (in²)": f"{A_prov:.2f}",
            "Spacing (in)": f"{spacing:.1f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Bar selections are heuristic — verify development length, spacing, and bar size constraints per ACI 318.")


# ---------- TAB 6: CALCULATION REPORT ----------
with tab_report:
    rep = f"""
# Combined Rectangular Footing — Calculation Report

**Project:** {project_name}  |  **Job No.:** {job_no}
**Engineer:** {engineer}  |  **Date:** {date_str}

---

## 1. Material Properties & Soil Data
- f'c = **{I.fc_prime:.0f} psi**,  fy = **{I.fy:.0f} psi**
- Allowable soil pressure qₐ = **{I.q_allowable:.0f} psf**
- Depth below grade = **{I.depth_grade:.1f} ft**, surcharge = **{I.surcharge:.0f} psf**
- Avg unit weight (soil + conc) = **{I.w_soil_concrete:.0f} pcf**

## 2. Column Loads
| | Service (k) | Factored 1.2D+1.6L (k) |
|---|---:|---:|
| Exterior | {R.P_ext_service:.1f} | {R.P_ext_factored:.1f} |
| Interior | {R.P_int_service:.1f} | {R.P_int_factored:.1f} |
| **Total** | **{R.P_total_service:.1f}** | **{R.P_total_factored:.1f}** |

## 3. Footing Geometry
- Effective soil pressure: qₑ = qₐ − (γ·Df + surcharge)
  = {I.q_allowable:.0f} − ({I.w_soil_concrete:.0f}×{I.depth_grade:.1f} + {I.surcharge:.0f})
  = **{R.q_e:.0f} psf**
- Required area A = P_service / qₑ = {R.P_total_service*1000:.0f} / {R.q_e:.0f} = **{R.Area_req:.2f} ft²**
- Resultant distance from ext. column: x̄ = P_int·L_cc / P_total = **{R.x_bar:.2f} ft**
- Required length L = 2·(x̄ + col_ext_long/2) = **{R.L_required:.2f} ft** → **L = {R.L_actual:.2f} ft**
- Required width B = A / L = **{R.B_req:.2f} ft** → **B = {R.B_actual:.2f} ft**
- Furnished area = **{R.L_actual*R.B_actual:.2f} ft²** (≥ {R.Area_req:.2f} ✓)

## 4. Ultimate Soil Pressure & Structural Analysis
- qᵤ = P_total_factored / (L·B) = **{R.q_u:.3f} ksf**
- Line load wᵤ = qᵤ · B = **{R.w_u:.2f} k/ft**
- Zero-shear location from left edge: x = P_ext / wᵤ = **{R.x_zero_shear:.2f} ft**
- Maximum negative moment: Mᵤ⁻ = wᵤ·x²/2 − P_ext·(x − a) = **{R.M_u_neg:,.0f} in-lb** ({R.M_u_neg/12000:.1f} ft-kip)

## 5. Shear Thickness Design (φ = 0.75)
- One-way shear check at distance d from interior column face: Vc = 2√f'c·b·d
- Two-way punching shear at exterior column: Vc = 4√f'c·b₀·d
- Iterative search converged at **d = {R.d_final:.2f} in**
- Total thickness h = d + cover = {R.d_final:.2f} + {I.clear_cover:.1f} = **{R.total_thickness:.2f} in**
"""
    if R.success:
        last = R.iterations[-1]
        rep += f"""
**Final check at d = {R.d_final:.2f} in:**
- One-way: Vᵤ = {last.V_u_oneway/1000:.1f} k ≤ φVc = {last.phi_V_c_oneway/1000:.1f} k  ✅
- Punching: Vᵤ = {last.V_u_punching/1000:.1f} k ≤ φVc = {last.phi_V_c_punching/1000:.1f} k  ✅
"""
    else:
        rep += "\n**⚠️ No feasible depth found within search bounds.**\n"

    rep += f"""
## 6. Flexural Reinforcement (φ = 0.90)
- Longitudinal top steel (negative moment):
  Rₙ = Mᵤ/(φ·b·d²) = {R.M_u_neg/(0.9*(R.B_actual*12)*(R.d_final**2)):.5f} psi
  ρ = {R.rho:.5f}  →  **As = {R.As_long_neg:.2f} in²**
- Longitudinal bottom steel (min): As = **{R.As_min_long:.2f} in²**
- Transverse strip — interior: As = **{R.As_min_trans_int:.2f} in²** (width {R.B_eff_int:.1f} in)
- Transverse strip — exterior: As = **{R.As_min_trans_ext:.2f} in²** (width {R.B_eff_ext:.1f} in)

---

*Generated by Combined Footing Designer — Streamlit Edition. Verify all results against the governing code before construction.*
"""
    st.markdown(rep)
    st.download_button("⬇ Download Report (Markdown)",
                       data=rep,
                       file_name=f"footing_report_{job_no}.md",
                       mime="text/markdown")
