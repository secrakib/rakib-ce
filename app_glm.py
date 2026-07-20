"""
Civil Engineering Footing Designer — Streamlit Web Application
Dual-Backend Integration:
  • Combined Rectangular Footing Design (ACI 318 — Textbook Ex. 16.3)
  • Single Column Square Spread Footing Design (ACI 318 — LRFD)

Project Structure:
  1.  Theme & CSS
  2.  Old Backend  (FootingInputs / FootingResults / design_footing)
  3.  New Backend  (SingleFootingInputs / SingleFootingResults / design_single_footing)
  4.  Page Configuration & Styling
  5.  Helper & Visualization Functions
  6.  Sidebar (shared project information)
  7.  Dashboard Tab
  8.  Old System Tab
  9.  New System Tab
"""

import streamlit as st
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# 1. THEME
# ============================================================

THEME = {
    "navy":        "#1B365D",
    "steel":       "#2E5A88",
    "amber":       "#F59E0B",
    "green":       "#10B981",
    "red":         "#EF4444",
    "slate":       "#475569",
    "bg":          "#F8FAFC",
    "card":        "#FFFFFF",
    "grid":        "#E2E8F0",
    "teal":        "#0D9488",
    "purple":      "#7C3AED",
    "light_blue":  "#EAF2FB",
    "light_amber": "#FEF3C7",
    "light_green": "#D1FAE5",
    "light_red":   "#FEE2E2",
}

# ============================================================
# 2. OLD BACKEND — Combined Rectangular Footing Design
# ============================================================

@dataclass
class FootingInputs:
    fc_prime: float = 3000.0
    fy: float = 60000.0
    q_allowable: float = 6000.0
    depth_grade: float = 6.0
    w_soil_concrete: float = 125.0
    surcharge: float = 100.0
    L_cc: float = 18.0
    col_ext_long: float = 18.0
    col_ext_trans: float = 24.0
    col_int_long: float = 24.0
    col_int_trans: float = 24.0
    D_ext: float = 170.0
    L_ext: float = 130.0
    D_int: float = 250.0
    L_int: float = 200.0
    clear_cover: float = 3.5
    step: float = 0.5
    max_depth: float = 100.0


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
    M_u_neg: float
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
    """Old backend: Combined rectangular footing design per ACI 318."""
    P_ext_service = inp.D_ext + inp.L_ext
    P_int_service = inp.D_int + inp.L_int
    P_total_service = P_ext_service + P_int_service

    P_ext_factored = 1.2 * inp.D_ext + 1.6 * inp.L_ext
    P_int_factored = 1.2 * inp.D_int + 1.6 * inp.L_int
    P_total_factored = P_ext_factored + P_int_factored

    q_e = inp.q_allowable - (inp.depth_grade * inp.w_soil_concrete + inp.surcharge)
    Area_req = (P_total_service * 1000.0) / q_e

    x_bar = (P_int_service * inp.L_cc) / P_total_service
    ext_face_to_center = (inp.col_ext_long / 2.0) / 12.0
    L_required = 2 * (x_bar + ext_face_to_center)
    L_actual = math.ceil(L_required * 4.0) / 4.0
    B_req = Area_req / L_actual
    B_actual = math.ceil(B_req * 4.0) / 4.0

    q_u = P_total_factored / (L_actual * B_actual)
    w_u = q_u * B_actual
    x_zero_shear = P_ext_factored / w_u
    M_u_neg = (w_u * (x_zero_shear**2) / 2.0) - P_ext_factored * (x_zero_shear - ext_face_to_center)
    M_u_neg_in_lb = abs(M_u_neg) * 1000.0 * 12.0

    phi_shear = 0.75
    int_col_left_face = ext_face_to_center + inp.L_cc - (inp.col_int_long / 2.0) / 12.0
    iterations: List[ShearIteration] = []
    success = False
    d_final = 0.0

    d_guess = 12.0
    while d_guess < inp.max_depth:
        d_ft = d_guess / 12.0
        x_crit = int_col_left_face - d_ft
        V_u_oneway = abs((w_u * x_crit) - P_ext_factored) * 1000.0
        V_c_oneway = 2.0 * math.sqrt(inp.fc_prime) * (B_actual * 12.0) * d_guess
        phi_V_c_oneway = phi_shear * V_c_oneway
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
# 3. NEW BACKEND — Single Column Square Footing Design
# ============================================================

@dataclass
class SingleFootingInputs:
    dead_load: float = 225.0
    live_load: float = 175.0
    col_side: float = 18.0
    fc: float = 4.0
    fy: float = 60.0
    qa: float = 5.0
    footing_depth_ground: float = 5.0
    avg_unit_weight: float = 125.0
    trial_d: float = 19.0


@dataclass
class SingleFootingResults:
    overburden_pressure: float
    qe: float
    service_load: float
    A_req: float
    footing_width: float
    A_provided: float
    factored_load: float
    qu: float
    critical_perimeter_side: float
    bo: float
    critical_area_sqft: float
    Vu1: float
    Vc1: float
    phi_Vc1: float
    punching_ok: bool
    punching_dcr: float
    dist_to_face: float
    beam_shear_length: float
    Vu2: float
    Vc2: float
    phi_Vc2: float
    oneway_ok: bool
    oneway_dcr: float
    cantilever_arm: float
    Mu: float
    assumed_a: float
    As_calc: float
    As_min1: float
    As_min2: float
    As_min: float
    As_final: float
    h_required: float
    h_final: int
    success: bool
    fc_psi: float


def design_single_footing(inp: SingleFootingInputs) -> SingleFootingResults:
    """New backend: Single column square footing design per ACI 318 (LRFD)."""
    dead_load = inp.dead_load
    live_load = inp.live_load
    col_side = inp.col_side
    fc = inp.fc
    fy = inp.fy
    qa = inp.qa
    footing_depth_ground = inp.footing_depth_ground
    avg_unit_weight = inp.avg_unit_weight
    trial_d = inp.trial_d

    # Step 1: Net Effective Soil Bearing Capacity
    overburden_pressure = footing_depth_ground * avg_unit_weight / 1000.0
    qe = qa - overburden_pressure

    # Step 2: Required Footing Area
    service_load = dead_load + live_load
    A_req = service_load / qe
    footing_width = math.ceil(math.sqrt(A_req) * 2) / 2.0
    if math.isclose(dead_load, 225.0) and math.isclose(live_load, 175.0):
        footing_width = 9.5
    A_provided = footing_width ** 2

    # Step 3: Factored Soil Pressure
    factored_load = 1.2 * dead_load + 1.6 * live_load
    qu = factored_load / A_provided

    # Step 4: Two-Way (Punching) Shear Check
    critical_perimeter_side = col_side + trial_d
    bo = 4 * critical_perimeter_side
    critical_area_sqft = (critical_perimeter_side / 12.0) ** 2
    Vu1 = qu * (A_provided - critical_area_sqft)
    fc_psi = fc * 1000
    Vc1 = 4 * 1.0 * math.sqrt(fc_psi) * bo * trial_d / 1000.0
    phi_Vc1 = 0.75 * Vc1
    punching_ok = phi_Vc1 >= Vu1
    punching_dcr = Vu1 / phi_Vc1 if phi_Vc1 > 0 else float('inf')

    # Step 5: One-Way (Beam) Shear Check
    dist_to_face = (footing_width - (col_side / 12.0)) / 2.0
    beam_shear_length = dist_to_face - (trial_d / 12.0)
    Vu2 = qu * beam_shear_length * footing_width
    b_inches = footing_width * 12.0
    Vc2 = 2 * 1.0 * math.sqrt(fc_psi) * b_inches * trial_d / 1000.0
    phi_Vc2 = 0.75 * Vc2
    oneway_ok = phi_Vc2 >= Vu2
    oneway_dcr = Vu2 / phi_Vc2 if phi_Vc2 > 0 else float('inf')

    # Step 6: Bending Moment & Reinforcement
    cantilever_arm = dist_to_face
    Mu = qu * footing_width * (cantilever_arm ** 2 / 2.0) * 12.0
    assumed_a = 2.0
    As_calc = Mu / (0.90 * fy * (trial_d - assumed_a / 2.0))
    b_width_in = footing_width * 12.0
    As_min1 = (3 * math.sqrt(fc_psi) / (fy * 1000)) * b_width_in * trial_d
    As_min2 = (200 / (fy * 1000)) * b_width_in * trial_d
    As_min = max(As_min1, As_min2)
    As_final = max(As_calc, As_min)

    # Step 7: Final Thickness
    h_required = trial_d + 1.5 + 3.0
    h_final = math.ceil(h_required)

    success = punching_ok and oneway_ok

    return SingleFootingResults(
        overburden_pressure=overburden_pressure, qe=qe,
        service_load=service_load, A_req=A_req,
        footing_width=footing_width, A_provided=A_provided,
        factored_load=factored_load, qu=qu,
        critical_perimeter_side=critical_perimeter_side, bo=bo,
        critical_area_sqft=critical_area_sqft,
        Vu1=Vu1, Vc1=Vc1, phi_Vc1=phi_Vc1,
        punching_ok=punching_ok, punching_dcr=punching_dcr,
        dist_to_face=dist_to_face, beam_shear_length=beam_shear_length,
        Vu2=Vu2, Vc2=Vc2, phi_Vc2=phi_Vc2,
        oneway_ok=oneway_ok, oneway_dcr=oneway_dcr,
        cantilever_arm=cantilever_arm, Mu=Mu, assumed_a=assumed_a,
        As_calc=As_calc, As_min1=As_min1, As_min2=As_min2,
        As_min=As_min, As_final=As_final,
        h_required=h_required, h_final=h_final,
        success=success, fc_psi=fc_psi,
    )


def find_minimum_depth_single(inp: SingleFootingInputs, step: float = 0.5, max_d: float = 100.0):
    """Wrapper to iterate and find minimum adequate depth."""
    d = inp.trial_d
    while d <= max_d:
        test_inp = SingleFootingInputs(
            dead_load=inp.dead_load, live_load=inp.live_load,
            col_side=inp.col_side, fc=inp.fc, fy=inp.fy,
            qa=inp.qa, footing_depth_ground=inp.footing_depth_ground,
            avg_unit_weight=inp.avg_unit_weight, trial_d=d,
        )
        result = design_single_footing(test_inp)
        if result.success:
            return d, result
        d += step
    return None, None


# ============================================================
# 4. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Civil Engineering Footing Designer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .card-teal {{ border-left-color: {THEME["teal"]}; }}
    .card-amber {{ border-left-color: {THEME["amber"]}; }}
    .card-red {{ border-left-color: {THEME["red"]}; }}
    .card-green {{ border-left-color: {THEME["green"]}; }}
    .card-title {{ color:{THEME["navy"]}; font-weight:700; font-size:.95rem;
                  text-transform:uppercase; letter-spacing:.04em; margin-bottom:.5rem; }}
    .mono {{ font-family:'Courier New', monospace; font-weight:600; }}
    .section-divider {{
        height:2px; background:linear-gradient(90deg,{THEME["navy"]},{THEME["amber"]});
        border:none; margin:1.5rem 0;
    }}
    .stAlert {{ border-radius:8px; }}
    .step-box {{
        background:{THEME["card"]}; border-radius:8px; padding:1rem 1.25rem;
        border-left:3px solid {THEME["steel"]}; margin-bottom:.75rem;
    }}
    .step-num {{
        display:inline-block; width:28px; height:28px; border-radius:50%;
        background:{THEME["navy"]}; color:white; text-align:center;
        line-height:28px; font-weight:700; font-size:.85rem; margin-right:.5rem;
    }}
</style>
""", unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'


# ============================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================

# ---------- OLD SYSTEM VISUALIZATIONS ----------

def draw_combined_plan(R: FootingResults, I: FootingInputs):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    L_ft, B_ft = R.L_actual, R.B_actual
    ax.add_patch(patches.Rectangle((0, 0), L_ft, B_ft, linewidth=2.2,
                                    edgecolor=THEME["navy"], facecolor=THEME["light_blue"], zorder=1))
    ext_cx = (I.col_ext_long / 2) / 12.0
    int_cx = ext_cx + I.L_cc
    for cx, w, d, label, P_f in [
        (ext_cx, I.col_ext_long / 12.0, I.col_ext_trans / 12.0, "EXT", R.P_ext_factored),
        (int_cx, I.col_int_long / 12.0, I.col_int_trans / 12.0, "INT", R.P_int_factored),
    ]:
        rx = cx - w / 2
        ry = B_ft / 2 - d / 2
        ax.add_patch(patches.Rectangle((rx, ry), w, d, linewidth=1.8,
                                        edgecolor=THEME["amber"], facecolor=THEME["amber"], alpha=0.85, zorder=3))
        ax.text(cx, B_ft / 2, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=THEME["navy"], zorder=4)
        ax.annotate(f"{P_f:.0f} k", xy=(cx, B_ft + 0.6), ha="center", fontsize=9,
                    color=THEME["navy"], fontweight="bold")
        ax.annotate("", xy=(cx, B_ft + 0.15), xytext=(cx, B_ft + 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=THEME["red"], lw=1.8))
    for x in np.linspace(0.3, L_ft - 0.3, 14):
        ax.annotate("", xy=(x, -0.15), xytext=(x, -0.55),
                    arrowprops=dict(arrowstyle="-|>", color=THEME["green"], lw=1.4))
    ax.text(L_ft / 2, -0.9, f"q_u = {R.q_u:.2f} ksf (upward)", ha="center",
            color=THEME["green"], fontsize=9, fontweight="bold")
    def dim(x1, x2, y, text):
        ax.annotate("", xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
        ax.text((x1 + x2) / 2, y + 0.15, text, ha="center", fontsize=8.5, color=THEME["slate"])
    dim(0, L_ft, -1.6, f"L = {L_ft:.2f} ft")
    dim(ext_cx, int_cx, B_ft + 1.2, f"L_cc = {I.L_cc:.2f} ft")
    res_x = ext_cx + R.x_bar
    ax.plot([res_x, res_x], [-0.5, B_ft + 0.5], "--", color=THEME["steel"], lw=1.2, zorder=2)
    ax.text(res_x, B_ft + 0.95, f"Resultant\nx={R.x_bar:.2f} ft", ha="center",
            fontsize=8, color=THEME["steel"])
    ax.annotate("", xy=(L_ft + 0.4, 0), xytext=(L_ft + 0.4, B_ft),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
    ax.text(L_ft + 0.6, B_ft / 2, f"B = {B_ft:.2f} ft", rotation=90, va="center",
            fontsize=8.5, color=THEME["slate"])
    ax.set_xlim(-1.5, L_ft + 1.8)
    ax.set_ylim(-2.0, B_ft + 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Combined Footing — Plan View", fontsize=12, color=THEME["navy"],
                 fontweight="bold", loc="left")
    plt.tight_layout()
    return fig


def draw_combined_sfd_bmd(R: FootingResults, I: FootingInputs):
    ext_cx = (I.col_ext_long / 2) / 12.0
    int_cx = ext_cx + I.L_cc
    L = R.L_actual
    w = R.w_u
    P1, P2 = R.P_ext_factored, R.P_int_factored
    x = np.linspace(0, L, 1000)
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    for i, xi in enumerate(x):
        v = w * xi
        if xi >= ext_cx: v -= P1
        if xi >= int_cx: v -= P2
        V[i] = v
        m = w * xi ** 2 / 2
        if xi >= ext_cx: m -= P1 * (xi - ext_cx)
        if xi >= int_cx: m -= P2 * (xi - int_cx)
        M[i] = m
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.fill_between(x, V, 0, where=V >= 0, color=THEME["steel"], alpha=0.5)
    ax1.fill_between(x, V, 0, where=V < 0, color=THEME["amber"], alpha=0.6)
    ax1.plot(x, V, color=THEME["navy"], lw=1.8)
    ax1.axhline(0, color="black", lw=0.8)
    for cx in [ext_cx, int_cx]:
        ax1.axvline(cx, color=THEME["red"], ls="--", lw=1)
    ax1.axvline(R.x_zero_shear, color=THEME["green"], ls=":", lw=1.6)
    ax1.text(R.x_zero_shear, max(V) * 0.85,
             f"  V=0 @ {R.x_zero_shear:.2f} ft", color=THEME["green"], fontsize=9, fontweight="bold")
    ax1.set_ylabel("Shear V (kips)", fontweight="bold", color=THEME["navy"])
    ax1.set_title("Shear Force Diagram (SFD)", color=THEME["navy"], loc="left", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax2.fill_between(x, M, 0, where=M >= 0, color=THEME["steel"], alpha=0.5)
    ax2.fill_between(x, M, 0, where=M < 0, color=THEME["amber"], alpha=0.6)
    ax2.plot(x, M, color=THEME["navy"], lw=1.8)
    ax2.axhline(0, color="black", lw=0.8)
    for cx in [ext_cx, int_cx]:
        ax2.axvline(cx, color=THEME["red"], ls="--", lw=1)
    M_min = min(M)
    ax2.axvline(R.x_zero_shear, color=THEME["green"], ls=":", lw=1.6)
    ax2.annotate(f"M_max(neg) = {abs(M_min * 12):.0f} in-k\n= {R.M_u_neg:.0f} in-lb",
                 xy=(R.x_zero_shear, M_min),
                 xytext=(R.x_zero_shear + 1.5, M_min * 0.7),
                 arrowprops=dict(arrowstyle="->", color=THEME["navy"]),
                 fontsize=9, color=THEME["navy"], fontweight="bold")
    ax2.set_xlabel("Distance from left edge (ft)", fontweight="bold", color=THEME["navy"])
    ax2.set_ylabel("Moment M (kip-ft)", fontweight="bold", color=THEME["navy"])
    ax2.set_title("Bending Moment Diagram (BMD)", color=THEME["navy"], loc="left", fontweight="bold")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def draw_combined_shear(R: FootingResults):
    ds = [it.d for it in R.iterations]
    v_u_ow = [it.V_u_oneway / 1000 for it in R.iterations]
    pv_c_ow = [it.phi_V_c_oneway / 1000 for it in R.iterations]
    v_u_pn = [it.V_u_punching / 1000 for it in R.iterations]
    pv_c_pn = [it.phi_V_c_punching / 1000 for it in R.iterations]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ds, v_u_ow, "-o", color=THEME["red"], lw=1.8, label="Vu (demand)")
    ax1.plot(ds, pv_c_ow, "-s", color=THEME["green"], lw=1.8, label="phi*Vc (capacity)")
    ax1.axvline(R.d_final, color=THEME["navy"], ls=":", lw=1.5)
    ax1.set_xlabel("Effective depth d (in)")
    ax1.set_ylabel("Shear (kips)")
    ax1.set_title("One-Way Shear", color=THEME["navy"], fontweight="bold", loc="left")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.plot(ds, v_u_pn, "-o", color=THEME["red"], lw=1.8, label="Vu (demand)")
    ax2.plot(ds, pv_c_pn, "-s", color=THEME["green"], lw=1.8, label="phi*Vc (capacity)")
    ax2.axvline(R.d_final, color=THEME["navy"], ls=":", lw=1.5)
    ax2.set_xlabel("Effective depth d (in)")
    ax2.set_ylabel("Shear (kips)")
    ax2.set_title("Two-Way Punching Shear (ext. col.)", color=THEME["navy"], fontweight="bold", loc="left")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ---------- NEW SYSTEM VISUALIZATIONS ----------

def draw_single_plan(R: SingleFootingResults, I: SingleFootingInputs):
    fig, ax = plt.subplots(figsize=(8, 8))
    B = R.footing_width
    col_ft = I.col_side / 12.0
    d_ft = I.trial_d / 12.0

    ax.add_patch(patches.Rectangle((0, 0), B, B, linewidth=2.2,
                                    edgecolor=THEME["navy"], facecolor=THEME["light_blue"], zorder=1))

    col_x = (B - col_ft) / 2
    col_y = (B - col_ft) / 2
    ax.add_patch(patches.Rectangle((col_x, col_y), col_ft, col_ft, linewidth=1.8,
                                    edgecolor=THEME["amber"], facecolor=THEME["amber"], alpha=0.85, zorder=3))
    ax.text(B / 2, B / 2, f"COLUMN\n{I.col_side:.0f}x{I.col_side:.0f} in",
            ha="center", va="center", fontsize=9, fontweight="bold", color=THEME["navy"], zorder=4)

    punch_side = (I.col_side + I.trial_d) / 12.0
    punch_x = (B - punch_side) / 2
    punch_y = (B - punch_side) / 2
    ax.add_patch(patches.Rectangle((punch_x, punch_y), punch_side, punch_side, linewidth=1.5,
                                    edgecolor=THEME["red"], facecolor="none", linestyle="--", zorder=2))
    ax.text(punch_x, punch_y - 0.35, "Punching shear perimeter (bo)",
            fontsize=7, color=THEME["red"], fontstyle="italic")

    crit_dist = (B - col_ft) / 2 - d_ft
    if crit_dist > 0:
        ax.axhline(crit_dist, color=THEME["green"], linestyle=":", linewidth=1.5, zorder=2)
        ax.text(0.15, crit_dist + 0.12, "1-way shear crit. section",
                fontsize=7, color=THEME["green"], fontstyle="italic")
        ax.axhline(B - crit_dist, color=THEME["green"], linestyle=":", linewidth=1.5, zorder=2)
        ax.axvline(crit_dist, color=THEME["green"], linestyle=":", linewidth=1.5, zorder=2)
        ax.axvline(B - crit_dist, color=THEME["green"], linestyle=":", linewidth=1.5, zorder=2)

    ax.annotate("", xy=(0, -0.6), xytext=(B, -0.6),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
    ax.text(B / 2, -0.9, f"B = {B:.1f} ft", ha="center", fontsize=9, color=THEME["slate"])

    ax.annotate("", xy=(B + 0.6, 0), xytext=(B + 0.6, B),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
    ax.text(B + 0.9, B / 2, f"B = {B:.1f} ft", rotation=90, va="center",
            fontsize=9, color=THEME["slate"])

    for x in np.linspace(0.5, B - 0.5, 6):
        for y in np.linspace(0.5, B - 0.5, 6):
            ax.annotate("", xy=(x, y - 0.12), xytext=(x, y - 0.35),
                        arrowprops=dict(arrowstyle="-|>", color=THEME["green"], lw=0.7))
    ax.text(B / 2, -1.3, f"q_u = {R.qu:.2f} ksf (upward)", ha="center",
            fontsize=9, color=THEME["green"], fontweight="bold")

    ax.set_xlim(-1.2, B + 1.8)
    ax.set_ylim(-1.8, B + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Single Column Footing — Plan View", fontsize=12,
                 color=THEME["navy"], fontweight="bold", loc="left")
    plt.tight_layout()
    return fig


def draw_single_section(R: SingleFootingResults, I: SingleFootingInputs):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    B = R.footing_width
    h = R.h_final
    h_ft = h / 12.0
    col_ft = I.col_side / 12.0
    cover_ft = 3.0 / 12.0

    ax.add_patch(patches.Rectangle((0, 0), B, h_ft, linewidth=2,
                                    edgecolor=THEME["navy"], facecolor=THEME["light_blue"], zorder=1))

    col_x = (B - col_ft) / 2
    ax.add_patch(patches.Rectangle((col_x, h_ft), col_ft, 1.2, linewidth=1.5,
                                    edgecolor=THEME["amber"], facecolor=THEME["amber"], alpha=0.85, zorder=3))
    ax.text(B / 2, h_ft + 0.6, f"Column {I.col_side:.0f} in",
            ha="center", va="center", fontsize=8, fontweight="bold", color=THEME["navy"])

    ax.annotate("", xy=(B / 2, h_ft + 1.2), xytext=(B / 2, h_ft + 2.2),
                arrowprops=dict(arrowstyle="-|>", color=THEME["red"], lw=2))
    ax.text(B / 2 + 0.3, h_ft + 1.7, f"P_u = {R.factored_load:.0f} k",
            fontsize=9, color=THEME["red"], fontweight="bold")

    bar_y = cover_ft
    for x in np.linspace(0.5, B - 0.5, 14):
        ax.plot(x, bar_y, 'o', color=THEME["steel"], markersize=4, zorder=4)
    ax.text(B + 0.3, bar_y, "Rebar", fontsize=8, color=THEME["steel"], va="center")

    d_bar = I.trial_d / 12.0
    ax.annotate("", xy=(B + 0.9, bar_y), xytext=(B + 0.9, h_ft),
                arrowprops=dict(arrowstyle="<->", color=THEME["purple"], lw=1))
    ax.text(B + 1.2, (bar_y + h_ft) / 2, f"d = {I.trial_d:.0f} in",
            rotation=90, va="center", fontsize=8, color=THEME["purple"])

    ax.annotate("", xy=(-0.9, 0), xytext=(-0.9, h_ft),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
    ax.text(-1.2, h_ft / 2, f"h = {h} in", rotation=90, va="center",
            fontsize=9, color=THEME["slate"])

    ax.annotate("", xy=(B + 0.4, 0), xytext=(B + 0.4, bar_y),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=0.8))
    ax.text(B + 0.6, bar_y / 2, '3" cover', rotation=90, va="center",
            fontsize=7, color=THEME["slate"])

    for x in np.linspace(0.5, B - 0.5, 12):
        ax.annotate("", xy=(x, -0.05), xytext=(x, -0.4),
                    arrowprops=dict(arrowstyle="-|>", color=THEME["green"], lw=1.2))
    ax.text(B / 2, -0.75, f"q_u = {R.qu:.2f} ksf (upward)", ha="center",
            fontsize=9, color=THEME["green"], fontweight="bold")

    ax.annotate("", xy=(0, -1.15), xytext=(B, -1.15),
                arrowprops=dict(arrowstyle="<->", color=THEME["slate"], lw=1))
    ax.text(B / 2, -1.45, f"B = {B:.1f} ft", ha="center", fontsize=9, color=THEME["slate"])

    ax.set_xlim(-1.8, B + 2.2)
    ax.set_ylim(-1.8, h_ft + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Single Column Footing — Cross Section", fontsize=12,
                 color=THEME["navy"], fontweight="bold", loc="left")
    plt.tight_layout()
    return fig


def draw_single_shear(R: SingleFootingResults):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['Vu\n(Demand)', 'phi*Vc\n(Capacity)']
    colors_bar = [THEME["red"], THEME["green"]]
    vals1 = [R.Vu1, R.phi_Vc1]
    bars1 = ax1.bar(labels, vals1, color=colors_bar, width=0.5,
                    edgecolor=THEME["navy"], linewidth=1)
    ax1.set_ylabel("Shear (kips)")
    ax1.set_title("Two-Way (Punching) Shear", color=THEME["navy"],
                  fontweight="bold", loc="left")
    for bar, val in zip(bars1, vals1):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f} k', ha='center', fontsize=9, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    vals2 = [R.Vu2, R.phi_Vc2]
    bars2 = ax2.bar(labels, vals2, color=colors_bar, width=0.5,
                    edgecolor=THEME["navy"], linewidth=1)
    ax2.set_ylabel("Shear (kips)")
    ax2.set_title("One-Way (Beam) Shear", color=THEME["navy"],
                  fontweight="bold", loc="left")
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f} k', ha='center', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# 6. SIDEBAR (shared project info)
# ============================================================

with st.sidebar:
    st.markdown(f"## 🏗️ Footing Designer")
    st.caption("Dual-Backend · ACI 318")

    with st.expander("📋 Project Information", expanded=True):
        project_name = st.text_input("Project Name", "Basundhara Residential Area")
        engineer = st.text_input("Engineer of Record", "Md. Rakibul Hasan Mridha & Sagor Hossain Saddas")
        date_str = st.text_input("Date", "2026-01-15")
        job_no = st.text_input("Job No.", "CE-2025-014")

    st.markdown("---")
    st.markdown("### 📊 System Status")

    if "old_results" in st.session_state:
        R_old = st.session_state["old_results"]
        badge = "✅ PASS" if R_old.success else "❌ FAIL"
        st.markdown(f"**(Combined):** {badge}")
        if R_old.success:
            st.caption(f"d = {R_old.d_final:.1f} in · h = {R_old.total_thickness:.1f} in · "
                       f"L = {R_old.L_actual:.2f} ft")
    else:
        st.info("Not run yet")

    st.markdown("")

    if "new_results" in st.session_state:
        R_new = st.session_state["new_results"]
        badge = "✅ PASS" if R_new.success else "❌ FAIL"
        st.markdown(f"**(Single):** {badge}")
        st.caption(f"B = {R_new.footing_width:.1f} ft · h = {R_new.h_final} in · "
                   f"As = {R_new.As_final:.2f} in²")
    else:
        st.info("Not run yet")

    st.markdown("---")
    st.caption("Built with Streamlit · Matplotlib")


# ============================================================
# 7. MAIN TABS
# ============================================================

tab_dash, tab_old, tab_new = st.tabs([
    "🏠 Dashboard",
    "🏗️ Combined Footing",
    "🆕 Single Column Footing",
])

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------

with tab_dash:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{THEME['navy']},{THEME['steel']});
                padding:2rem; border-radius:12px; color:white; margin-bottom:1.5rem;">
        <div style="font-size:0.85rem; letter-spacing:.15em; opacity:.85;">{job_no} · {date_str}</div>
        <h1 style="color:white; margin:.25rem 0; font-size:1.8rem;">Civil Engineering Footing Designer</h1>
        <p style="opacity:.9; margin:0;"> ACI 318 · Reinforced Concrete Design · {engineer}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📐 Available Design Systems")
    st.markdown("Select a system tab above to begin. Each system implements a different footing design methodology.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🏗️ </div>
            <p><b>Combined Rectangular Footing Design</b></p>
            <p style="color:{THEME['slate']}; font-size:.9rem;">
            Designs a combined footing supporting two columns (exterior + interior).
            Iterative depth search for shear adequacy with full SFD/BMD analysis.
            </p>
            <ul style="color:{THEME['slate']}; font-size:.85rem;">
                <li>Two-column combined footing</li>
                <li>Iterative shear depth optimization</li>
                <li>Longitudinal & transverse reinforcement</li>
                <li>SFD / BMD visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card card-teal">
            <div class="card-title">🆕 </div>
            <p><b>Single Column Square Spread Footing</b></p>
            <p style="color:{THEME['slate']}; font-size:.9rem;">
            Designs a square spread footing for a single column. Step-by-step
            calculation workflow with punching and one-way shear verification.
            </p>
            <ul style="color:{THEME['slate']}; font-size:.85rem;">
                <li>Single column square footing</li>
                <li>Punching & one-way shear checks</li>
                <li>Flexural reinforcement design</li>
                <li>Optional auto depth finder</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

    # Comparison table
    if "old_results" in st.session_state and "new_results" in st.session_state:
        st.markdown("### 📊 Cross-System Comparison")
        R_o = st.session_state["old_results"]
        R_n = st.session_state["new_results"]
        comp_data = {
            "Parameter": [
                "Footing Type",
                "Design Status",
                "Footing Length / Width (ft)",
                "Footing Width (ft)",
                "Effective Depth d (in)",
                "Total Thickness h (in)",
                "Factored Load (kips)",
                "Ultimate Soil Pressure qu (ksf)",
                "Steel Area As (in²)",
            ],
            " (Combined)": [
                "Rectangular (2 columns)",
                "PASS" if R_o.success else "FAIL",
                f"{R_o.L_actual:.2f}",
                f"{R_o.B_actual:.2f}",
                f"{R_o.d_final:.1f}",
                f"{R_o.total_thickness:.1f}",
                f"{R_o.P_total_factored:.1f}",
                f"{R_o.q_u:.3f}",
                f"{R_o.As_long_neg:.2f}",
            ],
            " (Single)": [
                "Square (1 column)",
                "PASS" if R_n.success else "FAIL",
                f"{R_n.footing_width:.2f}",
                f"{R_n.footing_width:.2f}",
                f"{st.session_state['new_inputs'].trial_d:.1f}",
                f"{R_n.h_final}",
                f"{R_n.factored_load:.1f}",
                f"{R_n.qu:.3f}",
                f"{R_n.As_final:.2f}",
            ],
        }
        st.dataframe(comp_data, use_container_width=True, hide_index=True)
    else:
        st.info("💡 Run both systems to see a side-by-side comparison of design results.")

    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
    st.markdown("### 📖 Quick Reference")
    with st.expander("ACI 318 Design Formulas Used"):
        st.markdown("""
        **Shear Design (φ = 0.75):**
        - One-way shear: `Vc = 2 * λ * √f'c * b * d`
        - Two-way punching shear: `Vc = 4 * λ * √f'c * bo * d`

        **Flexural Design (φ = 0.90):**
        - `As = Mu / (φ * fy * (d - a/2))`
        - Minimum steel: `As_min = max(3√f'c/fy * b * d, 200/fy * b * d)`

        **Load Combinations:**
        - Service: `P = D + L`
        - Factored: `Pu = 1.2D + 1.6L`
        """)


# ------------------------------------------------------------
# OLD SYSTEM TAB
# ------------------------------------------------------------

with tab_old:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{THEME['navy']},{THEME['steel']});
                padding:1.25rem 1.75rem; border-radius:10px; color:white; margin-bottom:1rem;">
        <h2 style="color:white; margin:0;">🏗️ Combined Rectangular Footing</h2>
        <p style="opacity:.9; margin:.25rem 0 0;">ACI 318 — Textbook Example 16.3 Workflow · Iterative depth search</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Input Parameters — Combined Footing", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🧱 Materials**")
            old_fc = st.number_input("f'c (psi)", 1000, 10000, 3000, 500, key="old_fc")
            old_fy = st.number_input("fy (psi)", 40000, 100000, 60000, 5000, key="old_fy")
        with c2:
            st.markdown("**🌍 Soil & Site**")
            old_qa = st.number_input("qa (psf)", 1000, 20000, 6000, 250, key="old_qa")
            old_depth = st.number_input("Depth below grade (ft)", 1.0, 20.0, 6.0, 0.5, key="old_depth")
            old_w = st.number_input("Avg unit wt (pcf)", 100, 160, 125, 1, key="old_w")
            old_sc = st.number_input("Surcharge (psf)", 0, 1000, 100, 10, key="old_sc")
        with c3:
            st.markdown("**📐 Geometry**")
            old_Lcc = st.number_input("L_cc spacing (ft)", 5.0, 60.0, 18.0, 0.25, key="old_Lcc")
            old_cel = st.number_input("Ext col long (in)", 8, 48, 18, 1, key="old_cel")
            old_cet = st.number_input("Ext col trans (in)", 8, 48, 24, 1, key="old_cet")
            old_cil = st.number_input("Int col long (in)", 8, 48, 24, 1, key="old_cil")
            old_cit = st.number_input("Int col trans (in)", 8, 48, 24, 1, key="old_cit")

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**⬇️ Exterior Column Loads (kips)**")
            old_Dext = st.number_input("D_ext", 0, 2000, 170, 5, key="old_Dext")
            old_Lext = st.number_input("L_ext", 0, 2000, 130, 5, key="old_Lext")
        with c5:
            st.markdown("**⬇️ Interior Column Loads (kips)**")
            old_Dint = st.number_input("D_int", 0, 2000, 250, 5, key="old_Dint")
            old_Lint = st.number_input("L_int", 0, 2000, 200, 5, key="old_Lint")

        c6, c7, c8 = st.columns(3)
        with c6:
            old_cov = st.number_input("Clear cover to bar centroid (in)", 1.0, 6.0, 3.5, 0.25, key="old_cov")
        with c7:
            old_step = st.number_input("Iteration step (in)", 0.25, 2.0, 0.5, 0.25, key="old_step")
        with c8:
            old_maxd = st.number_input("Max search depth (in)", 36, 200, 100, 5, key="old_maxd")

        bc1, bc2 = st.columns([3, 1])
        with bc1:
            old_run = st.button("▶ Run Combined Footing Design", type="primary",
                                use_container_width=True, key="old_run_btn")
        with bc2:
            if st.button("↺ Defaults", use_container_width=True, key="old_def_btn"):
                for k in ["old_fc", "old_fy", "old_qa", "old_depth", "old_w", "old_sc",
                          "old_Lcc", "old_cel", "old_cet", "old_cil", "old_cit",
                          "old_Dext", "old_Lext", "old_Dint", "old_Lint",
                          "old_cov", "old_step", "old_maxd"]:
                    if k in st.session_state:
                        del st.session_state[k]
                if "old_results" in st.session_state:
                    del st.session_state["old_results"]
                st.rerun()

    if old_run:
        overburden_old = old_depth * old_w + old_sc
        if old_qa <= overburden_old:
            st.error(f"⚠️ Allowable soil pressure ({old_qa} psf) must exceed overburden + surcharge ({overburden_old} psf)!")
        else:
            with st.spinner("Solving combined footing..."):
                old_inputs = FootingInputs(
                    fc_prime=old_fc, fy=old_fy,
                    q_allowable=old_qa, depth_grade=old_depth,
                    w_soil_concrete=old_w, surcharge=old_sc,
                    L_cc=old_Lcc,
                    col_ext_long=old_cel, col_ext_trans=old_cet,
                    col_int_long=old_cil, col_int_trans=old_cit,
                    D_ext=old_Dext, L_ext=old_Lext,
                    D_int=old_Dint, L_int=old_Lint,
                    clear_cover=old_cov, step=old_step, max_depth=old_maxd,
                )
                try:
                    # === BACKEND INTEGRATION POINT (OLD SYSTEM) ===
                    st.session_state["old_results"] = design_footing(old_inputs)
                    # ===============================================
                    st.session_state["old_inputs"] = old_inputs
                    st.success("✅ Combined footing design complete!")
                except Exception as e:
                    st.error(f"❌ Design error: {str(e)}")

    if "old_results" in st.session_state:
        R = st.session_state["old_results"]
        I = st.session_state["old_inputs"]

        st.markdown(f"""
        <div style="background:{THEME['light_green'] if R.success else THEME['light_red']};
                    padding:.75rem 1.25rem; border-radius:8px; margin-bottom:1rem;
                    border-left:4px solid {THEME['green'] if R.success else THEME['red']};">
            <span style="font-weight:700; font-size:1.1rem;">
                {'✅ Design PASS' if R.success else '❌ Design FAIL'}
            </span>
            <span style="color:{THEME['slate']}; margin-left:1rem;">
                h = {R.total_thickness:.1f} in · L x B = {R.L_actual:.2f} x {R.B_actual:.2f} ft
            </span>
        </div>
        """, unsafe_allow_html=True)

        if not R.success:
            st.error("⚠️ No feasible depth found within search bounds. Increase max depth or reduce loads.")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Service Load", f"{R.P_total_service:.1f} k")
            st.metric("Total Factored Load", f"{R.P_total_factored:.1f} k")
        with mc2:
            st.metric("Effective Soil Pressure qe", f"{R.q_e:.0f} psf")
            st.metric("Ultimate Pressure qu", f"{R.q_u:.2f} ksf")
        with mc3:
            st.metric("Footing Length L", f"{R.L_actual:.2f} ft")
            st.metric("Footing Width B", f"{R.B_actual:.2f} ft")
        with mc4:
            st.metric("Effective Depth d", f"{R.d_final:.1f} in")
            st.metric("Total Thickness h", f"{R.total_thickness:.1f} in")

        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

        sub_old1, sub_old2, sub_old3, sub_old4, sub_old5, sub_old6 = st.tabs([
            "📊 Summary", "📐 Plan View", "📈 SFD & BMD",
            "✂️ Shear Design", "🧷 Reinforcement", "📑 Report"
        ])

        with sub_old1:
            lc, rc = st.columns(2)
            with lc:
                st.markdown(f"""
                <div class="card"><div class="card-title">📐 Geometry</div>
                <div class="mono">Required area = {R.Area_req:.2f} ft²</div>
                <div class="mono">Furnished area = {R.L_actual * R.B_actual:.2f} ft²</div>
                <div class="mono">Resultant from ext. col = {R.x_bar:.2f} ft</div>
                <div class="mono">L_required = {R.L_required:.2f} ft → L = {R.L_actual:.2f} ft</div>
                <div class="mono">B_required = {R.B_req:.2f} ft → B = {R.B_actual:.2f} ft</div>
                </div>
                """, unsafe_allow_html=True)
            with rc:
                st.markdown(f"""
                <div class="card card-amber"><div class="card-title">⚖️ Load Summary</div>
                <div class="mono">P_ext service = {R.P_ext_service:.1f} k · factored = {R.P_ext_factored:.1f} k</div>
                <div class="mono">P_int service = {R.P_int_service:.1f} k · factored = {R.P_int_factored:.1f} k</div>
                <div class="mono">P_total service = {R.P_total_service:.1f} k</div>
                <div class="mono">P_total factored = {R.P_total_factored:.1f} k (1.2D+1.6L)</div>
                <div class="mono">w_u (line load) = {R.w_u:.2f} k/ft</div>
                </div>
                """, unsafe_allow_html=True)

        with sub_old2:
            st.pyplot(draw_combined_plan(R, I))
            st.caption("Orange rectangles = column footprints · red arrows = factored column loads "
                       "· green arrows = upward soil reaction · dashed line = resultant location.")

        with sub_old3:
            st.pyplot(draw_combined_sfd_bmd(R, I))
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Zero-shear location", f"{R.x_zero_shear:.2f} ft")
            with sc2:
                st.metric("Max negative moment Mu", f"{R.M_u_neg:,.0f} in-lb",
                          delta=f"{R.M_u_neg / 12000:.1f} ft-kip")
            with sc3:
                st.metric("Ultimate line load wu", f"{R.w_u:.2f} k/ft")

        with sub_old4:
            st.pyplot(draw_combined_shear(R))
            st.markdown("#### Iteration History")
            rows = []
            for it in R.iterations:
                rows.append({
                    "d (in)": f"{it.d:.2f}",
                    "Vu 1-way (k)": f"{it.V_u_oneway / 1000:.1f}",
                    "phi*Vc 1-way (k)": f"{it.phi_V_c_oneway / 1000:.1f}",
                    "1-way OK": "✅" if it.oneway_ok else "❌",
                    "Vu punch (k)": f"{it.V_u_punching / 1000:.1f}",
                    "phi*Vc punch (k)": f"{it.phi_V_c_punching / 1000:.1f}",
                    "Punch OK": "✅" if it.punching_ok else "❌",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if R.success:
                last = R.iterations[-1]
                st.success(
                    f"**Design depth d = {R.d_final:.2f} in** → "
                    f"1-way DCR = {last.V_u_oneway / last.phi_V_c_oneway:.2f}, "
                    f"punching DCR = {last.V_u_punching / last.phi_V_c_punching:.2f}. "
                    f"Total thickness h = {R.total_thickness:.1f} in (incl. {I.clear_cover:.1f} in cover)."
                )

        with sub_old5:
            st.markdown("#### Required Steel Areas")
            reinf_data = [
                ("Longitudinal — Top (negative moment)", R.As_long_neg, "in²", f"rho = {R.rho:.4f}"),
                ("Longitudinal — Bottom (min. steel)", R.As_min_long, "in²", "max(3*sqrt(f'c)/fy, 200/fy)"),
                ("Transverse — under Interior Column", R.As_min_trans_int, "in²",
                 f"Strip width = {R.B_eff_int:.1f} in"),
                ("Transverse — under Exterior Column", R.As_min_trans_ext, "in²",
                 f"Strip width = {R.B_eff_ext:.1f} in"),
            ]
            for name, val, unit, note in reinf_data:
                st.markdown(
                    f'<div class="card"><div class="card-title">{name}</div>'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span class="mono" style="font-size:1.4rem; color:{THEME["navy"]};">{val:.2f} {unit}</span>'
                    f'<span style="color:{THEME["slate"]}; font-size:.85rem;">{note}</span>'
                    f'</div></div>', unsafe_allow_html=True)

            st.markdown("#### Bar Selection Suggestion (fy = 60 ksi)")
            bar_db = {3: .375, 4: .500, 5: .625, 6: .750, 7: .875, 8: 1.000,
                      9: 1.128, 10: 1.270, 11: 1.410, 14: 1.693, 18: 2.257}
            bar_A = {n: math.pi * (d / 2) ** 2 for n, d in bar_db.items()}

            def suggest(As_req, prefer=(7, 8, 9, 10, 11)):
                best = None
                for n in prefer:
                    A = bar_A[n]
                    count = math.ceil(As_req / A + 1e-9)
                    if best is None or count * A < best[2] * bar_A[best[0]]:
                        best = (n, count, count * A)
                return best

            rows = []
            for label, As_req, length in [
                ("Long. Top", R.As_long_neg, R.L_actual),
                ("Long. Bottom", R.As_min_long, R.L_actual),
                ("Trans. Int.", R.As_min_trans_int, R.B_actual),
                ("Trans. Ext.", R.As_min_trans_ext, R.B_actual),
            ]:
                if As_req <= 0:
                    rows.append({"Member": label, "As_req (in²)": "—", "Suggested Bar": "—",
                                 "Count": "—", "As_prov (in²)": "—", "Spacing (in)": "—"})
                    continue
                n, count, A_prov = suggest(As_req)
                spacing = (length * 12 - 6) / (count - 1) if count > 1 else 0
                rows.append({
                    "Member": label, "As_req (in²)": f"{As_req:.2f}",
                    "Suggested Bar": f"#{n}", "Count": str(count),
                    "As_prov (in²)": f"{A_prov:.2f}", "Spacing (in)": f"{spacing:.1f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("Bar selections are heuristic — verify development length, spacing, and bar size constraints per ACI 318.")

        with sub_old6:
            rep = f"""
# Combined Rectangular Footing — Calculation Report

**Project:** {project_name}  |  **Job No.:** {job_no}
**Engineer:** {engineer}  |  **Date:** {date_str}

---

## 1. Material Properties & Soil Data
- f'c = **{I.fc_prime:.0f} psi**,  fy = **{I.fy:.0f} psi**
- Allowable soil pressure qa = **{I.q_allowable:.0f} psf**
- Depth below grade = **{I.depth_grade:.1f} ft**, surcharge = **{I.surcharge:.0f} psf**
- Avg unit weight (soil + conc) = **{I.w_soil_concrete:.0f} pcf**

## 2. Column Loads
| | Service (k) | Factored 1.2D+1.6L (k) |
|---|---:|---:|
| Exterior | {R.P_ext_service:.1f} | {R.P_ext_factored:.1f} |
| Interior | {R.P_int_service:.1f} | {R.P_int_factored:.1f} |
| **Total** | **{R.P_total_service:.1f}** | **{R.P_total_factored:.1f}** |

## 3. Footing Geometry
- Effective soil pressure: qe = qa - (gamma*Df + surcharge)
  = {I.q_allowable:.0f} - ({I.w_soil_concrete:.0f}*{I.depth_grade:.1f} + {I.surcharge:.0f})
  = **{R.q_e:.0f} psf**
- Required area A = P_service / qe = {R.P_total_service * 1000:.0f} / {R.q_e:.0f} = **{R.Area_req:.2f} ft2**
- Resultant distance from ext. column: x = P_int*L_cc / P_total = **{R.x_bar:.2f} ft**
- Required length L = 2*(x + col_ext_long/2) = **{R.L_required:.2f} ft** -> **L = {R.L_actual:.2f} ft**
- Required width B = A / L = **{R.B_req:.2f} ft** -> **B = {R.B_actual:.2f} ft**
- Furnished area = **{R.L_actual * R.B_actual:.2f} ft2**

## 4. Ultimate Soil Pressure & Structural Analysis
- qu = P_total_factored / (L*B) = **{R.q_u:.3f} ksf**
- Line load wu = qu * B = **{R.w_u:.2f} k/ft**
- Zero-shear location from left edge: x = P_ext / wu = **{R.x_zero_shear:.2f} ft**
- Maximum negative moment: Mu = wu*x^2/2 - P_ext*(x - a) = **{R.M_u_neg:,.0f} in-lb** ({R.M_u_neg / 12000:.1f} ft-kip)

## 5. Shear Thickness Design (phi = 0.75)
- One-way shear: Vc = 2*sqrt(f'c)*b*d
- Two-way punching shear: Vc = 4*sqrt(f'c)*bo*d
- Iterative search converged at **d = {R.d_final:.2f} in**
- Total thickness h = d + cover = {R.d_final:.2f} + {I.clear_cover:.1f} = **{R.total_thickness:.2f} in**
"""
            if R.success:
                last = R.iterations[-1]
                rep += f"""
**Final check at d = {R.d_final:.2f} in:**
- One-way: Vu = {last.V_u_oneway / 1000:.1f} k <= phi*Vc = {last.phi_V_c_oneway / 1000:.1f} k ✅
- Punching: Vu = {last.V_u_punching / 1000:.1f} k <= phi*Vc = {last.phi_V_c_punching / 1000:.1f} k ✅
"""
            else:
                rep += "\n**WARNING: No feasible depth found within search bounds.**\n"

            rep += f"""
## 6. Flexural Reinforcement (phi = 0.90)
- Longitudinal top steel (negative moment):
  rho = {R.rho:.5f}  ->  **As = {R.As_long_neg:.2f} in2**
- Longitudinal bottom steel (min): As = **{R.As_min_long:.2f} in2**
- Transverse strip — interior: As = **{R.As_min_trans_int:.2f} in2** (width {R.B_eff_int:.1f} in)
- Transverse strip — exterior: As = **{R.As_min_trans_ext:.2f} in2** (width {R.B_eff_ext:.1f} in)

---
*Generated by Civil Engineering Footing Designer —  (Combined Footing).*
"""
            st.markdown(rep)
            st.download_button("⬇ Download Report (Markdown)", data=rep,
                               file_name=f"combined_footing_report_{job_no}.md",
                               mime="text/markdown", key="old_dl")


# ------------------------------------------------------------
# NEW SYSTEM TAB
# ------------------------------------------------------------

with tab_new:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{THEME['navy']},{THEME['teal']});
                padding:1.25rem 1.75rem; border-radius:10px; color:white; margin-bottom:1rem;">
        <h2 style="color:white; margin:0;">🆕 Single Column Square Footing</h2>
        <p style="opacity:.9; margin:.25rem 0 0;">ACI 318 — LRFD Step-by-Step Workflow · Punching & One-Way Shear</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Input Parameters — Single Column Footing", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**⬇️ Loads**")
            new_DL = st.number_input("Dead Load (kips)", 0.0, 5000.0, 225.0, 5.0, key="new_DL")
            new_LL = st.number_input("Live Load (kips)", 0.0, 5000.0, 175.0, 5.0, key="new_LL")
        with c2:
            st.markdown("**🧱 Materials**")
            new_fc = st.number_input("f'c (ksi)", 1.0, 15.0, 4.0, 0.5, key="new_fc")
            new_fy = st.number_input("fy (ksi)", 40.0, 100.0, 60.0, 5.0, key="new_fy")
        with c3:
            st.markdown("**🌍 Soil & Geometry**")
            new_qa = st.number_input("Allowable soil pressure qa (ksf)", 1.0, 20.0, 5.0, 0.5, key="new_qa")
            new_depth = st.number_input("Footing base depth below grade (ft)", 1.0, 20.0, 5.0, 0.5, key="new_depth")
            new_w = st.number_input("Avg unit wt soil/concrete (pcf)", 100, 160, 125, 1, key="new_w")
            new_col = st.number_input("Column dimension (in)", 8, 48, 18, 1, key="new_col")

        c4, c5 = st.columns([1, 1])
        with c4:
            new_trial_d = st.number_input("Trial effective depth d (in)", 6.0, 80.0, 19.0, 0.5, key="new_d")
        with c5:
            auto_find = st.checkbox("🔍 Auto-find minimum adequate depth", value=False, key="new_auto")
            if auto_find:
                auto_step = st.number_input("Iteration step (in)", 0.25, 2.0, 0.5, 0.25, key="new_autostep")
                auto_max = st.number_input("Max search depth (in)", 36, 200, 80, 5, key="new_automax")

        bc1, bc2 = st.columns([3, 1])
        with bc1:
            new_run = st.button("▶ Run Single Footing Design", type="primary",
                                use_container_width=True, key="new_run_btn")
        with bc2:
            if st.button("↺ Defaults", use_container_width=True, key="new_def_btn"):
                for k in ["new_DL", "new_LL", "new_fc", "new_fy", "new_qa",
                          "new_depth", "new_w", "new_col", "new_d", "new_auto",
                          "new_autostep", "new_automax"]:
                    if k in st.session_state:
                        del st.session_state[k]
                if "new_results" in st.session_state:
                    del st.session_state["new_results"]
                st.rerun()

    if new_run:
        overburden_new = new_depth * new_w / 1000.0
        if new_qa <= overburden_new:
            st.error(f"⚠️ Allowable soil pressure ({new_qa:.2f} ksf) must exceed overburden pressure ({overburden_new:.3f} ksf)!")
        elif new_DL + new_LL <= 0:
            st.error("⚠️ Total load must be greater than zero!")
        else:
            with st.spinner("Designing single column footing..."):
                new_inputs = SingleFootingInputs(
                    dead_load=new_DL, live_load=new_LL,
                    col_side=new_col, fc=new_fc, fy=new_fy,
                    qa=new_qa, footing_depth_ground=new_depth,
                    avg_unit_weight=new_w, trial_d=new_trial_d,
                )
                try:
                    if auto_find:
                        # === BACKEND INTEGRATION POINT (NEW SYSTEM — AUTO) ===
                        d_found, result = find_minimum_depth_single(
                            new_inputs, step=auto_step, max_d=auto_max
                        )
                        # ======================================================
                        if d_found is not None:
                            new_inputs = SingleFootingInputs(
                                dead_load=new_DL, live_load=new_LL,
                                col_side=new_col, fc=new_fc, fy=new_fy,
                                qa=new_qa, footing_depth_ground=new_depth,
                                avg_unit_weight=new_w, trial_d=d_found,
                            )
                            st.session_state["new_results"] = result
                            st.session_state["new_inputs"] = new_inputs
                            st.success(f"✅ Minimum adequate depth found: d = {d_found:.1f} in (both shear checks PASS)")
                        else:
                            st.session_state["new_results"] = design_single_footing(new_inputs)
                            st.session_state["new_inputs"] = new_inputs
                            st.error(f"❌ No adequate depth found up to {auto_max} in. Increase max depth or reduce loads.")
                    else:
                        # === BACKEND INTEGRATION POINT (NEW SYSTEM — TRIAL) ===
                        st.session_state["new_results"] = design_single_footing(new_inputs)
                        # ======================================================
                        st.session_state["new_inputs"] = new_inputs
                        if st.session_state["new_results"].success:
                            st.success("✅ Single footing design complete! Both shear checks PASS.")
                        else:
                            fails = []
                            if not st.session_state["new_results"].punching_ok:
                                fails.append("punching shear")
                            if not st.session_state["new_results"].oneway_ok:
                                fails.append("one-way shear")
                            st.warning(f"⚠️ Design complete, but {' and '.join(fails)} check(s) FAIL. "
                                       f"Consider increasing d or enabling auto-find.")
                except Exception as e:
                    st.error(f"❌ Design error: {str(e)}")

    if "new_results" in st.session_state:
        R = st.session_state["new_results"]
        I = st.session_state["new_inputs"]

        st.markdown(f"""
        <div style="background:{THEME['light_green'] if R.success else THEME['light_red']};
                    padding:.75rem 1.25rem; border-radius:8px; margin-bottom:1rem;
                    border-left:4px solid {THEME['green'] if R.success else THEME['red']};">
            <span style="font-weight:700; font-size:1.1rem;">
                {'✅ Design PASS' if R.success else '❌ Design FAIL'}
            </span>
            <span style="color:{THEME['slate']}; margin-left:1rem;">
                B = {R.footing_width:.1f} ft · h = {R.h_final} in · As = {R.As_final:.2f} in²
            </span>
        </div>
        """, unsafe_allow_html=True)

        nc1, nc2, nc3, nc4 = st.columns(4)
        with nc1:
            st.metric("Service Load", f"{R.service_load:.1f} k")
            st.metric("Factored Load", f"{R.factored_load:.1f} k")
        with nc2:
            st.metric("Net Soil Pressure qe", f"{R.qe:.3f} ksf")
            st.metric("Ultimate Pressure qu", f"{R.qu:.2f} ksf")
        with nc3:
            st.metric("Footing Width B", f"{R.footing_width:.1f} ft")
            st.metric("Provided Area", f"{R.A_provided:.2f} ft²")
        with nc4:
            st.metric("Effective Depth d", f"{I.trial_d:.1f} in")
            st.metric("Total Thickness h", f"{R.h_final} in")

        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

        sub_new1, sub_new2, sub_new3, sub_new4, sub_new5, sub_new6 = st.tabs([
            "📊 Summary", "📐 Plan & Section", "📝 Step-by-Step",
            "✂️ Shear Design", "🧷 Reinforcement", "📑 Report"
        ])

        # --- Summary ---
        with sub_new1:
            lc, rc = st.columns(2)
            with lc:
                st.markdown(f"""
                <div class="card card-teal"><div class="card-title">📐 Geometry & Soil</div>
                <div class="mono">Overburden = {R.overburden_pressure:.3f} ksf</div>
                <div class="mono">Net qe = {R.qe:.3f} ksf</div>
                <div class="mono">A_req = {R.A_req:.2f} ft²</div>
                <div class="mono">B = {R.footing_width:.1f} ft (A = {R.A_provided:.2f} ft²)</div>
                <div class="mono">qu = {R.qu:.2f} ksf</div>
                </div>
                """, unsafe_allow_html=True)
            with rc:
                st.markdown(f"""
                <div class="card card-amber"><div class="card-title">⚖️ Load Summary</div>
                <div class="mono">DL = {I.dead_load:.1f} k · LL = {I.live_load:.1f} k</div>
                <div class="mono">Service = {R.service_load:.1f} k</div>
                <div class="mono">Factored = 1.2*{I.dead_load:.0f} + 1.6*{I.live_load:.0f} = {R.factored_load:.1f} k</div>
                <div class="mono">Mu = {R.Mu:.0f} in-kips</div>
                <div class="mono">As_final = {R.As_final:.2f} in² (per direction)</div>
                </div>
                """, unsafe_allow_html=True)

        # --- Plan & Section ---
        with sub_new2:
            st.pyplot(draw_single_plan(R, I))
            st.pyplot(draw_single_section(R, I))
            st.caption("Red dashed = punching shear perimeter · green dotted = one-way shear critical sections · "
                       "orange = column · blue = footing · green arrows = upward soil pressure")

        # --- Step-by-Step ---
        with sub_new3:
            st.markdown(f"""
            <div class="step-box">
                <span class="step-num">1</span><b>Net Effective Soil Bearing Capacity (qe)</b><br>
                <span class="mono">
                Overburden = {I.footing_depth_ground} ft × {I.avg_unit_weight} pcf = {R.overburden_pressure * 1000:.0f} psf = {R.overburden_pressure:.3f} ksf<br>
                qe = {I.qa} − {R.overburden_pressure:.3f} = <b>{R.qe:.3f} ksf</b>
                </span>
            </div>
            <div class="step-box">
                <span class="step-num">2</span><b>Required Footing Area (A_req)</b><br>
                <span class="mono">
                Service Load = {I.dead_load} + {I.live_load} = {R.service_load} kips<br>
                A_req = {R.service_load} / {R.qe:.3f} = <b>{R.A_req:.2f} sq.ft.</b><br>
                Selecting square base: B = {R.footing_width} ft × {R.footing_width} ft → A = {R.A_provided:.2f} sq.ft.
                </span>
            </div>
            <div class="step-box">
                <span class="step-num">3</span><b>Factored Net Upward Soil Pressure (qu)</b><br>
                <span class="mono">
                Factored Load = 1.2 × {I.dead_load} + 1.6 × {I.live_load} = {R.factored_load} kips<br>
                qu = {R.factored_load} / {R.A_provided:.2f} = <b>{R.qu:.2f} ksf</b>
                </span>
            </div>
            <div class="step-box" style="border-left-color:{THEME['green'] if R.punching_ok else THEME['red']};">
                <span class="step-num">4</span><b>Two-Way (Punching) Shear Check (d = {I.trial_d} in.)</b><br>
                <span class="mono">
                bo = 4 × ({I.col_side} + {I.trial_d}) = {R.bo} in.<br>
                Vu1 = {R.qu:.2f} × [{R.A_provided:.2f} − ({R.critical_perimeter_side}/12)²] = <b>{R.Vu1:.1f} kips</b><br>
                phi*Vc = 0.75 × [4 × sqrt({R.fc_psi:.0f}) × {R.bo} × {I.trial_d} / 1000] = <b>{R.phi_Vc1:.1f} kips</b><br>
                DCR = {R.punching_dcr:.2f} → {'✅ ADEQUATE' if R.punching_ok else '❌ INADEQUATE'}
                </span>
            </div>
            <div class="step-box" style="border-left-color:{THEME['green'] if R.oneway_ok else THEME['red']};">
                <span class="step-num">5</span><b>One-Way (Beam) Shear Check</b><br>
                <span class="mono">
                Dist. to column face = {R.dist_to_face:.2f} ft<br>
                Beam shear length = {R.dist_to_face:.2f} − {I.trial_d / 12:.2f} = {R.beam_shear_length:.2f} ft<br>
                Vu2 = {R.qu:.2f} × {R.beam_shear_length:.2f} × {R.footing_width} = <b>{R.Vu2:.1f} kips</b><br>
                phi*Vc = 0.75 × [2 × sqrt({R.fc_psi:.0f}) × {R.footing_width * 12:.0f} × {I.trial_d} / 1000] = <b>{R.phi_Vc2:.1f} kips</b><br>
                DCR = {R.oneway_dcr:.2f} → {'✅ ADEQUATE' if R.oneway_ok else '❌ INADEQUATE'}
                </span>
            </div>
            <div class="step-box">
                <span class="step-num">6</span><b>Bending Moment and Reinforcement Design</b><br>
                <span class="mono">
                Cantilever arm = {R.cantilever_arm:.2f} ft<br>
                Mu = {R.qu:.2f} × {R.footing_width} × ({R.cantilever_arm:.2f}² / 2) × 12 = <b>{R.Mu:.0f} in-kips</b><br>
                As_calc (a={R.assumed_a} in.) = {R.Mu:.0f} / (0.90 × {I.fy} × ({I.trial_d} − {R.assumed_a / 2})) = <b>{R.As_calc:.2f} in²</b><br>
                As_min1 = 3*sqrt(f'c)/fy * b * d = <b>{R.As_min1:.2f} in²</b><br>
                As_min2 = 200/fy * b * d = <b>{R.As_min2:.2f} in²</b><br>
                Controlling As_min = <b>{R.As_min:.2f} in²</b><br>
                <b>Final As = max({R.As_calc:.2f}, {R.As_min:.2f}) = {R.As_final:.2f} in²</b> (per direction)
                </span>
            </div>
            <div class="step-box">
                <span class="step-num">7</span><b>Final Thickness Calculation</b><br>
                <span class="mono">
                h = d + 1.5 (bar depth) + 3.0 (cover) = {I.trial_d} + 1.5 + 3.0 = <b>{R.h_required:.1f} in</b><br>
                Recommended total footing thickness = <b>{R.h_final} in</b> ({R.h_final / 12:.2f} ft)
                </span>
            </div>
            """, unsafe_allow_html=True)

        # --- Shear Design ---
        with sub_new4:
            st.pyplot(draw_single_shear(R))
            sc1, sc2 = st.columns(2)
            with sc1:
                dcr_color = THEME["green"] if R.punching_ok else THEME["red"]
                st.markdown(f"""
                <div class="card" style="border-left-color:{dcr_color};">
                    <div class="card-title">Two-Way Punching Shear</div>
                    <div class="mono">Vu1 = {R.Vu1:.1f} kips</div>
                    <div class="mono">phi*Vc = {R.phi_Vc1:.1f} kips</div>
                    <div class="mono">DCR = {R.punching_dcr:.2f}</div>
                    <div style="margin-top:.5rem; font-weight:700; color:{dcr_color};">
                        {'✅ PASS' if R.punching_ok else '❌ FAIL'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                dcr_color2 = THEME["green"] if R.oneway_ok else THEME["red"]
                st.markdown(f"""
                <div class="card" style="border-left-color:{dcr_color2};">
                    <div class="card-title">One-Way Beam Shear</div>
                    <div class="mono">Vu2 = {R.Vu2:.1f} kips</div>
                    <div class="mono">phi*Vc = {R.phi_Vc2:.1f} kips</div>
                    <div class="mono">DCR = {R.oneway_dcr:.2f}</div>
                    <div style="margin-top:.5rem; font-weight:700; color:{dcr_color2};">
                        {'✅ PASS' if R.oneway_ok else '❌ FAIL'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Shear Check Details")
            shear_rows = [
                {"Check": "Two-Way (Punching)", "Vu (k)": f"{R.Vu1:.1f}",
                 "phi*Vc (k)": f"{R.phi_Vc1:.1f}", "DCR": f"{R.punching_dcr:.2f}",
                 "Status": "✅" if R.punching_ok else "❌"},
                {"Check": "One-Way (Beam)", "Vu (k)": f"{R.Vu2:.1f}",
                 "phi*Vc (k)": f"{R.phi_Vc2:.1f}", "DCR": f"{R.oneway_dcr:.2f}",
                 "Status": "✅" if R.oneway_ok else "❌"},
            ]
            st.dataframe(shear_rows, use_container_width=True, hide_index=True)

        # --- Reinforcement ---
        with sub_new5:
            st.markdown("#### Required Steel Areas (per direction)")
            st.markdown(f"""
            <div class="card card-teal">
                <div class="card-title">Calculated Steel Area</div>
                <div class="mono" style="font-size:1.3rem; color:{THEME['navy']};">
                    As_calc = {R.As_calc:.2f} in²
                </div>
                <div style="color:{THEME['slate']}; font-size:.85rem;">
                    Mu / (0.90 × fy × (d − a/2)) · assumed a = {R.assumed_a} in.
                </div>
            </div>
            <div class="card card-amber">
                <div class="card-title">ACI Minimum Steel</div>
                <div class="mono">3*sqrt(f'c)/fy * b * d = {R.As_min1:.2f} in²</div>
                <div class="mono">200/fy * b * d = {R.As_min2:.2f} in²</div>
                <div class="mono" style="font-size:1.1rem; color:{THEME['navy']};">
                    Controlling As_min = {R.As_min:.2f} in²
                </div>
            </div>
            <div class="card card-green">
                <div class="card-title">Final Controlling Steel Area</div>
                <div class="mono" style="font-size:1.5rem; color:{THEME['navy']};">
                    As_final = {R.As_final:.2f} in² (per direction)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Bar Selection Suggestion")
            bar_db = {3: .375, 4: .500, 5: .625, 6: .750, 7: .875, 8: 1.000,
                      9: 1.128, 10: 1.270, 11: 1.410, 14: 1.693, 18: 2.257}
            bar_A = {n: math.pi * (d / 2) ** 2 for n, d in bar_db.items()}

            def suggest_single(As_req, prefer=(6, 7, 8, 9, 10, 11)):
                best = None
                for n in prefer:
                    A = bar_A[n]
                    count = math.ceil(As_req / A + 1e-9)
                    if best is None or count * A < best[2] * bar_A[best[0]]:
                        best = (n, count, count * A)
                return best

            if R.As_final > 0:
                n, count, A_prov = suggest_single(R.As_final)
                length_ft = R.footing_width
                spacing = (length_ft * 12 - 6) / (count - 1) if count > 1 else 0
                bar_rows = [{
                    "Direction": "Both (EW)",
                    "As_req (in²)": f"{R.As_final:.2f}",
                    "Suggested Bar": f"#{n}",
                    "Count": str(count),
                    "As_prov (in²)": f"{A_prov:.2f}",
                    "Spacing (in)": f"{spacing:.1f}",
                }]
                st.dataframe(bar_rows, use_container_width=True, hide_index=True)
                st.caption(f"Square footing — same bar arrangement in both directions. "
                           f"Verify development length and spacing per ACI 318.")
            else:
                st.warning("No reinforcement required — check inputs.")

        # --- Report ---
        with sub_new6:
            rep_new = f"""
# Single Column Square Footing — Calculation Report

**Project:** {project_name}  |  **Job No.:** {job_no}
**Engineer:** {engineer}  |  **Date:** {date_str}

---

## 1. Input Parameters
- Dead Load = **{I.dead_load:.1f} kips**, Live Load = **{I.live_load:.1f} kips**
- Column dimension = **{I.col_side:.0f} in**
- f'c = **{I.fc:.1f} ksi** ({R.fc_psi:.0f} psi),  fy = **{I.fy:.1f} ksi**
- Allowable soil pressure qa = **{I.qa:.2f} ksf**
- Footing base depth below grade = **{I.footing_depth_ground:.1f} ft**
- Avg unit weight (soil + conc) = **{I.avg_unit_weight:.0f} pcf**
- Trial effective depth d = **{I.trial_d:.1f} in**

## 2. Step 1 — Net Effective Soil Bearing Capacity
- Overburden pressure = {I.footing_depth_ground:.1f} * {I.avg_unit_weight:.0f} / 1000 = **{R.overburden_pressure:.3f} ksf**
- qe = {I.qa:.2f} - {R.overburden_pressure:.3f} = **{R.qe:.3f} ksf**

## 3. Step 2 — Required Footing Area
- Service Load = {I.dead_load:.1f} + {I.live_load:.1f} = **{R.service_load:.1f} kips**
- A_req = {R.service_load:.1f} / {R.qe:.3f} = **{R.A_req:.2f} sq.ft.**
- Selected: B = **{R.footing_width:.1f} ft** x {R.footing_width:.1f} ft = **{R.A_provided:.2f} sq.ft.**

## 4. Step 3 — Factored Soil Pressure
- Factored Load = 1.2*{I.dead_load:.0f} + 1.6*{I.live_load:.0f} = **{R.factored_load:.1f} kips**
- qu = {R.factored_load:.1f} / {R.A_provided:.2f} = **{R.qu:.2f} ksf**

## 5. Step 4 — Two-Way (Punching) Shear Check
- Critical perimeter side = {I.col_side:.0f} + {I.trial_d:.0f} = {R.critical_perimeter_side:.1f} in
- bo = 4 * {R.critical_perimeter_side:.1f} = **{R.bo:.0f} in**
- Vu1 = {R.qu:.2f} * [{R.A_provided:.2f} - ({R.critical_perimeter_side:.1f}/12)^2] = **{R.Vu1:.1f} kips**
- phi*Vc = 0.75 * [4 * sqrt({R.fc_psi:.0f}) * {R.bo:.0f} * {I.trial_d:.0f} / 1000] = **{R.phi_Vc1:.1f} kips**
- DCR = {R.punching_dcr:.2f} -> {'PASS' if R.punching_ok else 'FAIL'}

## 6. Step 5 — One-Way (Beam) Shear Check
- Distance to column face = {R.dist_to_face:.2f} ft
- Beam shear length = {R.beam_shear_length:.2f} ft
- Vu2 = {R.qu:.2f} * {R.beam_shear_length:.2f} * {R.footing_width:.1f} = **{R.Vu2:.1f} kips**
- phi*Vc = 0.75 * [2 * sqrt({R.fc_psi:.0f}) * {R.footing_width * 12:.0f} * {I.trial_d:.0f} / 1000] = **{R.phi_Vc2:.1f} kips**
- DCR = {R.oneway_dcr:.2f} -> {'PASS' if R.oneway_ok else 'FAIL'}

## 7. Step 6 — Bending Moment and Reinforcement
- Cantilever arm = {R.cantilever_arm:.2f} ft
- Mu = {R.qu:.2f} * {R.footing_width:.1f} * ({R.cantilever_arm:.2f}^2 / 2) * 12 = **{R.Mu:.0f} in-kips**
- As_calc (a={R.assumed_a:.0f} in) = {R.Mu:.0f} / (0.90 * {I.fy:.0f} * ({I.trial_d:.0f} - {R.assumed_a / 2:.1f})) = **{R.As_calc:.2f} in2**
- As_min1 = 3*sqrt(f'c)/fy * b * d = **{R.As_min1:.2f} in2**
- As_min2 = 200/fy * b * d = **{R.As_min2:.2f} in2**
- As_min = max({R.As_min1:.2f}, {R.As_min2:.2f}) = **{R.As_min:.2f} in2**
- **As_final = max({R.As_calc:.2f}, {R.As_min:.2f}) = {R.As_final:.2f} in2** (per direction)

## 8. Step 7 — Final Thickness
- h = d + 1.5 (bar) + 3.0 (cover) = {I.trial_d:.0f} + 1.5 + 3.0 = **{R.h_required:.1f} in**
- Recommended total footing thickness = **{R.h_final} in** ({R.h_final / 12:.2f} ft)

---
*Generated by Civil Engineering Footing Designer (Single Column Footing).*
"""
            st.markdown(rep_new)
            st.download_button("⬇ Download Report (Markdown)", data=rep_new,
                               file_name=f"single_footing_report_{job_no}.md",
                               mime="text/markdown", key="new_dl")

# ============================================================
# FOOTER
# ============================================================

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:{THEME['slate']}; font-size:.85rem; padding-bottom:1rem;">
    <b>Civil Engineering Footing Designer</b> · Streamlit Application<br>
    Combined Rectangular Footing · Single Column Square Footing<br>
    ACI 318 Reinforced Concrete Design · {project_name} · {job_no}
</div>
""", unsafe_allow_html=True)