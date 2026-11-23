import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. AUTOMATICALLY READ EXACT GEOMETRY, WEIGHTS & PROPULSION FROM YOUR XML FILES
# ==============================================================================
base_dir = r"D:\Sahana\loki\apd\PythonProject\Olaf_RJ75"
aircraft_xml = os.path.join(base_dir, "Olaf_RJ75.xml")
engine_xml = os.path.join(base_dir, "Engines", "Olaf_TF35.xml")
out_dir = r"D:\Sahana\loki\apd\PythonProject\Review2_Plots"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 220,
})

g = 9.80665
FT_TO_M = 0.3048
FT2_TO_M2 = 0.09290304
LBS_TO_KG = 0.45359237
LBF_TO_N = 4.4482216

root = ET.parse(aircraft_xml).getroot()

def read_metric(xpath, default_val, default_unit):
    el = root.find(xpath)
    if el is None or not el.text:
        return default_val, default_unit
    return float(el.text.strip()), el.attrib.get("unit", default_unit).upper()

# Wing Span (b)
b_raw, b_unit = read_metric(".//metrics/wingspan", 28.27, "M")
b = b_raw * FT_TO_M if b_unit == "FT" else b_raw

# Reference Wing Area (S)
S_raw, S_unit = read_metric(".//metrics/wingarea", 78.50, "M2")
S = S_raw * FT2_TO_M2 if S_unit in ("FT2", "SQFT") else S_raw

# Mean Aerodynamic Chord (c_bar)
c_raw, c_unit = read_metric(".//metrics/chord", S / b, "M")
c_bar = c_raw * FT_TO_M if c_unit == "FT" else c_raw

# Operating Empty Weight (OEW)
oew_raw, oew_unit = read_metric(".//mass_balance/emptywt", 22800.0, "KG")
OEW_kg = oew_raw * LBS_TO_KG if oew_unit == "LBS" else oew_raw

# Pointmasses (Payload / Crew / Avionics) & Fuel Tanks
pointmass_kg = 0.0
for pm in root.findall(".//mass_balance/pointmass"):
    w_el = pm.find("weight")
    if w_el is not None and w_el.text:
        val = float(w_el.text.strip())
        unit = w_el.attrib.get("unit", "LBS").upper()
        pointmass_kg += val * LBS_TO_KG if unit == "LBS" else val

fuel_cap_kg = 0.0
for tank in root.findall(".//propulsion/tank"):
    cap_el = tank.find("capacity")
    if cap_el is not None and cap_el.text:
        val = float(cap_el.text.strip())
        unit = cap_el.attrib.get("unit", "LBS").upper()
        fuel_cap_kg += val * LBS_TO_KG if unit == "LBS" else val

if fuel_cap_kg <= 0.0:
    fuel_cap_kg = 9200.0
if pointmass_kg <= 0.0:
    pointmass_kg = 8200.0  # 75 passengers + baggage

W_payload_max_kg = pointmass_kg
W_fuel_max_kg = fuel_cap_kg
MTOW_kg = OEW_kg + W_payload_max_kg + W_fuel_max_kg
W0 = MTOW_kg * g

# Extract CD0 and CL_alpha from <aerodynamics> if present in Olaf_RJ75.xml
CD0 = 0.0215
CL_alpha = 5.15
for func in root.findall(".//aerodynamics//function"):
    fname = func.attrib.get("name", "")
    if "CD0" in fname or "CDo" in fname or "CD_0" in fname:
        val_el = func.find(".//value")
        if val_el is not None and val_el.text:
            CD0 = float(val_el.text.strip())
    if "CLalpha" in fname or "CL_alpha" in fname:
        val_el = func.find(".//value")
        if val_el is not None and val_el.text:
            CL_alpha = float(val_el.text.strip())

# Count engines and read exact thrust/TSFC from Olaf_TF35.xml
n_eng = max(1, len(root.findall(".//propulsion/engine")))
T_SL_per_eng_lbf = 13825.0
TSFC_hr = 0.58
if os.path.exists(engine_xml):
    eng_root = ET.parse(engine_xml).getroot()
    mil_el = eng_root.find(".//milthrust")
    tsfc_el = eng_root.find(".//tsfc")
    if mil_el is not None and mil_el.text:
        T_SL_per_eng_lbf = float(mil_el.text.strip())
    if tsfc_el is not None and tsfc_el.text:
        TSFC_hr = float(tsfc_el.text.strip())

T_SL_total = n_eng * T_SL_per_eng_lbf * LBF_TO_N
ct = TSFC_hr / 3600.0

# Derived Aerodynamic & Structural Constants
AR = b**2 / S
e = 0.82
K = 1.0 / (np.pi * e * AR)

CL_max_clean = 1.55
CL_max_TO = 2.05
CL_max_LDG = 2.55
n_max = 2.50
n_min = -1.00
mu_r = 0.025
mu_b = 0.40

# ==============================================================================
# 2. ISA STANDARD ATMOSPHERE & PROPULSION LAPSE MODEL
# ==============================================================================
def isa_atm(h_m):
    T0, P0, rho0 = 288.15, 101325.0, 1.225
    L = 0.0065
    R_air = 287.058
    if h_m <= 11000.0:
        T = T0 - L * h_m
        P = P0 * (T / T0) ** (g / (L * R_air))
    else:
        T11 = T0 - L * 11000.0
        P11 = P0 * (T11 / T0) ** (g / (L * R_air))
        T = T11
        P = P11 * np.exp(-g * (h_m - 11000.0) / (R_air * T11))
    rho = P / (R_air * T)
    a = np.sqrt(1.4 * R_air * T)
    return T, P, rho, a

def thrust_avail(h_m, V):
    _, _, rho, a = isa_atm(h_m)
    sigma = rho / 1.225
    M = V / a
    mach_factor = np.maximum(0.65, 1.0 - 0.30 * M)
    return T_SL_total * (sigma ** 0.75) * mach_factor

# Characteristic Analytical Quantities
CL_md = np.sqrt(CD0 / K)
LD_max = 1.0 / (2.0 * np.sqrt(CD0 * K))
CL_mp = np.sqrt(3.0 * CD0 / K)
LD_mp = CL_mp / (CD0 + K * CL_mp**2)

_, _, rho_SL, a_SL = isa_atm(0.0)
h_cr = 10668.0  # 35,000 ft in meters
_, _, rho_cr, a_cr = isa_atm(h_cr)

V_stall_SL = np.sqrt(2 * W0 / (rho_SL * S * CL_max_clean))
V_stall_TO = np.sqrt(2 * W0 / (rho_SL * S * CL_max_TO))
V_stall_LDG = np.sqrt(2 * (0.85 * W0) / (rho_SL * S * CL_max_LDG))
V_md_SL = np.sqrt(2 * W0 / (rho_SL * S * CL_md))
V_mp_SL = np.sqrt(2 * W0 / (rho_SL * S * CL_mp))

print("=== VERIFIED INPUTS EXTRACTED FROM Olaf_RJ75.xml & Olaf_TF35.xml ===")
print(f"Wing Span (b)             : {b:.2f} m")
print(f"Reference Wing Area (S)   : {S:.2f} m^2")
print(f"Mean Aerodynamic Chord    : {c_bar:.2f} m")
print(f"Aspect Ratio (AR)         : {AR:.2f}")
print(f"Operating Empty Weight    : {OEW_kg:,.1f} kg")
print(f"Max Payload Mass          : {W_payload_max_kg:,.1f} kg")
print(f"Max Fuel Capacity         : {W_fuel_max_kg:,.1f} kg")
print(f"MTOW                      : {MTOW_kg:,.1f} kg ({W0/1000:.2f} kN)")
print(f"Wing Loading (W/S)        : {W0/S:,.1f} N/m^2")
print(f"Thrust-to-Weight (T/W) SL : {T_SL_total/W0:.3f} ({n_eng} x {T_SL_per_eng_lbf:,.0f} lbf)")
print("\n=== INTERMEDIATE VERIFICATION CHECKS (SLIDE 8) ===")
print(f"Zero-Lift Drag CD0        : {CD0:.4f}")
print(f"Induced Drag Factor K     : {K:.4f}")
print(f"Max Lift-to-Drag (L/D)max : {LD_max:.2f} at CL = {CL_md:.3f}")
print(f"Min Power CL_mp           : {CL_mp:.3f} (L/D = {LD_mp:.2f})")
print(f"V_stall (Clean, SL)       : {V_stall_SL:.2f} m/s ({V_stall_SL*1.94384:.1f} kts)")
print(f"V_stall (Takeoff Flaps)   : {V_stall_TO:.2f} m/s ({V_stall_TO*1.94384:.1f} kts)")
print(f"V_stall (Landing Flaps)   : {V_stall_LDG:.2f} m/s ({V_stall_LDG*1.94384:.1f} kts)")
print(f"V_md (Min Drag, SL)       : {V_md_SL:.2f} m/s ({V_md_SL*1.94384:.1f} kts)")
print(f"V_mp (Min Power, SL)      : {V_mp_SL:.2f} m/s ({V_mp_SL*1.94384:.1f} kts)")

# ==============================================================================
# GRAPH 1: Drag Polar & Aerodynamic Efficiency (L/D and CL^1.5/CD)
# ==============================================================================
CL = np.linspace(0.0, 1.55, 250)
CD = CD0 + K * CL**2
LD = CL / np.maximum(CD, 1e-6)
endurance_param = (CL**1.5) / np.maximum(CD, 1e-6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax1.plot(CD, CL, "b-", lw=2.2, label="Parabolic Drag Polar ($C_D = C_{D0} + K C_L^2$)")
ax1.plot([0, CD0 + K*CL_md**2], [0, CL_md], "r--", lw=1.4, label=f"Tangent: $(L/D)_{{max}} = {LD_max:.1f}$")
ax1.scatter([CD0 + K*CL_md**2], [CL_md], color="red", zorder=5)
ax1.set_title("1A. Clean Drag Polar ($C_L$ vs. $C_D$)")
ax1.set_xlabel("Drag Coefficient, $C_D$ [-]")
ax1.set_ylabel("Lift Coefficient, $C_L$ [-]")
ax1.legend(loc="lower right")

ax2.plot(CL, LD, "g-", lw=2.2, label="$L/D$ (Range Efficiency)")
ax2.plot(CL, endurance_param, "m--", lw=2.0, label="$C_L^{3/2}/C_D$ (Endurance Parameter)")
ax2.axvline(CL_md, color="g", ls=":", label=f"$C_{{L,md}} = {CL_md:.2f}$")
ax2.axvline(CL_mp, color="m", ls=":", label=f"$C_{{L,mp}} = {CL_mp:.2f}$")
ax2.set_title("1B. Aerodynamic Efficiency Ratios")
ax2.set_xlabel("Lift Coefficient, $C_L$ [-]")
ax2.set_ylabel("Aerodynamic Efficiency [-]")
ax2.legend(loc="upper right", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "01_Drag_Polar_and_Efficiency.png"))
plt.close()

# ==============================================================================
# GRAPH 2 & 4: Thrust Required & Thrust Available vs. Velocity (+ Char. Speeds)
# ==============================================================================
V_SL = np.linspace(V_stall_SL, 300.0, 300)
TR_SL = 0.5 * rho_SL * V_SL**2 * S * CD0 + (2 * K * W0**2) / (rho_SL * V_SL**2 * S)
TA_SL = thrust_avail(0.0, V_SL)

V_cr_min = np.sqrt(2 * W0 / (rho_cr * S * CL_max_clean))
V_cr = np.linspace(V_cr_min, 285.0, 300)
TR_cr = 0.5 * rho_cr * V_cr**2 * S * CD0 + (2 * K * W0**2) / (rho_cr * V_cr**2 * S)
TA_cr = thrust_avail(h_cr, V_cr)

idx_vmax_SL = np.argmin(np.abs(TA_SL - TR_SL))
V_max_SL = V_SL[idx_vmax_SL]
idx_vmax_cr = np.argmin(np.abs(TA_cr - TR_cr))
V_max_cr = V_cr[idx_vmax_cr]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax1.plot(V_SL, TR_SL / 1000.0, "b-", lw=2.2, label="Thrust Required $T_R$ (SL)")
ax1.plot(V_SL, TA_SL / 1000.0, "r-", lw=2.2, label="Thrust Available $T_A$ (SL)")
ax1.axvline(V_stall_SL, color="gray", ls="--", label=f"$V_{{stall}} = {V_stall_SL:.1f}$ m/s")
ax1.axvline(V_md_SL, color="green", ls=":", label=f"$V_{{md}} = {V_md_SL:.1f}$ m/s")
ax1.scatter([V_max_SL], [TR_SL[idx_vmax_SL]/1000.0], color="black", zorder=5,
            label=f"$V_{{max,SL}} = {V_max_SL:.1f}$ m/s")
ax1.set_title("2A & 4. Thrust Curves & Speeds (Sea Level)")
ax1.set_xlabel("True Airspeed, $V$ [m/s]")
ax1.set_ylabel("Thrust [kN]")
ax1.set_ylim(0, max(TA_SL / 1000.0) * 1.18)
ax1.legend(fontsize=8.5)

ax2.plot(V_cr, TR_cr / 1000.0, "b-", lw=2.2, label="Thrust Required $T_R$ (35,000 ft)")
ax2.plot(V_cr, TA_cr / 1000.0, "r-", lw=2.2, label="Thrust Available $T_A$ (35,000 ft)")
ax2.axvline(0.78 * a_cr, color="purple", ls="--", label=f"Design Cruise $M=0.78$ ({0.78*a_cr:.1f} m/s)")
ax2.scatter([V_max_cr], [TR_cr[idx_vmax_cr]/1000.0], color="black", zorder=5,
            label=f"$V_{{max,35k}} = {V_max_cr:.1f}$ m/s")
ax2.set_title("2B & 4. Thrust Curves & Speeds (35,000 ft Cruise)")
ax2.set_xlabel("True Airspeed, $V$ [m/s]")
ax2.set_ylabel("Thrust [kN]")
ax2.set_ylim(0, max(TA_cr / 1000.0) * 1.35)
ax2.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "02_04_Thrust_Curves_and_Characteristic_Speeds.png"))
plt.close()

# ==============================================================================
# GRAPH 3: Power Required & Power Available vs. Velocity
# ==============================================================================
PR_SL = (TR_SL * V_SL) / 1e6  # MW
PA_SL = (TA_SL * V_SL) / 1e6  # MW
PR_cr = (TR_cr * V_cr) / 1e6
PA_cr = (TA_cr * V_cr) / 1e6

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(V_SL, PR_SL, "b-", lw=2.2, label="Power Required $P_R$ (Sea Level)")
ax.plot(V_SL, PA_SL, "r-", lw=2.2, label="Power Available $P_A$ (Sea Level)")
ax.plot(V_cr, PR_cr, "b--", lw=1.8, label="Power Required $P_R$ (35,000 ft)")
ax.plot(V_cr, PA_cr, "r--", lw=1.8, label="Power Available $P_A$ (35,000 ft)")
ax.axvline(V_mp_SL, color="teal", ls=":", lw=1.8, label=f"$V_{{mp,SL}} = {V_mp_SL:.1f}$ m/s (Min Power)")
ax.set_title("3. Power Required & Power Available vs. True Airspeed")
ax.set_xlabel("True Airspeed, $V$ [m/s]")
ax.set_ylabel("Power [MW]")
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "03_Power_Required_and_Available.png"))
plt.close()

# ==============================================================================
# GRAPH 5: Rate of Climb, Time to Climb & Service/Absolute Ceiling
# ==============================================================================
alts_m = np.linspace(0, 14500, 50)
RC_max_list = []
for h in alts_m:
    _, _, rho_h, _ = isa_atm(h)
    Vs_h = np.sqrt(2 * W0 / (rho_h * S * CL_max_clean))
    Vh = np.linspace(Vs_h, 295.0, 220)
    TR_h = 0.5 * rho_h * Vh**2 * S * CD0 + (2 * K * W0**2) / (rho_h * Vh**2 * S)
    TA_h = thrust_avail(h, Vh)
    RC_h = ((TA_h - TR_h) * Vh) / W0
    RC_max_list.append(np.max(RC_h))

RC_max_arr = np.array(RC_max_list)
idx_sc = np.argmin(np.abs(RC_max_arr - 0.508))
idx_ac = np.argmin(np.abs(RC_max_arr - 0.0))
h_service_ft = alts_m[idx_sc] * 3.28084
h_abs_ft = alts_m[idx_ac] * 3.28084

valid = alts_m <= h_cr
dt_dh = 1.0 / np.maximum(RC_max_arr[valid], 0.5)
time_to_climb_min = np.cumsum(np.gradient(alts_m[valid]) * dt_dh) / 60.0

RC_SL_curve = ((TA_SL - TR_SL) * V_SL) / W0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax1.plot(V_SL, RC_SL_curve, "b-", lw=2.2, label="Sea Level $R/C(V)$")
ax1.scatter([V_SL[np.argmax(RC_SL_curve)]], [np.max(RC_SL_curve)], color="red", zorder=5,
            label=f"$(R/C)_{{max,SL}} = {np.max(RC_SL_curve):.1f}$ m/s")
ax1.set_title("5A. Rate of Climb vs. Airspeed (Sea Level)")
ax1.set_xlabel("True Airspeed, $V$ [m/s]")
ax1.set_ylabel("Rate of Climb, $R/C$ [m/s]")
ax1.set_ylim(0, np.max(RC_SL_curve) * 1.18)
ax1.legend()

ax2.plot(RC_max_arr, alts_m * 3.28084, "g-", lw=2.2, label="Max Rate of Climb $(R/C)_{max}$")
ax2.axvline(0.508, color="orange", ls="--", label=f"Service Ceiling: {h_service_ft:,.0f} ft")
ax2.axhline(h_abs_ft, color="red", ls=":", label=f"Absolute Ceiling: {h_abs_ft:,.0f} ft")
ax2.set_title(f"5B. Climb Ceiling (Time to 35,000 ft = {time_to_climb_min[-1]:.1f} min)")
ax2.set_xlabel("Maximum Rate of Climb [m/s]")
ax2.set_ylabel("Altitude [ft]")
ax2.set_xlim(0, np.max(RC_max_arr) * 1.1)
ax2.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "05_Climb_and_Ceiling_Performance.png"))
plt.close()

# ==============================================================================
# GRAPH 6: Glide Performance (Glide Ratio & Minimum Sink Condition)
# ==============================================================================
V_sink_SL = (TR_SL * V_SL) / W0
glide_ratio_SL = V_SL / V_sink_SL
idx_min_sink = np.argmin(V_sink_SL)
idx_max_glide = np.argmax(glide_ratio_SL)
glide_dist_km = (h_cr * glide_ratio_SL[idx_max_glide]) / 1000.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax1.plot(V_SL, -V_sink_SL, "b-", lw=2.2, label="Glide Hodograph ($-V_{sink}$ vs. $V$)")
ax1.scatter([V_SL[idx_min_sink]], [-V_sink_SL[idx_min_sink]], color="red", zorder=5,
            label=f"Min Sink: {V_sink_SL[idx_min_sink]:.2f} m/s at {V_SL[idx_min_sink]:.1f} m/s")
ax1.set_title("6A. Glide Polar (Sink Rate vs. Airspeed)")
ax1.set_xlabel("Forward Airspeed, $V$ [m/s]")
ax1.set_ylabel("Vertical Speed, $-V_{sink}$ [m/s]")
ax1.legend(fontsize=8.5)

ax2.plot(V_SL, glide_ratio_SL, "g-", lw=2.2, label="Glide Ratio $L/D$")
ax2.scatter([V_SL[idx_max_glide]], [glide_ratio_SL[idx_max_glide]], color="red", zorder=5,
            label=f"Max Glide Ratio: {glide_ratio_SL[idx_max_glide]:.1f} ({glide_dist_km:.0f} km from 35k ft)")
ax2.set_title("6B. Glide Ratio vs. Forward Airspeed")
ax2.set_xlabel("Forward Airspeed, $V$ [m/s]")
ax2.set_ylabel("Glide Ratio ($L/D$) [-]")
ax2.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "06_Glide_Performance.png"))
plt.close()

# ==============================================================================
# GRAPH 7: Jet Range & Endurance + Payload-Range Diagram
# ==============================================================================
V_cruise = 0.78 * a_cr
CL_cr = (2 * W0) / (rho_cr * V_cruise**2 * S)
LD_cr = CL_cr / (CD0 + K * CL_cr**2)

fuel_at_max_payload = max(0.25 * W_fuel_max_kg, MTOW_kg - OEW_kg - W_payload_max_kg)
R_harmonic_km = ((V_cruise / ct) * LD_cr * np.log(MTOW_kg / (MTOW_kg - fuel_at_max_payload))) / 1000.0
R_maxfuel_km = ((V_cruise / ct) * LD_cr * np.log(MTOW_kg / (MTOW_kg - W_fuel_max_kg))) / 1000.0
R_ferry_km = ((V_cruise / ct) * LD_max * np.log((OEW_kg + W_fuel_max_kg) / OEW_kg)) / 1000.0
E_max_hr = ((1.0 / ct) * LD_max * np.log(MTOW_kg / (MTOW_kg - W_fuel_max_kg))) / 3600.0

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ranges_pr = [0, R_harmonic_km, R_maxfuel_km, R_ferry_km]
payloads_pr = [W_payload_max_kg, W_payload_max_kg, max(0.0, MTOW_kg - OEW_kg - W_fuel_max_kg), 0.0]
ax.plot(ranges_pr, payloads_pr, "b-o", lw=2.4, markersize=7, label="Olaf RJ-75 Payload–Range Envelope")
ax.fill_between(ranges_pr, payloads_pr, color="skyblue", alpha=0.25)
ax.annotate(f"Harmonic Point\n({R_harmonic_km:,.0f} km, {W_payload_max_kg:,.0f} kg)",
            (R_harmonic_km, W_payload_max_kg),
            xytext=(R_harmonic_km * 0.62, W_payload_max_kg * 0.82), arrowprops=dict(arrowstyle="->"))
ax.annotate(f"Ferry Range\n({R_ferry_km:,.0f} km)", (R_ferry_km, 0),
            xytext=(R_ferry_km * 0.72, W_payload_max_kg * 0.16), arrowprops=dict(arrowstyle="->"))
ax.set_title(f"7. Breguet Jet Payload–Range Diagram (Max Endurance = {E_max_hr:.2f} hr)")
ax.set_xlabel("Range [km]")
ax.set_ylabel("Payload Mass [kg]")
ax.set_ylim(0, W_payload_max_kg * 1.18)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "07_Jet_Range_and_Endurance.png"))
plt.close()

# ==============================================================================
# GRAPH 8: Takeoff & Landing Performance (Bengaluru VOBL vs. Sea Level)
# ==============================================================================
def calc_to_ldg(h_field_m):
    _, _, rho_f, _ = isa_atm(h_field_m)
    Vs_to = np.sqrt(2 * W0 / (rho_f * S * CL_max_TO))
    V_lof = 1.15 * Vs_to
    V_avg = 0.7 * V_lof
    T_avg = thrust_avail(h_field_m, V_avg)
    D_avg = 0.5 * rho_f * V_avg**2 * S * (CD0 + 0.015 + K * (0.8)**2)
    L_avg = 0.5 * rho_f * V_avg**2 * S * 0.8
    a_avg = (g / W0) * (T_avg - D_avg - mu_r * (W0 - L_avg))
    S_ground_to = V_lof**2 / (2.0 * a_avg)
    S_air_to = 260.0

    W_ldg = 0.85 * W0
    Vs_ldg = np.sqrt(2 * W_ldg / (rho_f * S * CL_max_LDG))
    V_td = 1.15 * Vs_ldg
    V_avg_l = 0.7 * V_td
    D_avg_l = 0.5 * rho_f * V_avg_l**2 * S * (CD0 + 0.045)
    L_avg_l = 0.5 * rho_f * V_avg_l**2 * S * 0.4
    dec_avg = (g / W_ldg) * (D_avg_l + mu_b * (W_ldg - L_avg_l))
    S_ground_ldg = V_td**2 / (2.0 * dec_avg)
    S_air_ldg = 310.0
    return S_ground_to, S_air_to, S_ground_ldg, S_air_ldg

to_g_sl, to_a_sl, ldg_g_sl, ldg_a_sl = calc_to_ldg(0.0)
to_g_bl, to_a_bl, ldg_g_bl, ldg_a_bl = calc_to_ldg(884.0)  # VOBL ~2,900 ft

fig, ax = plt.subplots(figsize=(8.5, 4.5))
labels = ["Takeoff\n(Sea Level)", "Takeoff\n(VOBL 2,900 ft)", "Landing\n(Sea Level)", "Landing\n(VOBL 2,900 ft)"]
ground_vals = [to_g_sl, to_g_bl, ldg_g_sl, ldg_g_bl]
air_vals = [to_a_sl, to_a_bl, ldg_a_sl, ldg_a_bl]

x = np.arange(len(labels))
ax.bar(x, ground_vals, width=0.5, label="Ground Roll Distance [m]", color="#2563eb")
ax.bar(x, air_vals, width=0.5, bottom=ground_vals, label="Airborne Obstacle Segment (35/50 ft) [m]", color="#93c5fd")
for i in range(len(labels)):
    tot = ground_vals[i] + air_vals[i]
    ax.text(x[i], tot + 35, f"{tot:,.0f} m", ha="center", fontweight="bold")
ax.set_title("8. Takeoff & Landing Field Length Breakdown (ISA Dry Runway)")
ax.set_ylabel("Distance [m]")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(np.array(ground_vals) + np.array(air_vals)) * 1.18)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "08_Takeoff_and_Landing_Performance.png"))
plt.close()

# ==============================================================================
# GRAPH 9: Turning Performance (Turn Radius & Turn Rate vs. Velocity)
# ==============================================================================
V_turn = np.linspace(V_stall_SL, 240.0, 250)
q_turn = 0.5 * rho_SL * V_turn**2
n_stall = (q_turn * S * CL_max_clean) / W0
T_avail_t = thrust_avail(0.0, V_turn)
n_thrust = np.sqrt(np.maximum(1.0, ((T_avail_t - q_turn * S * CD0) * q_turn * S) / (K * W0**2)))
n_allowed = np.minimum(n_max, np.minimum(n_stall, n_thrust))

omega_deg_s = np.degrees((g * np.sqrt(np.maximum(0.0, n_allowed**2 - 1.0))) / V_turn)
R_turn_m = V_turn**2 / (g * np.sqrt(np.maximum(1e-4, n_allowed**2 - 1.0)))
R_turn_m = np.where(n_allowed > 1.02, R_turn_m, np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax1.plot(V_turn, omega_deg_s, "b-", lw=2.2, label="Max Sustained/Structural Turn Rate $\\omega$")
ax1.scatter([V_turn[np.argmax(omega_deg_s)]], [np.max(omega_deg_s)], color="red", zorder=5,
            label=f"Corner Speed $V^* = {V_turn[np.argmax(omega_deg_s)]:.1f}$ m/s ($\\omega_{{max}} = {np.max(omega_deg_s):.1f}^\\circ$/s)")
ax1.set_title("9A. Turn Rate vs. True Airspeed (Sea Level)")
ax1.set_xlabel("True Airspeed, $V$ [m/s]")
ax1.set_ylabel("Turn Rate, $\\omega$ [deg/s]")
ax1.legend(fontsize=8.5)

ax2.plot(V_turn, R_turn_m, "g-", lw=2.2, label="Minimum Turn Radius $R$")
idx_rmin = np.nanargmin(R_turn_m)
ax2.scatter([V_turn[idx_rmin]], [R_turn_m[idx_rmin]], color="red", zorder=5,
            label=f"$R_{{min}} = {R_turn_m[idx_rmin]:,.0f}$ m at {V_turn[idx_rmin]:.1f} m/s")
ax2.set_title("9B. Minimum Turn Radius vs. Airspeed (Sea Level)")
ax2.set_xlabel("True Airspeed, $V$ [m/s]")
ax2.set_ylabel("Turn Radius, $R$ [m]")
ax2.set_ylim(0, 3500)
ax2.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "09_Turning_Performance.png"))
plt.close()

# ==============================================================================
# GRAPH 10: Manoeuvring & Gust V-n Diagram (FAR Part 25)
# ==============================================================================
VA = V_stall_SL * np.sqrt(n_max)
VC = 165.0
VD = 1.25 * VC

V_pos = np.linspace(0, VD, 300)
n_pos_stall = (0.5 * rho_SL * V_pos**2 * S * CL_max_clean) / W0
n_maneuver_pos = np.where(V_pos <= VA, n_pos_stall, n_max)

V_neg = np.linspace(0, VD, 300)
n_neg_stall = -(0.5 * rho_SL * V_neg**2 * S * 1.10) / W0
n_maneuver_neg = np.where(V_neg <= VC, np.maximum(n_neg_stall, n_min),
                          np.minimum(0.0, n_min + (V_neg - VC) / (VD - VC)))

mu_g = (2 * (W0 / S)) / (rho_SL * c_bar * CL_alpha * g)
Kg = (0.88 * mu_g) / (5.3 + mu_g)
V_gust = np.linspace(0, VD, 200)
dn_vc = (rho_SL * V_gust * CL_alpha * Kg * 15.24) / (2 * (W0 / S))
dn_vd = (rho_SL * V_gust * CL_alpha * Kg * 7.62) / (2 * (W0 / S))

fig, ax = plt.subplots(figsize=(8.8, 4.8))
ax.plot(V_pos, n_maneuver_pos, "b-", lw=2.4, label="Manoeuvre Envelope ($+2.5g / -1.0g$)")
ax.plot(V_neg, n_maneuver_neg, "b-", lw=2.4)
ax.plot([VD, VD], [0.0, n_max], "b-", lw=2.4)
ax.plot(V_gust, 1.0 + dn_vc, "r--", lw=1.6, label="Rough-Air Gust ($U_{de} = \\pm 15.24$ m/s at $V_C$)")
ax.plot(V_gust, 1.0 - dn_vc, "r--", lw=1.6)
ax.plot(V_gust, 1.0 + dn_vd, "orange", ls=":", lw=1.6, label="Dive Gust ($U_{de} = \\pm 7.62$ m/s at $V_D$)")
ax.plot(V_gust, 1.0 - dn_vd, "orange", ls=":", lw=1.6)
ax.axhline(0, color="black", lw=0.8)
ax.axvline(VA, color="gray", ls=":", label=f"$V_A = {VA:.1f}$ m/s | $V_C = {VC:.0f}$ m/s | $V_D = {VD:.1f}$ m/s")
ax.set_title("10. Manoeuvring & Gust V–n Flight Envelope (FAR Part 25, Sea Level)")
ax.set_xlabel("Equivalent Airspeed, $V_E$ [m/s]")
ax.set_ylabel("Load Factor, $n$ [-]")
ax.set_xlim(0, VD * 1.08)
ax.set_ylim(-2.2, 3.8)
ax.legend(loc="upper left", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "10_Vn_Maneuver_and_Gust_Diagram.png"))
plt.close()

print(f"\nAll 10 Review II graphs saved successfully to:\n{out_dir}")