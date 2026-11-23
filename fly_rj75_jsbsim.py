import os
import jsbsim
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# OLAF RJ-75 — JSBSIM 6-DOF FLIGHT DYNAMICS & STABILITY TEST SUITE
# Integrates: OpenVSP Mass/Inertia + VSPAERO + XFLR5 + EngineSim
# ==============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Resolve paths automatically inside D:\Sahana\loki\apd\PythonProject\
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, "Olaf_RJ75.xml")):
    model_dir = script_dir
    aircraft_root = os.path.dirname(script_dir)
else:
    aircraft_root = script_dir
    model_dir = os.path.join(script_dir, "Olaf_RJ75")

engine_dir = os.path.join(model_dir, "engines")
plot_dir = os.path.join(model_dir, "JSBSim_Flight_Plots")
os.makedirs(engine_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Ensure EngineSim-calibrated turbofan & nozzle XMLs exist in Olaf_RJ75/engines/
direct_xml = """<?xml version="1.0"?>
<direct name="Direct">
  <thruster>
    <sense>1.0</sense>
  </thruster>
</direct>
"""

engine_xml = """<?xml version="1.0"?>
<turbine_engine name="Olaf_TF35">
  <milthrust> 13825.0 </milthrust>
  <maxthrust> 13825.0 </maxthrust>
  <bypassratio> 5.2 </bypassratio>
  <tsfc> 0.58 </tsfc>
  <atsfc> 0.58 </atsfc>
  <bleed> 0.03 </bleed>
  <idlen1> 28.0 </idlen1>
  <idlen2> 58.0 </idlen2>
  <maxn1> 100.0 </maxn1>
  <maxn2> 100.0 </maxn2>
  <augmented> 0 </augmented>
  <injected> 0 </injected>

  <function name="IdleThrust">
    <table>
      <independentVar lookup="row">velocities/mach</independentVar>
      <independentVar lookup="column">atmosphere/density-altitude</independentVar>
      <tableData>
               0.0     15000.0   35000.0
        0.0    0.045   0.040     0.035
        0.5    0.025   0.022     0.020
        0.85   0.015   0.015     0.018
      </tableData>
    </table>
  </function>

  <function name="MilThrust">
    <table>
      <independentVar lookup="row">velocities/mach</independentVar>
      <independentVar lookup="column">atmosphere/density-altitude</independentVar>
      <tableData>
               0.0     15000.0   35000.0
        0.0    1.000   0.650     0.320
        0.4    0.840   0.560     0.275
        0.78   0.760   0.510     0.240
        0.85   0.730   0.490     0.230
      </tableData>
    </table>
  </function>
</turbine_engine>
"""

with open(os.path.join(engine_dir, "direct.xml"), "w", encoding="utf-8") as f:
    f.write(direct_xml)
with open(os.path.join(engine_dir, "Olaf_TF35.xml"), "w", encoding="utf-8") as f:
    f.write(engine_xml)


def init_fdm():
    fdm = jsbsim.FGFDMExec(aircraft_root)
    fdm.set_aircraft_path(aircraft_root)
    fdm.set_engine_path(engine_dir)
    if not fdm.load_model("Olaf_RJ75"):
        raise RuntimeError(f"Failed to load Olaf_RJ75.xml from {model_dir}")
    return fdm


# ==============================================================================
# FLIGHT TEST 1: CRUISE TRIM & 6-DOF DYNAMIC STABILITY (35,000 ft, Mach 0.78)
# ==============================================================================
fdm = init_fdm()

fdm["ic/h-sl-ft"] = 35000.0        # 10,668 m
fdm["ic/mach"] = 0.78
fdm["ic/alpha-deg"] = 1.15
fdm["ic/theta-deg"] = 1.15
fdm["ic/gamma-deg"] = 0.0
fdm["ic/beta-deg"] = 0.0
fdm["gear/gear-cmd-norm"] = 0.0
fdm["gear/gear-pos-norm"] = 0.0
fdm["fcs/flap-cmd-norm"] = 0.0

fdm["propulsion/set-running"] = -1
fdm.run_ic()

try:
    fdm.do_trim(1)  # 1 = Longitudinal steady-level cruise trim
except RuntimeError as e:
    print(f"[Notice] Trim solver note: {e}")

mass_kg = fdm["inertia/weight-lbs"] * 0.45359237
thrust_eng_N = fdm["propulsion/engine[0]/thrust-lbs"] * 4.4482216
trim_elev = fdm["fcs/elevator-cmd-norm"]

print("=" * 70)
print("TEST 1: JSBSIM 6-DOF CRUISE TRIM VERIFICATION (10,668 m, Mach 0.78)")
print("=" * 70)
print(f"Total Aircraft Mass             : {mass_kg:,.1f} kg")
print(f"Trimmed Angle of Attack (alpha) : {fdm['aero/alpha-deg']:.2f} deg")
print(f"Trimmed Elevator Deflection     : {np.degrees(fdm['fcs/elevator-pos-rad']):.2f} deg")
print(f"Trimmed Cruise Throttle         : {fdm['fcs/throttle-cmd-norm[0]'] * 100:.1f} %")
print(f"Net Thrust per Engine           : {thrust_eng_N:,.0f} N (Total: {2 * thrust_eng_N / 1000:.2f} kN)")
print("=" * 70)

# Run 90-second simulation with a 2-second nose-up elevator doublet at t = 5s
t_hist, alt_hist, mach_hist, alpha_hist, theta_hist, nz_hist = [], [], [], [], [], []

while fdm.get_sim_time() < 90.0:
    t = fdm.get_sim_time()
    if 5.0 <= t < 6.0:
        fdm["fcs/elevator-cmd-norm"] = trim_elev - 0.08   # Nose-up pulse
    elif 6.0 <= t < 7.0:
        fdm["fcs/elevator-cmd-norm"] = trim_elev + 0.08   # Nose-down recovery pulse
    else:
        fdm["fcs/elevator-cmd-norm"] = trim_elev

    fdm.run()

    t_hist.append(t)
    alt_hist.append(fdm["position/h-sl-ft"] * 0.3048)
    mach_hist.append(fdm["velocities/mach"])
    alpha_hist.append(fdm["aero/alpha-deg"])
    theta_hist.append(fdm["attitude/theta-deg"])
    nz_hist.append(-fdm["accelerations/n-pilot-z-norm"])

fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
fig.suptitle("Olaf RJ-75 — JSBSim 6-DOF Cruise Dynamic Stability (Elevator Doublet at t = 5–7 s)", fontweight="bold")

axs[0, 0].plot(t_hist, alpha_hist, color="#1d4ed8", lw=2.0)
axs[0, 0].set_ylabel("Angle of Attack $\\alpha$ [deg]")
axs[0, 0].set_title("Short-Period Mode Damping ($\\alpha$)")

axs[0, 1].plot(t_hist, theta_hist, color="#7c3aed", lw=2.0)
axs[0, 1].set_ylabel("Pitch Attitude $\\theta$ [deg]")
axs[0, 1].set_title("Pitch Attitude Response ($\\theta$)")

axs[1, 0].plot(t_hist, alt_hist, color="#059669", lw=2.0)
axs[1, 0].set_ylabel("Altitude [m]")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_title("Long-Period Phugoid Altitude Response")

axs[1, 1].plot(t_hist, mach_hist, color="#dc2626", lw=2.0)
axs[1, 1].set_ylabel("Mach Number")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_title("Cruise Mach Stability")

plt.tight_layout()
plot1_path = os.path.join(plot_dir, "07_JSBSim_6DOF_Cruise_Stability.png")
plt.savefig(plot1_path)
plt.close()


# ==============================================================================
# FLIGHT TEST 2: RUNWAY TAKEOFF GROUND ROLL, ROTATION & CLIMB-OUT (SEA LEVEL)
# ==============================================================================
fdm_to = init_fdm()

fdm_to["ic/h-agl-ft"] = 7.8            # Bogeys resting on sea-level runway
fdm_to["ic/vc-kts"] = 0.0
fdm_to["ic/theta-deg"] = 0.0
fdm_to["ic/phi-deg"] = 0.0
fdm_to["ic/psi-true-deg"] = 90.0       # Runway 09 heading
fdm_to["gear/gear-cmd-norm"] = 1.0
fdm_to["gear/gear-pos-norm"] = 1.0
fdm_to["fcs/flap-cmd-norm"] = 0.5      # Takeoff Flaps 15 deg

fdm_to["propulsion/set-running"] = -1
fdm_to.run_ic()
try:
    fdm_to.do_trim(2)  # 2 = Ground trim on landing gear
except RuntimeError:
    pass

# Release brakes and apply 100% Takeoff Thrust
fdm_to["fcs/left-brake-cmd-norm"] = 0.0
fdm_to["fcs/right-brake-cmd-norm"] = 0.0
fdm_to["fcs/center-brake-cmd-norm"] = 0.0
fdm_to["fcs/throttle-cmd-norm[0]"] = 1.0
fdm_to["fcs/throttle-cmd-norm[1]"] = 1.0

to_t, to_v, to_alt, to_theta, to_dist = [], [], [], [], []
liftoff_time, liftoff_dist, liftoff_speed = None, None, None
V_rotate_ms = 59.0  # Rotation speed VR (~115 kts, above Vs_TO = 56.3 m/s)

while fdm_to.get_sim_time() < 55.0:
    t = fdm_to.get_sim_time()
    v_ms = fdm_to["velocities/vt-fps"] * 0.3048
    alt_m = fdm_to["position/h-agl-ft"] * 0.3048
    theta_deg = fdm_to["attitude/theta-deg"]
    q_deg_s = np.degrees(fdm_to["velocities/q-aero-rad_sec"])

    # Rotation & Climb Pitch-Attitude Autopilot
    if v_ms < V_rotate_ms:
        fdm_to["fcs/elevator-cmd-norm"] = 0.0
    else:
        target_theta = 9.5  # Target initial climb pitch attitude [deg]
        err = target_theta - theta_deg
        cmd = -0.045 * err + 0.025 * q_deg_s
        fdm_to["fcs/elevator-cmd-norm"] = float(np.clip(cmd, -0.45, 0.35))

    # Record Liftoff Point
    if liftoff_time is None and alt_m > 0.5:
        liftoff_time = t
        liftoff_speed = v_ms
        liftoff_dist = np.trapezoid(to_v, to_t) if len(to_t) > 1 else 0.0

    # Retract Landing Gear once safely airborne (> 15 m AGL)
    if alt_m > 15.0:
        fdm_to["gear/gear-cmd-norm"] = 0.0

    fdm_to.run()

    to_t.append(t)
    to_v.append(v_ms)
    to_alt.append(alt_m)
    to_theta.append(theta_deg)
    to_dist.append(np.trapezoid(to_v, to_t) if len(to_t) > 1 else 0.0)

print("\n" + "=" * 70)
print("TEST 2: JSBSIM 6-DOF TAKEOFF & CLIMB VERIFICATION (SEA LEVEL, FLAPS 15)")
print("=" * 70)
if liftoff_time:
    print(f"Liftoff Time                    : {liftoff_time:.1f} s")
    print(f"Liftoff Airspeed (V_LOF)        : {liftoff_speed:.2f} m/s ({liftoff_speed * 1.94384:.1f} kts)")
    print(f"Takeoff Ground Roll Distance    : {liftoff_dist:.0f} m")
print(f"Altitude at t = 55 s            : {to_alt[-1]:.1f} m ({to_alt[-1] * 3.28084:.0f} ft)")
print(f"Climb Airspeed at t = 55 s      : {to_v[-1]:.2f} m/s ({to_v[-1] * 1.94384:.1f} kts)")
print("=" * 70)

fig, axs = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True)
fig.suptitle("Olaf RJ-75 — JSBSim 6-DOF Runway Takeoff Roll, Rotation & Climb-Out", fontweight="bold")

axs[0].plot(to_t, to_v, color="#1d4ed8", lw=2.2)
axs[0].axvline(V_rotate_ms, color="#f59e0b", ls=":", lw=1.2)
if liftoff_time:
    axs[0].axvline(liftoff_time, color="#059669", ls="--", lw=1.4, label=f"Liftoff ($t={liftoff_time:.1f}$ s, $V={liftoff_speed:.1f}$ m/s, Roll={liftoff_dist:.0f} m)")
axs[0].set_ylabel("Airspeed $V_{\\text{TAS}}$ [m/s]")
axs[0].set_title("Takeoff Acceleration & Climb Speed")
axs[0].legend(loc="lower right")

axs[1].plot(to_t, to_theta, color="#7c3aed", lw=2.2)
if liftoff_time:
    axs[1].axvline(liftoff_time, color="#059669", ls="--", lw=1.4)
axs[1].set_ylabel("Pitch Angle $\\theta$ [deg]")
axs[1].set_title("Rotation & Climb Pitch Attitude")

axs[2].plot(to_t, to_alt, color="#059669", lw=2.2)
if liftoff_time:
    axs[2].axvline(liftoff_time, color="#059669", ls="--", lw=1.4)
axs[2].set_ylabel("Altitude AGL [m]")
axs[2].set_xlabel("Time from Brake Release [s]")
axs[2].set_title("Height Above Runway (Gear Retracted at 15 m)")

plt.tight_layout()
plot2_path = os.path.join(plot_dir, "08_JSBSim_6DOF_Takeoff_Climb.png")
plt.savefig(plot2_path)
plt.close()

print(f"\nSaved JSBSim 6-DOF flight test plots to:\n  1) {plot1_path}\n  2) {plot2_path}")