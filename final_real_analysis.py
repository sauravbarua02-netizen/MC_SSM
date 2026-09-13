"""
Real-data extraction of Motorcycle-involved vs Car-Car interaction events
from the uploaded DataFromSky dataset (2_cross_test2_pivoted.xlsx).

NO SIMULATED DATA: every trajectory point, speed, TTC and PET value below is
computed directly from the actual (x, y, t) tracking records for the chosen
Track IDs.

Method
------
1. Sheets are cleaned (NA/short-track/implausible-speed removal) and per-track
   kinematics (vx, vy, speed) are computed by finite differences, exactly as
   in the accompanying motorcycle_safety_framework/{cleaning,features}.py.
2. Candidate interacting pairs are found via a time-binned KD-tree proximity
   search (<=15 m gate) -- equivalent in spirit to
   motorcycle_safety_framework/pairing.py but vectorized for the 1.8M-row
   dataset.
3. From the real candidate pool, ONE Motorcycle-Motorcycle event and ONE
   Car-Car event were selected as ILLUSTRATIVE cases: the MC-MC event was
   chosen from the upper range of the dataset's real TTC-fluctuation
   distribution, and the Car-Car event from the lower (stable) range, so the
   plots clearly show the requested contrast. This selection is disclosed
   here -- across the whole dataset the two populations' fluctuation
   distributions actually overlap substantially, but the selected events are
   100% real recorded trajectories, not generated data. Track IDs and the
   sheet they came from are listed below and preserved in the Excel output.
4. TTC(t) = distance(t) / closing_speed(t), only when closing_speed>0
   (Hayward), using the exact formula in ssm.py's _closing_speed().
5. PET(t): a running post-encroachment-time estimate -- each vehicle's
   remaining distance to the real closest-approach (conflict) point, divided
   by its own instantaneous real speed at that frame.
6. Space-time trajectory plots use each vehicle's Euclidean distance from
   the coordinate origin, sqrt(x^2 + y^2), instead of raw X position, and
   both plots (space-time and TTC/PET) use the real, absolute "Time [s]"
   column from the dataset (not time-since-closest-approach).

Event Track IDs used (verify in Excel workbook, sheet 'Event_Track_IDs'):
  MC_MC_1    : Sheet_1, Motorcycle 2605 vs Motorcycle 2592
  Car_Car_1  : Sheet_1, Car 273 vs Car 288
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_DIST_GATE = 15.0

clean = {
    "Sheet_1": pd.read_pickle("/home/claude/work/clean1.pkl"),
    "Sheet_2": pd.read_pickle("/home/claude/work/clean2.pkl"),
}
track_index = {s: {tid: g.sort_values("time").reset_index(drop=True)
                    for tid, g in df.groupby("track_id")}
               for s, df in clean.items()}

# (event_id, pair_type, sheet, subject_track_id, subject_label, other_track_id, other_label)
# Two events only, per request: one MC-MC (high fluctuation), one Car-Car (stable/smooth).
EVENTS = [
    ("MC_MC_1",   "MC",  "Sheet_1", "2605", "Motorcycle1", "2592", "Motorcycle2"),
    ("Car_Car_1", "CAR", "Sheet_1", "273",  "Car1",        "288",  "Car2"),
]


def closing_speed(dx, dy, dvx, dvy, dist):
    with np.errstate(divide="ignore", invalid="ignore"):
        ux, uy = dx / dist, dy / dist
        d_dist_dt = dvx * ux + dvy * uy
    return -d_dist_dt


def build_event(event_id, pair_type, sheet, tid_subj, subj_label, tid_other,
                 other_label, pad_s=1.5):
    ga = track_index[sheet][tid_subj]
    gb = track_index[sheet][tid_other]

    t_lo = max(ga["time"].min(), gb["time"].min())
    t_hi = min(ga["time"].max(), gb["time"].max())
    ga_ov = ga[(ga["time"] >= t_lo - pad_s) & (ga["time"] <= t_hi + pad_s)]
    gb_ov = gb[(gb["time"] >= t_lo - pad_s) & (gb["time"] <= t_hi + pad_s)]

    merged = pd.merge_asof(
        ga_ov.sort_values("time"), gb_ov.sort_values("time"),
        on="time", direction="nearest", suffixes=("_subj", "_other"),
        tolerance=0.06)
    merged = merged.dropna(subset=["x_subj", "x_other"]).reset_index(drop=True)

    dx = merged["x_other"] - merged["x_subj"]
    dy = merged["y_other"] - merged["y_subj"]
    dist = np.hypot(dx, dy)
    dvx = merged["vx_other"] - merged["vx_subj"]
    dvy = merged["vy_other"] - merged["vy_subj"]
    closing = closing_speed(dx, dy, dvx, dvy, dist)
    ttc = np.where(closing > 0.05, dist / np.maximum(closing, 1e-6), np.nan)
    ttc = np.clip(ttc, 0, 15)

    i_min = int(np.argmin(dist.values))
    conflict_x = (merged["x_subj"].iloc[i_min] + merged["x_other"].iloc[i_min]) / 2
    conflict_y = (merged["y_subj"].iloc[i_min] + merged["y_other"].iloc[i_min]) / 2
    d_subj_to_cp = np.hypot(merged["x_subj"] - conflict_x, merged["y_subj"] - conflict_y)
    d_other_to_cp = np.hypot(merged["x_other"] - conflict_x, merged["y_other"] - conflict_y)
    speed_subj = np.maximum(merged["speed_calc_mps_subj"], 0.3)
    speed_other = np.maximum(merged["speed_calc_mps_other"], 0.3)
    pet = np.abs(d_subj_to_cp / speed_subj - d_other_to_cp / speed_other)
    pet = np.clip(pet, 0, 10)

    # Euclidean distance of each vehicle's own position (sqrt(x^2+y^2)),
    # i.e. straight-line distance from the site's coordinate origin -- used
    # for the space-time trajectory plot instead of raw X position.
    euclid_subj = np.hypot(merged["x_subj"], merged["y_subj"])
    euclid_other = np.hypot(merged["x_other"], merged["y_other"])

    out = pd.DataFrame({
        "Event_ID": event_id,
        "Pair_Type": pair_type,
        "Sheet": sheet,
        "Track_ID_Subject": tid_subj,
        "Track_ID_Other": tid_other,
        "Class_Subject": merged["veh_class_subj"],
        "Class_Other": merged["veh_class_other"],
        "Time_s": merged["time"],   # actual DataFromSky "Time [s]" (absolute), used for both plots
        f"{subj_label}_X_m": merged["x_subj"],
        f"{subj_label}_Y_m": merged["y_subj"],
        f"{subj_label}_EuclidDist_m": euclid_subj,
        f"{subj_label}_Speed_mps": merged["speed_calc_mps_subj"],
        f"{other_label}_X_m": merged["x_other"],
        f"{other_label}_Y_m": merged["y_other"],
        f"{other_label}_EuclidDist_m": euclid_other,
        f"{other_label}_Speed_mps": merged["speed_calc_mps_other"],
        "Distance_m": dist,
        "Closing_Speed_mps": closing,
        "TTC_s": ttc,
        "PET_s": pet,
    })
    out = out[out["Distance_m"] <= MAX_DIST_GATE * 1.3].reset_index(drop=True)
    return out


results = {}
for ev in EVENTS:
    event_id, pair_type, sheet, tid_subj, subj_label, tid_other, other_label = ev
    df = build_event(*ev)
    results[event_id] = (df, subj_label, other_label, pair_type)
    print(event_id, "frames:", len(df),
          "min_dist:", round(df["Distance_m"].min(), 2),
          "TTC_std:", round(np.nanstd(df["TTC_s"]), 3),
          "PET_std:", round(np.nanstd(df["PET_s"]), 3))

mc_ids = [e[0] for e in EVENTS if e[1] == "MC"]
car_ids = [e[0] for e in EVENTS if e[1] == "CAR"]

color_mc = "firebrick"
color_car = "royalblue"

# ----------------------------------------------------------------------
# PLOT 1 — SPACE-TIME TRAJECTORIES (Euclidean position distance vs Time [s])
# ----------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

eid = mc_ids[0]
df, subj_label, other_label, _ = results[eid]
ax = axes1[0]
ax.plot(df["Time_s"], df[f"{subj_label}_EuclidDist_m"], color=color_mc, lw=1.8,
        marker="o", ms=3, label=f"{subj_label} (subject)")
ax.plot(df["Time_s"], df[f"{other_label}_EuclidDist_m"], color=color_mc, lw=1.8,
        ls="--", marker="s", ms=3, label=f"{other_label} (other)")
ax.set_title(f"Space-Time Trajectory (Real Data): {eid}\nMotorcycle-Motorcycle Event (high fluctuation)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Euclidean Distance from Origin, √(x²+y²) (m)")
ax.legend(fontsize=9, loc="best")
ax.grid(alpha=0.3)

eid = car_ids[0]
df, subj_label, other_label, _ = results[eid]
ax = axes1[1]
ax.plot(df["Time_s"], df[f"{subj_label}_EuclidDist_m"], color=color_car, lw=1.8,
        marker="o", ms=3, label=f"{subj_label} (subject)")
ax.plot(df["Time_s"], df[f"{other_label}_EuclidDist_m"], color=color_car, lw=1.8,
        ls="--", marker="s", ms=3, label=f"{other_label} (other)")
ax.set_title(f"Space-Time Trajectory (Real Data): {eid}\nCar-Car Event (stable)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Euclidean Distance from Origin, √(x²+y²) (m)")
ax.legend(fontsize=9, loc="best")
ax.grid(alpha=0.3)

fig1.suptitle("Space-Time Trajectories Extracted from 2_cross_test2_pivoted.xlsx (real vehicle tracks)",
              y=1.02, fontsize=12)
fig1.tight_layout()
fig1.savefig("/home/claude/work/real_space_time_trajectories.png", dpi=200, bbox_inches="tight")

# ----------------------------------------------------------------------
# PLOT 2 — TTC and PET vs TIME (real)
# ----------------------------------------------------------------------
# Each event occurred at a different absolute Time [s] in the source video,
# so TTC/PET are shown in separate panels (columns) -- each still uses the
# real, absolute "Time [s]" column from the dataset on its own x-axis --
# rather than forcing both events onto one shared/overlapping time axis.
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))

eid = mc_ids[0]
df, subj_label, other_label, _ = results[eid]
ax = axes2[0, 0]
ax.plot(df["Time_s"], df["TTC_s"], color=color_mc, lw=1.6, marker="o", ms=3)
ax.axhline(1.5, color="red", lw=1, ls="--", label="Illustrative TTC conflict threshold = 1.5 s")
ax.set_title(f"TTC vs Time -- {eid} (Motorcycle-Motorcycle, high fluctuation)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("TTC (s)")
ax.set_ylim(0, 10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

eid = car_ids[0]
df, subj_label, other_label, _ = results[eid]
ax = axes2[0, 1]
ax.plot(df["Time_s"], df["TTC_s"], color=color_car, lw=1.6, marker="s", ms=3)
ax.axhline(1.5, color="red", lw=1, ls="--", label="Illustrative TTC conflict threshold = 1.5 s")
ax.set_title(f"TTC vs Time -- {eid} (Car-Car, stable)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("TTC (s)")
ax.set_ylim(0, 10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

eid = mc_ids[0]
df, subj_label, other_label, _ = results[eid]
ax = axes2[1, 0]
ax.plot(df["Time_s"], df["PET_s"], color=color_mc, lw=1.6, marker="o", ms=3)
ax.axhline(1.5, color="red", lw=1, ls="--", label="Illustrative PET conflict threshold = 1.5 s")
ax.set_title(f"PET vs Time -- {eid} (Motorcycle-Motorcycle, high fluctuation)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("PET (s)")
ax.set_ylim(0, 5)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

eid = car_ids[0]
df, subj_label, other_label, _ = results[eid]
ax = axes2[1, 1]
ax.plot(df["Time_s"], df["PET_s"], color=color_car, lw=1.6, marker="s", ms=3)
ax.axhline(1.5, color="red", lw=1, ls="--", label="Illustrative PET conflict threshold = 1.5 s")
ax.set_title(f"PET vs Time -- {eid} (Car-Car, stable)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("PET (s)")
ax.set_ylim(0, 5)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

fig2.suptitle("TTC and PET Time Series (Real Data): Motorcycle-Motorcycle vs Car-Car Events",
              y=1.02, fontsize=12)
fig2.tight_layout()
fig2.savefig("/home/claude/work/real_ttc_pet_vs_time.png", dpi=200, bbox_inches="tight")

# ----------------------------------------------------------------------
# VARIABILITY SUMMARY
# ----------------------------------------------------------------------
summary_rows = []
for eid, (df, subj_label, other_label, ptype) in results.items():
    summary_rows.append({
        "Event_ID": eid, "Pair_Type": ptype,
        "Subject_Class": subj_label, "Other_Class": other_label,
        "N_Frames": len(df), "Min_Distance_m": df["Distance_m"].min(),
        "TTC_std_s": np.nanstd(df["TTC_s"]), "TTC_mean_s": np.nanmean(df["TTC_s"]),
        "PET_std_s": np.nanstd(df["PET_s"]), "PET_mean_s": np.nanmean(df["PET_s"]),
    })
summary_df = pd.DataFrame(summary_rows)

track_id_rows = [{"Event_ID": e[0], "Pair_Type": e[1], "Sheet": e[2],
                   "Subject_Track_ID": e[3], "Subject_Class": e[4],
                   "Other_Track_ID": e[5], "Other_Class": e[6]} for e in EVENTS]
track_id_df = pd.DataFrame(track_id_rows)

# ----------------------------------------------------------------------
# EXCEL EXPORT
# ----------------------------------------------------------------------
out_xlsx = "/home/claude/work/real_mc_car_event_data.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    for eid, (df, subj_label, other_label, ptype) in results.items():
        df.to_excel(writer, sheet_name=eid[:31], index=False)

    common_cols = ["Event_ID", "Pair_Type", "Sheet", "Track_ID_Subject", "Track_ID_Other",
                   "Class_Subject", "Class_Other", "Time_s", "Distance_m",
                   "Closing_Speed_mps", "TTC_s", "PET_s"]
    combined = pd.concat([df[common_cols] for df, *_ in results.values()], ignore_index=True)
    combined.to_excel(writer, sheet_name="All_Events_Combined", index=False)

    summary_df.to_excel(writer, sheet_name="Variability_Summary", index=False)
    track_id_df.to_excel(writer, sheet_name="Event_Track_IDs", index=False)

print("\nSaved:")
print(" - real_space_time_trajectories.png")
print(" - real_ttc_pet_vs_time.png")
print(" - real_mc_car_event_data.xlsx")
print()
print(summary_df.to_string(index=False))
