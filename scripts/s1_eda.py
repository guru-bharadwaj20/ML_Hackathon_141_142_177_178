"""Stage 1: raw-data inspection and exploratory data analysis.

Partitions used: train (80 trips) and student_test (10 trips). Evaluator trips are never touched.
Outputs: results/eda/*.csv, results/plots/eda_*.png, maps/eda_*.html, results/eda/data_quality_notes.md
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.data_io import list_trips, load_trip
from src.geo import LocalProjection, point_segment
from src.candidates import ArcIndex
from src.maps import trip_map

OUT = C.RESULTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def raw_counts(path, ncols):
    """Count rows, rows with wrong column count and non-numeric/NaN values in a raw file."""
    n = bad_cols = nonnum = 0
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            n += 1
            if len(parts) != ncols:
                bad_cols += 1
                continue
            try:
                vals = [float(x) for x in parts]
                if any(v != v for v in vals):
                    nonnum += 1
            except ValueError:
                nonnum += 1
    return n, bad_cols, nonnum


def inspect(partition, pdir, tid):
    d = pdir / tid
    row = {"partition": partition, "trip_id": tid}
    for ext, nc in (("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)):
        f = d / f"{tid}.{ext}"
        row[f"has_{ext}"] = f.exists()
        row[f"{ext}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else np.nan
        if f.exists():
            n, bc, nn = raw_counts(f, nc)
            row[f"{ext}_rows"] = n
            row[f"{ext}_badcols"] = bc
            row[f"{ext}_nonnumeric"] = nn
    if not (row["has_track"] and row["has_nodes"] and row["has_arcs"]):
        return row, None
    T = load_trip(pdir, tid)
    tr, nodes, arcs, route = T["track"], T["nodes"], T["arcs"], T["route"]
    t = tr[:, 2]
    dt = np.diff(t)
    row.update({
        "n_points": len(tr),
        "duration_s": t[-1] - t[0],
        "invalid_coords": int((~np.isfinite(tr[:, :2]).all(1) | (np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
        "non_increasing_t": int((dt <= 0).sum()),
        "dt_median": float(np.median(dt)),
        "dt_frac_1s": float((dt == 1).mean()),
        "dt_max": float(dt.max()),
        "n_gaps_gt10s": int((dt > 10).sum()),
        "lon_center": float(np.median(tr[:, 0])), "lat_center": float(np.median(tr[:, 1])),
    })
    proj = LocalProjection.from_track(tr)
    xy = proj.to_xy(tr[:, :2])
    nxy = proj.to_xy(nodes)
    row["extent_x_km"] = float(np.ptp(xy[:, 0]) / 1000)
    row["extent_y_km"] = float(np.ptp(xy[:, 1]) / 1000)
    row["path_len_km"] = float(np.hypot(*np.diff(xy, axis=0).T).sum() / 1000)
    step = np.hypot(*np.diff(xy, axis=0).T)
    sp = step / np.maximum(dt, 1e-9)
    row["speed_p95_ms"] = float(np.percentile(sp, 95))
    row["n_speed_gt60ms"] = int((sp > 60).sum())
    row["n_nodes"] = len(nodes)
    row["n_arcs"] = len(arcs)
    row["arc_ids_valid"] = bool((arcs >= 0).all() and (arcs < len(nodes)).all())
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(map(tuple, arcs))
    deg = np.bincount(arcs.ravel(), minlength=len(nodes))
    row["isolated_nodes"] = int((deg == 0).sum())
    wcc = [len(c) for c in nx.weakly_connected_components(G)]
    row["n_weak_components"] = len(wcc)
    row["largest_wcc_frac"] = max(wcc) / len(nodes)
    row["n_strong_components"] = nx.number_strongly_connected_components(G)
    key = set(map(tuple, arcs))
    row["two_way_frac"] = float(np.mean([(b, a) in key for a, b in arcs]))
    row["self_loops"] = int((arcs[:, 0] == arcs[:, 1]).sum())
    row["duplicate_arcs"] = len(arcs) - len(key)
    L = np.hypot(*(nxy[arcs[:, 1]] - nxy[arcs[:, 0]]).T)
    row["arc_len_median_m"] = float(np.median(L))
    if route is not None:
        row["n_route_arcs"] = len(route)
        row["route_ids_valid"] = bool((route >= 0).all() and (route < len(arcs)).all())
        row["route_contiguous_frac"] = float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]))
        row["route_repeated_arcs"] = len(route) - len(set(route.tolist()))
        row["route_len_km"] = float(L[route].sum() / 1000)
        A, B = nxy[arcs[route, 0]], nxy[arcs[route, 1]]
        sub = xy[:: max(1, len(xy) // 2000)]
        dmin = np.array([point_segment(x, y, A[:, 0], A[:, 1], B[:, 0], B[:, 1])[0].min() for x, y in sub])
        row["gps_route_dist_median_m"] = float(np.median(dmin))
        row["gps_route_dist_p95_m"] = float(np.percentile(dmin, 95))
        row["gps_far_from_route_frac"] = float((dmin > 50).mean())
    extra = {"dt": dt, "xy": xy, "dmin": dmin if route is not None else None, "speed": sp}
    return row, extra


def main():
    rows, dts, dmins = [], [], []
    for part, pdir in (("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)):
        for tid in list_trips(pdir):
            row, ex = inspect(part, pdir, tid)
            rows.append(row)
            if ex is not None:
                dts.append(pd.DataFrame({"partition": part, "trip_id": tid, "dt": ex["dt"]}))
                if ex["dmin"] is not None:
                    dmins.append(pd.DataFrame({"partition": part, "trip_id": tid, "d": ex["dmin"]}))
            print(part, tid, row.get("n_points"), flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    dts = pd.concat(dts)
    dmins = pd.concat(dmins)

    # ---- missing-value analysis table
    miss = []
    for part, g in inv.groupby("partition"):
        for ext in ("track", "nodes", "arcs", "route"):
            miss.append({"partition": part, "file": ext, "trips": len(g),
                         "trips_missing_file": int((~g[f"has_{ext}"]).sum()),
                         "rows_wrong_column_count": int(g.get(f"{ext}_badcols", pd.Series(0)).fillna(0).sum()),
                         "rows_non_numeric_or_nan": int(g.get(f"{ext}_nonnumeric", pd.Series(0)).fillna(0).sum())})
    miss = pd.DataFrame(miss)
    miss.to_csv(OUT / "missing_values.csv", index=False)

    # ---- partition summary
    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), gps_points=("n_points", "sum"), median_points=("n_points", "median"),
        min_points=("n_points", "min"), max_points=("n_points", "max"),
        median_nodes=("n_nodes", "median"), median_arcs=("n_arcs", "median"),
        median_route_arcs=("n_route_arcs", "median"), trips_with_gap_gt10s=("n_gaps_gt10s", lambda s: int((s > 0).sum())),
        median_two_way_frac=("two_way_frac", "median"))
    summ.to_csv(OUT / "partition_summary.csv")
    dt_all = dts["dt"].to_numpy()
    sampling = {"frac_dt_eq_1s": float((dt_all == 1).mean()), "frac_dt_le_2s": float((dt_all <= 2).mean()),
                "n_gaps_gt10s": int((dt_all > 10).sum()), "n_gaps_gt60s": int((dt_all > 60).sum()),
                "max_gap_s": float(dt_all.max())}
    json.dump(sampling, open(OUT / "sampling_summary.json", "w"), indent=2)
    print(summ.T); print(miss); print(sampling)

    # ---- plots
    P = C.PLOTS_DIR
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for part, g in inv.groupby("partition"):
        ax[0].hist(g["n_points"].dropna(), bins=np.logspace(2, 4.5, 25), alpha=0.6, label=part)
    ax[0].set_xscale("log"); ax[0].set_xlabel("GPS points per trip"); ax[0].set_ylabel("trips"); ax[0].legend()
    ax[0].set_title("Trajectory length (train + student_test)")
    ax[1].hist(np.clip(dt_all, 0, 30), bins=30, log=True)
    ax[1].set_xlabel("time between consecutive GPS points (s, clipped at 30)"); ax[1].set_ylabel("count (log)")
    ax[1].set_title("Sampling interval")
    fig.tight_layout(); fig.savefig(P / "eda_points_and_sampling.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    g = inv.dropna(subset=["n_points"]).sort_values("dt_max", ascending=False).head(15)
    ax.barh(g["trip_id"] + " (" + g["partition"].str[:5] + ")", g["dt_max"])
    ax.set_xscale("log"); ax.set_xlabel("largest gap between GPS points (s, log)"); ax.invert_yaxis()
    ax.set_title("Trips with the largest GPS gaps")
    fig.tight_layout(); fig.savefig(P / "eda_largest_gaps.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(inv["n_nodes"], inv["n_arcs"], c=(inv["partition"] == "train").map({True: "C0", False: "C1"}), s=15)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs (road segments)")
    ax[0].set_title("Road-network size per trip (blue=train, orange=student_test)")
    ax[1].hist(inv["largest_wcc_frac"].dropna(), bins=20)
    ax[1].set_xlabel("fraction of nodes in largest weakly-connected component"); ax[1].set_ylabel("trips")
    ax[1].set_title("Road-network connectivity")
    fig.tight_layout(); fig.savefig(P / "eda_network_size_connectivity.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(dmins["d"], 0, 100), bins=50, log=True)
    ax.axvline(C.CAND_RADIUS_M, color="r", ls="--", label=f"candidate radius {C.CAND_RADIUS_M:.0f} m")
    ax.set_xlabel("distance GPS point -> nearest reference-route arc (m, clipped at 100)"); ax.set_ylabel("points (log)")
    ax.set_title("GPS noise relative to the reference route (train + student_test)"); ax.legend()
    fig.tight_layout(); fig.savefig(P / "eda_gps_to_route_distance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for part, g in inv.groupby("partition"):
        ax.scatter(g["lon_center"], g["lat_center"], label=part, s=20)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title("Trip locations (median GPS coordinate) - trips span several continents")
    fig.tight_layout(); fig.savefig(P / "eda_trip_locations.png", dpi=130); plt.close(fig)

    # GPS trajectory plots + road network for 4 training trips
    show = ["00000001", "00000016", "00000044", "00000081"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    for ax, tid in zip(axs.ravel(), show):
        T = load_trip(C.TRAIN_DIR, tid)
        proj = LocalProjection.from_track(T["track"]); xy = proj.to_xy(T["track"][:, :2]); nxy = proj.to_xy(T["nodes"])
        idx = ArcIndex(nxy, T["arcs"])
        _, near = idx.within(xy[::3], 150)
        near = np.unique(near)
        for a in near:
            ax.plot([idx.A[a, 0], idx.B[a, 0]], [idx.A[a, 1], idx.B[a, 1]], color="0.75", lw=0.8)
        for a in T["route"]:
            ax.plot([idx.A[a, 0], idx.B[a, 0]], [idx.A[a, 1], idx.B[a, 1]], color="g", lw=2.5, alpha=0.7)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="C0", zorder=3)
        ax.set_aspect("equal"); ax.set_title(f"train {tid}: GPS (blue), reference route (green), network (grey)", fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(P / "eda_trajectories_network.png", dpi=130); plt.close(fig)

    # ---- Folium maps (train partition)
    for tid in ("00000001", "00000054"):
        T = load_trip(C.TRAIN_DIR, tid)
        proj = LocalProjection.from_track(T["track"]); xy = proj.to_xy(T["track"][:, :2])
        idx = ArcIndex(proj.to_xy(T["nodes"]), T["arcs"])
        _, near = idx.within(xy[::2], 100)
        m = trip_map(T["track"], T["nodes"], T["arcs"], network_arc_ids=np.unique(near),
                     reference_arc_ids=T["route"], title=f"EDA - train trip {tid}")
        m.save(str(C.MAPS_DIR / f"eda_train_{tid}.html"))

    # ---- data-quality notes
    notes = ["# Data-quality notes (generated by scripts/s1_eda.py)", ""]
    for _, r in inv.iterrows():
        issues = []
        for ext in ("track", "nodes", "arcs", "route"):
            if not r[f"has_{ext}"]:
                issues.append(f"missing .{ext} file")
        if r.get("n_gaps_gt10s", 0) and r["n_gaps_gt10s"] > 0:
            issues.append(f"{int(r['n_gaps_gt10s'])} gaps >10 s (max {r['dt_max']:.0f} s)")
        if r.get("gps_far_from_route_frac", 0) and r["gps_far_from_route_frac"] > 0.05:
            issues.append(f"{100*r['gps_far_from_route_frac']:.0f}% of GPS points >50 m from reference route")
        if r.get("n_speed_gt60ms", 0) and r["n_speed_gt60ms"] > 0:
            issues.append(f"{int(r['n_speed_gt60ms'])} implausible jumps (>60 m/s)")
        if r.get("route_contiguous_frac", 1) < 1:
            issues.append(f"route not fully contiguous ({100*r['route_contiguous_frac']:.1f}%)")
        if issues:
            notes.append(f"- {r['partition']} {r['trip_id']}: " + "; ".join(issues))
    (OUT / "data_quality_notes.md").write_text("\n".join(notes))
    print("\n".join(notes[:60]))


if __name__ == "__main__":
    main()
