"""Folium map generation (used by EDA, HMM analysis, the report and the UI)."""
import numpy as np
import folium
from branca.element import Element

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "classifier": "#ff7f0e", "hmm": "#d62728", "unmatched": "#000000"}


def _arc_lines(nodes_lonlat, arcs, arc_ids):
    a = arcs[np.asarray(arc_ids, dtype=int)]
    p, q = nodes_lonlat[a[:, 0]], nodes_lonlat[a[:, 1]]
    return [[[p[i, 1], p[i, 0]], [q[i, 1], q[i, 0]]] for i in range(len(a))]


def _legend(m, items):
    rows = "".join(f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in items)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px sans-serif">'
            f'<b>Legend</b>{rows}</div>')
    m.get_root().html.add_child(Element(html))


def trip_map(track, nodes_lonlat, arcs, network_arc_ids=None, reference_arc_ids=None,
             predicted=None, title=None, max_gps=3000):
    lat, lon = track[:, 1], track[:, 0]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    legend = []

    if network_arc_ids is not None and len(network_arc_ids):
        fg = folium.FeatureGroup(name="Road network (near track)", show=True)
        folium.PolyLine(_arc_lines(nodes_lonlat, arcs, network_arc_ids), color=COLORS["network"],
                        weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m); legend.append(("Road network", COLORS["network"]))

    if reference_arc_ids is not None and len(reference_arc_ids):
        fg = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        folium.PolyLine(_arc_lines(nodes_lonlat, arcs, reference_arc_ids), color=COLORS["reference"],
                        weight=7, opacity=0.6).add_to(fg)
        fg.add_to(m); legend.append(("Reference route", COLORS["reference"]))

    for name, ids in (predicted or {}).items():
        col = COLORS["hmm"] if "hmm" in name.lower() else COLORS["classifier"]
        fg = folium.FeatureGroup(name=f"Predicted: {name}", show=True)
        ids = [a for a in ids if a >= 0]
        if ids:
            folium.PolyLine(_arc_lines(nodes_lonlat, arcs, ids), color=col, weight=4, opacity=0.9).add_to(fg)
        fg.add_to(m); legend.append((f"Predicted ({name})", col))

    fg = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(track) // max_gps)
    for i in range(0, len(track), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"pt {i}  t={track[i, 2]:.0f}").add_to(fg)
    fg.add_to(m); legend.append(("GPS points", COLORS["gps"]))
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, legend)
    if title:
        m.get_root().html.add_child(Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 8px;'
            f'border:1px solid #888;font:bold 13px sans-serif">{title}</div>'))
    return m
