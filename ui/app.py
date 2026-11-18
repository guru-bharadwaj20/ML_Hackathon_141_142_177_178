"""PathSnap evaluator UI (laptop 2): streamlit run ui/app.py   (set PATHSNAP_API_URL=http://<laptop-1-ip>:8000)"""
import os
import requests
import folium
import streamlit as st
from streamlit_folium import st_folium
from branca.element import Element

API = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap map matching")
st.caption(f"API: {API}")


@st.cache_data(ttl=60)
def get_trips():
    r = requests.get(f"{API}/trips", timeout=20)
    r.raise_for_status()
    return r.json()["trips"]


def make_map(res):
    gps = res["gps"]
    lat, lon = [p[0] for p in gps], [p[1] for p in gps]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[min(lat), min(lon)], [max(lat), max(lon)]])
    fg = folium.FeatureGroup(name="Road network (near track)")
    for ln in res["road_network"]:
        folium.PolyLine(ln, color="#9e9e9e", weight=2, opacity=0.5).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Classifier only (argmax)", show=False)
    for ln in res["classifier_route"]:
        folium.PolyLine(ln, color="#ff7f0e", weight=4, opacity=0.9).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
    for ln in res["predicted_route"]:
        folium.PolyLine(ln, color="#d62728", weight=5, opacity=0.9).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="GPS observations")
    for p in gps:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    items = [("GPS points", "#1f77b4"), ("Predicted route (HMM)", "#d62728"),
             ("Classifier only", "#ff7f0e"), ("Road network", "#9e9e9e")]
    rows = "".join(f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for n, c in items)
    m.get_root().html.add_child(Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
        f'border:1px solid #888;border-radius:4px;font:12px sans-serif"><b>Legend</b>{rows}</div>'))
    return m


try:
    trips = get_trips()
except Exception as e:
    st.error(f"Cannot reach the API at {API}: {e}")
    st.stop()

st.subheader(f"Evaluation trips ({len(trips)})")
trip = st.radio("Select a trip", trips, horizontal=True)
if st.button("Run map matching", type="primary"):
    with st.spinner(f"Sending trip {trip} to the API..."):
        r = requests.post(f"{API}/predict", json={"trip_id": trip}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    st.subheader(f"Trip {res['trip_id']}")
    mt = res.get("metrics")
    c = st.columns(5)
    if mt:
        h, k = mt["hmm"], mt["classifier_only"]
        c[0].metric("Top-1 accuracy (HMM)", f"{h['top1_accuracy']:.2%}",
                    f"{h['top1_accuracy'] - k['top1_accuracy']:+.2%} vs classifier only")
        c[1].metric("Classifier-only accuracy", f"{k['top1_accuracy']:.2%}")
        c[2].metric("Route continuity", f"{h['route_continuity']:.2%}")
        c[3].metric("Route arc precision / recall", f"{h['route_arc_precision']:.2%} / {h['route_arc_recall']:.2%}")
    c[4].metric("API processing time", f"{res['timings_s']['total_s']:.1f} s")
    st.caption(f"{res['n_points']} GPS points, {res['matched_points']} matched, {res['chain_breaks']} HMM chain breaks. "
               f"Model: {res['model']}, HMM beta = {res['hmm_beta_m']:g} m. Accuracy is computed on the API server.")
    st_folium(make_map(res), height=620, use_container_width=True, returned_objects=[])
    with st.expander("Stage timings (s)"):
        st.json(res["timings_s"])
