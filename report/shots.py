"""Screenshots for the report: Folium maps and the Streamlit UI (API and UI must be running locally)."""
import re
import sys
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.data_io import load_route
from src.labels import align_route
from src.pipeline import Matcher

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TRIP = sys.argv[1] if len(sys.argv) > 1 else "00000087"
ZOOM_TRIP = "00000076"


def fix_location(tid):
    r = Matcher().match(C.STUDENT_TEST_DIR, tid)
    route = load_route(C.STUDENT_TEST_DIR, tid)
    idx = r["prep"]["index"]
    k, _ = align_route(r["prep"]["xy"], idx.A[route], idx.B[route])
    ref = route[k]
    fixed = np.flatnonzero((r["pred_clf"] != ref) & (r["pred_hmm"] == ref))
    xy = r["prep"]["xy"][fixed]
    best = fixed[np.argmax([(np.hypot(*(xy - q).T) < 150).sum() for q in xy])]
    lon, lat = r["prep"]["track"][best, :2]
    return float(lat), float(lon)


def add_osm(page):
    m = re.search(r"var (map_\w+) = L\.map", page.content()).group(1)
    page.evaluate(f"L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', "
                  f"{{maxZoom: 19, attribution: 'OpenStreetMap'}}).addTo({m}).bringToBack()")
    return m


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    m = None
    for src in ["eda_train_00000001.html", f"student_test_{ZOOM_TRIP}.html"]:
        pg.goto((ROOT / "maps" / src).as_uri(), wait_until="networkidle")
        m = add_osm(pg)
        pg.wait_for_load_state("networkidle")
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / ("map_" + src.replace(".html", ".png"))))
    lat, lon = fix_location(ZOOM_TRIP)
    pg.evaluate(f"{m}.setView([{lat}, {lon}], 17)")
    pg.wait_for_load_state("networkidle")
    pg.wait_for_timeout(3000)
    pg.screenshot(path=str(OUT / f"map_student_test_{ZOOM_TRIP}_zoom.png"))
    print("zoom at", lat, lon)

    pg = b.new_page(viewport={"width": 1500, "height": 1150})
    pg.goto("http://127.0.0.1:8501", wait_until="networkidle")
    pg.get_by_text(TRIP, exact=True).first.click()
    pg.get_by_role("button", name="Run map matching").click()
    pg.get_by_text("Top-1 accuracy (HMM)").wait_for(timeout=60000)
    pg.wait_for_timeout(4000)
    for fr in pg.frames:
        try:
            add_osm(fr)
        except Exception:
            pass
    pg.wait_for_timeout(5000)
    pg.screenshot(path=str(OUT / f"ui_trip_{TRIP}.png"), full_page=True)
    b.close()
