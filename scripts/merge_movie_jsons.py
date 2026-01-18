import os
import re
import json
import time
import unicodedata
import requests
from collections import defaultdict

SOURCES_URL = os.environ.get("SOURCES_URL", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "merged").strip()
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "peliculas_merged.json").strip()
ONLY_MOVIES = (os.environ.get("ONLY_MOVIES", "true").strip().lower() == "true")

if not SOURCES_URL:
    raise SystemExit("Falta SOURCES_URL")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utils
# -------------------------
def strip_accents(s: str) -> str:
    if not s:
        return ""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm_group_key(name: str) -> str:
    return strip_accents((name or "").strip().lower())

YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

def extract_year_int(s: str) -> int:
    """Devuelve año int si hay, si no 0."""
    m = YEAR_RE.search((s or "").strip())
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0

def ensure_sample_schema(sample: dict) -> dict:
    # asegura todos los campos existan, sin borrar nada
    sample = dict(sample or {})
    sample.setdefault("name", sample.get("name","") or "")
    sample.setdefault("url", sample.get("url","") or "")
    sample.setdefault("icono", sample.get("icono","") or "")
    sample.setdefault("iconoHorizontal", sample.get("iconoHorizontal","") or "")
    sample.setdefault("iconpng", sample.get("iconpng","") or "")
    sample.setdefault("type", sample.get("type","") or "PELICULA")
    sample.setdefault("descripcion", sample.get("descripcion","") or "")
    sample.setdefault("anio", sample.get("anio","") or "")
    sample.setdefault("genero", sample.get("genero","") or "")
    sample.setdefault("duracion", sample.get("duracion","") or "")
    return sample

def fetch_text_or_local(path_or_url: str) -> str:
    if path_or_url.lower().startswith("http://") or path_or_url.lower().startswith("https://"):
        r = requests.get(path_or_url, timeout=90)
        r.raise_for_status()
        return r.text
    # local file in repo
    with open(path_or_url, "r", encoding="utf-8") as f:
        return f.read()

def fetch_json(url: str):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()

def parse_sources_list(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []

    # JSON array o JSON object {json_urls:[...]}
    if raw.startswith("{") or raw.startswith("["):
        try:
            js = json.loads(raw)
            if isinstance(js, list):
                return [str(x).strip() for x in js if str(x).strip()]
            if isinstance(js, dict) and "json_urls" in js and isinstance(js["json_urls"], list):
                return [str(x).strip() for x in js["json_urls"] if str(x).strip()]
        except Exception:
            pass

    # TXT: 1 url por línea, # comentarios
    urls = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        urls.append(ln)
    return urls

# -------------------------
# Load list of JSON URLs
# -------------------------
sources_raw = fetch_text_or_local(SOURCES_URL)
json_urls = parse_sources_list(sources_raw)

if not json_urls:
    raise SystemExit("No se encontraron URLs en SOURCES_URL (lista vacia)")

# Dedup de lista de fuentes
json_urls = list(dict.fromkeys(json_urls))

print(f"Fuentes detectadas: {len(json_urls)}")

# -------------------------
# Merge logic
# -------------------------
# groups_map: norm_key -> {"display": originalDisplay, "samples": []}
groups_map = {}
seen_video_urls = set()

stats = {
    "sources": len(json_urls),
    "records_read": 0,
    "samples_read": 0,
    "samples_kept": 0,
    "samples_skipped_empty_url": 0,
    "samples_skipped_non_movie": 0,
    "samples_dedup_dropped": 0,
    "groups_out": 0,
}

def get_or_create_group(group_name: str):
    g = (group_name or "").strip()
    if not g:
        g = "Sin información"
    nk = norm_group_key(g)
    if nk not in groups_map:
        groups_map[nk] = {"display": g, "samples": []}
    return groups_map[nk]

for idx, url in enumerate(json_urls, start=1):
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"[WARN] No se pudo descargar/parsear JSON ({url}): {e}")
        continue

    if not isinstance(data, list):
        print(f"[WARN] JSON no es lista [{url}] -> ignorado")
        continue

    for record in data:
        stats["records_read"] += 1
        group_name = ""
        if isinstance(record, dict):
            group_name = (record.get("name") or "").strip()
            samples = record.get("samples") or []
        else:
            continue

        if not isinstance(samples, list):
            continue

        grp = get_or_create_group(group_name)

        for s in samples:
            stats["samples_read"] += 1
            if not isinstance(s, dict):
                continue

            s = ensure_sample_schema(s)

            if ONLY_MOVIES:
                t = (s.get("type") or "").strip().upper()
                if t and t != "PELICULA":
                    stats["samples_skipped_non_movie"] += 1
                    continue

            video_url = (s.get("url") or "").strip()
            if not video_url:
                stats["samples_skipped_empty_url"] += 1
                continue

            if video_url in seen_video_urls:
                stats["samples_dedup_dropped"] += 1
                continue

            seen_video_urls.add(video_url)
            grp["samples"].append(s)
            stats["samples_kept"] += 1

# -------------------------
# Sort samples by year desc (recent first), then name
# -------------------------
def sample_sort_key(s: dict):
    y = extract_year_int(s.get("anio") or "")
    if y <= 0:
        y = extract_year_int(s.get("name") or "")
    missing = 1 if y <= 0 else 0
    # missing last, then year desc, then name asc
    return (missing, -y, strip_accents((s.get("name") or "").strip().lower()))

for g in groups_map.values():
    g["samples"].sort(key=sample_sort_key)

# -------------------------
# Sort groups alphabetically, with "Sin información" at end
# -------------------------
def group_sort_key(display_name: str):
    dn = (display_name or "").strip()
    dn_norm = norm_group_key(dn)
    if dn_norm in ("sin informacion", "sin información"):
        return (1, dn_norm)
    return (0, dn_norm)

groups_sorted = sorted(groups_map.values(), key=lambda g: group_sort_key(g["display"]))

# Build final records list
final_records = [{"name": g["display"], "samples": g["samples"]} for g in groups_sorted]
stats["groups_out"] = len(final_records)

# Ensure Sin información LAST (por si venía con otra capitalización)
# (ya lo hace el sort, esto es solo reforzar)
# -------------------------
# Save outputs
# -------------------------
out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_records, f, indent=2, ensure_ascii=False)

# Report + duplicates info (solo conteos)
report_path = os.path.join(OUTPUT_DIR, "merge_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(json.dumps(stats, indent=2, ensure_ascii=False))
print(f"Salida: {out_path}")
