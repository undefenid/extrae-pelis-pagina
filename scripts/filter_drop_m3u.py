import os
import json
import re
import requests

INPUT_JSON = os.environ.get("INPUT_JSON", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "filtered").strip()
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "peliculas_no_m3u.json").strip()
ONLY_MOVIES = (os.environ.get("ONLY_MOVIES", "true").strip().lower() == "true")

if not INPUT_JSON:
    raise SystemExit("Falta INPUT_JSON")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# termina en .m3u, permitiendo ?query o #hash; y EXCLUYE .m3u8
M3U_END_RE = re.compile(r"\.m3u(?=$|[?#])", re.IGNORECASE)

def load_json(path_or_url: str):
    if path_or_url.lower().startswith("http://") or path_or_url.lower().startswith("https://"):
        r = requests.get(path_or_url, timeout=120)
        r.raise_for_status()
        return r.json()
    with open(path_or_url, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json(INPUT_JSON)

if not isinstance(data, list):
    raise SystemExit("El JSON de entrada debe ser una lista de records [{name, samples}]")

stats = {
    "input_records": len(data),
    "input_samples": 0,
    "dropped_m3u": 0,
    "kept_samples": 0,
    "empty_groups_removed": 0,
    "only_movies": ONLY_MOVIES,
}

out_records = []

for rec in data:
    if not isinstance(rec, dict):
        continue
    name = rec.get("name", "")
    samples = rec.get("samples") or []
    if not isinstance(samples, list):
        samples = []

    new_samples = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        stats["input_samples"] += 1

        if ONLY_MOVIES:
            t = (s.get("type") or "").strip().upper()
            if t and t != "PELICULA":
                # no se filtra si no es pelicula
                new_samples.append(s)
                stats["kept_samples"] += 1
                continue

        url = (s.get("url") or "").strip()
        if url and M3U_END_RE.search(url):
            # ojo: esto NO matchea .m3u8
            stats["dropped_m3u"] += 1
            continue

        new_samples.append(s)
        stats["kept_samples"] += 1

    # si un grupo queda vacío, lo omitimos
    if new_samples:
        out_records.append({"name": name, "samples": new_samples})
    else:
        stats["empty_groups_removed"] += 1

out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_records, f, indent=2, ensure_ascii=False)

report_path = os.path.join(OUTPUT_DIR, "filter_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(json.dumps(stats, indent=2, ensure_ascii=False))
print(f"Salida: {out_path}")
