import os
import re
import json
import unicodedata
import requests
from urllib.parse import urlparse
from collections import defaultdict, OrderedDict

MERGED_JSON_URL = os.environ.get("MERGED_JSON_URL", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "duplicates").strip()
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "peliculas_duplicadas.json").strip()
SIM_THR = float((os.environ.get("SIMILARITY_THRESHOLD") or "0.78").strip())
MIN_CLUSTER_SIZE = int((os.environ.get("MIN_CLUSTER_SIZE") or "2").strip())

if not MERGED_JSON_URL:
    raise SystemExit("Falta MERGED_JSON_URL")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# IO helpers
# -------------------------
def fetch_text_or_local(path_or_url: str) -> str:
    if path_or_url.lower().startswith("http://") or path_or_url.lower().startswith("https://"):
        r = requests.get(path_or_url, timeout=120)
        r.raise_for_status()
        return r.text
    with open(path_or_url, "r", encoding="utf-8") as f:
        return f.read()

raw = fetch_text_or_local(MERGED_JSON_URL)
data = json.loads(raw)

if not isinstance(data, list):
    raise SystemExit("El JSON general debe ser una lista: [{name, samples:[...]}]")

# -------------------------
# Normalizers
# -------------------------
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
QUALITY_RE = re.compile(
    r"\b(480p|720p|1080p|2160p|4k|hdr|sdr|webrip|web\-dl|bluray|brrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|ac3|dts|latino|castellano|subtitulado|sub|dual|multi|esp|eng|vose)\b",
    re.IGNORECASE
)

def strip_accents(s: str) -> str:
    if not s:
        return ""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm_spaces(s: str) -> str:
    return re.sub(r"\s{2,}", " ", (s or "").strip())

def normalize_title_strict(title: str) -> str:
    """
    Para duplicados IDENTICOS (más estricto):
    - lower, sin acentos
    - quita puntuación
    - quita espacios dobles
    """
    t = strip_accents((title or "").lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = norm_spaces(t)
    return t

def base_key(title: str) -> str:
    """
    Para duplicados SIMILARES:
    - quita calidad/idioma
    - quita año
    - quita signos y separadores
    - normaliza espacios
    """
    t = strip_accents((title or "").lower())
    t = QUALITY_RE.sub(" ", t)
    t = YEAR_RE.sub(" ", t)
    t = re.sub(r"[\|\._\-]+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = norm_spaces(t)
    # quita números sueltos al final (a veces son tags)
    t = re.sub(r"\s+\d{1,3}\s*$", "", t).strip()
    return t

def token_set(s: str):
    return set([x for x in s.split(" ") if x])

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return (inter / uni) if uni else 0.0

# -------------------------
# Flatten samples
# -------------------------
items = []  # each: {group, sample, title, url, strict, base, tokens}
for rec in data:
    if not isinstance(rec, dict):
        continue
    group = (rec.get("name") or "").strip()
    for s in (rec.get("samples") or []):
        if not isinstance(s, dict):
            continue
        t = (s.get("type") or "").upper().strip()
        if t and t != "PELICULA":
            continue
        title = (s.get("name") or "").strip()
        url = (s.get("url") or "").strip()
        if not title:
            continue
        strict = normalize_title_strict(title)
        bk = base_key(title)
        tok = token_set(bk)
        items.append({
            "group": group,
            "sample": s,
            "title": title,
            "url": url,
            "strict": strict,
            "base": bk,
            "tokens": tok,
        })

# -------------------------
# 1) Identicos (strict)
# -------------------------
by_strict = defaultdict(list)
for it in items:
    by_strict[it["strict"]].append(it)

identical_clusters = [v for v in by_strict.values() if len(v) >= MIN_CLUSTER_SIZE]

# -------------------------
# 2) Similares (base-key buckets + jaccard)
#    Para hacerlo rápido, primero bucket por base_key exacto.
#    Luego, dentro del bucket (si es grande), agrupamos por similitud.
# -------------------------
by_base = defaultdict(list)
for it in items:
    if it["base"]:
        by_base[it["base"]].append(it)

base_exact_clusters = [v for v in by_base.values() if len(v) >= MIN_CLUSTER_SIZE]

# Ahora clusters “fuzzy” dentro de los buckets exactos (opcional)
# En la práctica, muchos casos ya se detectan con base exacta.
# Para capturar variaciones pequeñas ("ET" vs "E T" etc.) hacemos un paso más:
# Bucket por primera letra + longitud aproximada para reducir comparaciones.
fuzzy_bucket = defaultdict(list)
for it in items:
    b = it["base"]
    if not b:
        continue
    key = (b[:1], len(b)//5)  # agrupación grosera
    fuzzy_bucket[key].append(it)

def build_fuzzy_clusters(bucket_items):
    clusters = []
    used = set()
    for i, it in enumerate(bucket_items):
        if i in used:
            continue
        c = [it]
        used.add(i)
        for j in range(i+1, len(bucket_items)):
            if j in used:
                continue
            other = bucket_items[j]
            # si base exacta igual, ya está cubierto, pero no molesta
            sim = jaccard(it["tokens"], other["tokens"])
            if sim >= SIM_THR:
                c.append(other)
                used.add(j)
        if len(c) >= MIN_CLUSTER_SIZE:
            clusters.append(c)
    return clusters

fuzzy_clusters = []
for _, bitems in fuzzy_bucket.items():
    if len(bitems) < MIN_CLUSTER_SIZE:
        continue
    # evitar O(n^2) gigante
    if len(bitems) > 1200:
        # si es enorme, no hacemos fuzzy, solo base exacta ya cubre bastante
        continue
    fuzzy_clusters.extend(build_fuzzy_clusters(bitems))

# -------------------------
# Merge clusters (identicos + base_exact + fuzzy) sin duplicar items
# -------------------------
# Usamos "fingerprint" por url si existe, si no por (title, group).
def fp(it):
    if it["url"]:
        return ("url", it["url"])
    return ("tg", it["strict"], it["group"])

def cluster_to_fps(cluster):
    return set(fp(x) for x in cluster)

all_clusters = []
seen_clusters = []

def add_cluster(cluster, kind):
    fps = cluster_to_fps(cluster)
    if len(fps) < MIN_CLUSTER_SIZE:
        return
    # dedupe de clusters muy solapados
    for prev in seen_clusters:
        inter = len(fps & prev)
        if inter >= max(2, int(0.7 * min(len(fps), len(prev)))):  # 70% overlap
            return
    seen_clusters.append(fps)
    all_clusters.append((kind, cluster))

for c in identical_clusters:
    add_cluster(c, "identico")
for c in base_exact_clusters:
    add_cluster(c, "base_exact")
for c in fuzzy_clusters:
    add_cluster(c, "similar")

# -------------------------
# Output: JSON con solo duplicados
# Estructura:
# [
#   { "name": "Duplicados - <kind> - <clave>", "samples": [ ...samples... ] }
# ]
# -------------------------
def cluster_title(kind, cluster):
    # elegir un título representativo
    titles = [x["title"] for x in cluster if x.get("title")]
    rep = titles[0] if titles else "Sin titulo"
    return f"Duplicados - {kind} - {rep}"

out_records = []
total_samples_out = 0

# ordenar clusters por nombre representativo
all_clusters.sort(key=lambda kc: normalize_title_strict(cluster_title(kc[0], kc[1])))

for kind, cluster in all_clusters:
    # orden interno: por año desc y luego título
    def year_int(it):
        y = it["sample"].get("anio") or ""
        m = YEAR_RE.search(str(y))
        return int(m.group(1)) if m else 0

    cluster_sorted = sorted(
        cluster,
        key=lambda it: (-year_int(it), normalize_title_strict(it["title"]))
    )

    samples = [it["sample"] for it in cluster_sorted]
    total_samples_out += len(samples)
    out_records.append({"name": cluster_title(kind, cluster_sorted), "samples": samples})

out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_records, f, indent=2, ensure_ascii=False)

report = {
    "source": MERGED_JSON_URL,
    "items_scanned": len(items),
    "clusters_out": len(out_records),
    "samples_out_total": total_samples_out,
    "similarity_threshold": SIM_THR,
    "min_cluster_size": MIN_CLUSTER_SIZE,
    "notes": [
        "identico: titulo normalizado estricto igual",
        "base_exact: base_key igual (quita año/calidad/idioma/signos)",
        "similar: jaccard(tokens(base_key)) >= threshold dentro de buckets"
    ]
}

with open(os.path.join(OUTPUT_DIR, "duplicates_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"Salida: {out_path}")
