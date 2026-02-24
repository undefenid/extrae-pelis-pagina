import os
import re
import json
import time
import requests
import unicodedata
import difflib
from collections import defaultdict

API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
JSON_PATH = os.environ.get("JSON_PATH", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "tmdb_fix").strip()
LANGUAGE = (os.environ.get("LANGUAGE") or "es-MX").strip()
FALLBACK_LANGUAGE = (os.environ.get("FALLBACK_LANGUAGE") or "en-US").strip()
ONLY_FILL_EMPTY = (os.environ.get("ONLY_FILL_EMPTY", "true").strip().lower() == "true")
MIN_DELAY_MS = int((os.environ.get("MIN_DELAY_MS") or "150").strip())

if not API_KEY:
    raise SystemExit("Falta secrets.TMDB_API_KEY")
if not JSON_PATH:
    raise SystemExit("Falta JSON_PATH")
if not os.path.exists(JSON_PATH):
    raise SystemExit(f"No existe el archivo: {JSON_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/original"
TMDB_IMG_HOST = "image.tmdb.org"

# -----------------------
# Rate-limit soft throttle
# -----------------------
_last_call = 0.0
def _throttle():
    global _last_call
    now = time.time()
    wait = (MIN_DELAY_MS / 1000.0) - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def tmdb_get(path, params=None, retries=6):
    """
    🔥 Cambio clave:
    - NO levanta excepción si TMDB falla (500, 502, 503, 504, 429, etc) tras reintentos.
    - Devuelve dict en éxito, o None en fallo.
    """
    url = TMDB_BASE + path
    params = dict(params or {})
    params["api_key"] = API_KEY

    last_exc = None
    last_resp = None

    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, params=params, timeout=30)
            last_resp = r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.0 + attempt)
            continue

        # rate limit
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "2") or "2")
            time.sleep(retry_after + (attempt * 1.5))
            continue

        # server errors: reintentar
        if r.status_code in (500, 502, 503, 504):
            time.sleep(1.0 + attempt)
            continue

        # otros 4xx: no reintentar infinito, cortar suave
        if 400 <= r.status_code < 500:
            return None

        # éxito
        try:
            r.raise_for_status()
        except requests.HTTPError:
            return None

        try:
            return r.json()
        except Exception:
            return None

    # agotó reintentos: no tumbar el workflow
    return None

# -----------------------
# Utilidades JSON / schema
# -----------------------
def ensure_sample_schema(sample: dict) -> dict:
    sample.setdefault("name", sample.get("name","") or "")
    sample.setdefault("url", sample.get("url","") or "")
    sample.setdefault("icono", sample.get("icono","") or "")
    sample.setdefault("iconoHorizontal", sample.get("iconoHorizontal","") or "")
    sample.setdefault("iconpng", sample.get("iconpng","") or "")
    sample.setdefault("type", (sample.get("type","") or "PELICULA"))
    sample.setdefault("descripcion", sample.get("descripcion","") or "")
    sample.setdefault("anio", sample.get("anio","") or "")
    sample.setdefault("genero", sample.get("genero","") or "")
    sample.setdefault("duracion", sample.get("duracion","") or "")
    return sample

def should_set(current_val: str) -> bool:
    if not ONLY_FILL_EMPTY:
        return True
    return not (current_val or "").strip()

def is_tmdb_image(url: str) -> bool:
    u = (url or "").strip().lower()
    return (TMDB_IMG_HOST in u) and ("/t/p/" in u)

# -----------------------
# Normalización de título (conservadora)
# -----------------------
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
JUNK_RE = re.compile(
    r"(\b(1080p|720p|2160p|4k|uhd|hdr|sdr|webrip|web\-?dl|bluray|brrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|ac3|dts|"
    r"latino|castellano|subtitulado|sub|dual|multi|esp|eng|vose|español|english)\b)",
    re.IGNORECASE,
)
BRACKETS_RE = re.compile(r"[\[\(\{].*?[\]\)\}]")

# ✅ Cambio: agrego "/" como separador para evitar queries tipo "3/El ..."
SEP_RE = re.compile(r"[._\-/]+")
MULTISPACE_RE = re.compile(r"\s{2,}")
PART_WORD_BEFORE_NUM_RE = re.compile(r"\b(parte|part)\b\s*(?=\d+\b)", re.IGNORECASE)

def extract_year(text: str) -> str:
    if not text:
        return ""
    yrs = YEAR_RE.findall(text)
    return yrs[-1] if yrs else ""

def _strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _fix_broken_words_safe(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\b[gG]\s+nesis\b", "genesis", t)
    t = re.sub(r"\b(\w+?)ci\s+n\b", r"\1cion", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\w+?)li\s+n\b", r"\1lion", t, flags=re.IGNORECASE)
    return t

def clean_title(name: str) -> str:
    t = (name or "").strip()
    if not t:
        return ""
    t = _fix_broken_words_safe(t)
    t = BRACKETS_RE.sub(" ", t)
    t = SEP_RE.sub(" ", t)
    t = JUNK_RE.sub(" ", t)
    t = PART_WORD_BEFORE_NUM_RE.sub(" ", t)
    t = re.sub(r"\s+\b(19\d{2}|20\d{2})\b\s*$", "", t).strip()
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def norm_for_match(s: str) -> str:
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

# -----------------------
# TMDB helpers
# -----------------------
def search_movie(query: str, year: str, lang: str, stats: dict):
    params = {"language": lang, "query": query, "include_adult": "false"}
    if year:
        params["year"] = year
    res = tmdb_get("/search/movie", params=params)
    if res is None:
        stats["search_failures"] += 1
        return []
    return res.get("results", []) or []

def fetch_details(movie_id: int, lang: str, stats: dict):
    det = tmdb_get(f"/movie/{movie_id}", params={"language": lang})
    if det is None:
        stats["details_failures"] += 1
        return None, None
    imgs = tmdb_get(
        f"/movie/{movie_id}/images",
        params={"include_image_language": "es,en,null"}
    )
    # imgs puede ser None; no pasa nada
    return det, imgs

def to_img_url(path: str) -> str:
    return (IMG_BASE + path) if path else ""

def pick_logo_url(imgs: dict) -> str:
    logos = (imgs or {}).get("logos") or []
    def score(x):
        iso = x.get("iso_639_1")
        if iso == "es":
            return 0
        if iso == "en":
            return 1
        if iso in (None, "", "null"):
            return 2
        return 3
    logos_sorted = sorted(logos, key=score)
    if not logos_sorted:
        return ""
    fp = logos_sorted[0].get("file_path") or ""
    return (IMG_BASE + fp) if fp else ""

def choose_best_result(results: list, query_clean: str, year: str):
    qn = norm_for_match(query_clean)
    if not qn:
        return None, 0.0

    best = None
    best_score = 0.0

    for r in (results or []):
        rid = r.get("id")
        if not rid:
            continue

        cand_year = (r.get("release_date") or "")[:4]
        titles = []
        if r.get("title"):
            titles.append(r.get("title"))
        if r.get("original_title"):
            titles.append(r.get("original_title"))

        pop = float(r.get("popularity") or 0.0)
        pop_bonus = min(pop, 50.0) / 5.0  # 0..10

        for tt in titles:
            tt_clean = clean_title(tt)
            tn = norm_for_match(tt_clean)
            if not tn:
                continue

            sim = _sim(qn, tn) * 100.0
            score = sim

            if tn == qn:
                score += 50.0
            if year and cand_year == year:
                score += 30.0
            if tn.startswith(qn) or qn.startswith(tn):
                score += 10.0

            score += pop_bonus

            if score > best_score:
                best_score = score
                best = r

    if best and best_score >= 50.0:
        return best, best_score
    return None, best_score

# Cache para no repetir queries
cache = {}  # (clean_lower, year) -> hit dict or None

def build_hit_for_sample(orig_name: str, orig_year: str, stats: dict):
    q_clean = clean_title(orig_name)
    year_for_search = (orig_year or "").strip() or extract_year(orig_name)

    cache_key = (q_clean.lower(), year_for_search)
    if cache_key in cache:
        return cache[cache_key]

    hit = None

    variants = []
    if q_clean:
        variants.append(q_clean)

    q_ascii = _strip_accents(q_clean)
    if q_ascii and q_ascii.lower() != q_clean.lower():
        variants.append(q_ascii)

    best_r = None
    best_score = 0.0

    for lang_try in (LANGUAGE, "en-US"):
        for qv in variants:
            results = search_movie(qv, year_for_search, lang_try, stats)
            if not results and year_for_search:
                results = search_movie(qv, "", lang_try, stats)

            cand, score = choose_best_result(results, qv, year_for_search)
            if cand and score > best_score:
                best_score = score
                best_r = cand

    if best_r:
        mid = best_r.get("id")
        if mid:
            det_es, imgs = fetch_details(mid, LANGUAGE, stats)
            if det_es is None:
                cache[cache_key] = None
                return None

            det_fb = None
            if FALLBACK_LANGUAGE and FALLBACK_LANGUAGE != LANGUAGE:
                det_fb, _ = fetch_details(mid, FALLBACK_LANGUAGE, stats)

            title_es = (det_es.get("title") or "").strip()
            overview_es = (det_es.get("overview") or "").strip()

            title_fb = (det_fb.get("title") or "").strip() if det_fb else ""
            overview_fb = (det_fb.get("overview") or "").strip() if det_fb else ""

            release_date = (det_es.get("release_date") or "").strip()
            tmdb_year = release_date[:4] if release_date else ""
            runtime = det_es.get("runtime") or 0
            genres_list = [g.get("name","").strip() for g in (det_es.get("genres") or []) if g.get("name")]

            poster = to_img_url(det_es.get("poster_path") or "")
            backdrop = to_img_url(det_es.get("backdrop_path") or "")
            logo = pick_logo_url(imgs)

            hit = {
                "title_es": title_es,
                "title_fb": title_fb,
                "overview_es": overview_es,
                "overview_fb": overview_fb,
                "year": tmdb_year,
                "genres_list": genres_list,
                "genres_str": ", ".join([g for g in genres_list if g]),
                "runtime": (f"{int(runtime)} min" if runtime else ""),
                "poster": poster,
                "backdrop": backdrop,
                "logo": logo,
            }

    cache[cache_key] = hit
    return hit

def enrich_sample_in_place(sample: dict, stats: dict):
    sample = ensure_sample_schema(sample)

    tipo = (sample.get("type") or "").upper().strip()
    if tipo and tipo != "PELICULA":
        return sample, False

    orig_name = (sample.get("name") or "").strip()
    orig_year = (sample.get("anio") or "").strip()

    if not orig_name:
        return sample, False

    hit = build_hit_for_sample(orig_name, orig_year, stats)
    if not hit:
        return sample, False

    changed = False

    # Rellenar vacíos
    if should_set(sample.get("name","")):
        if hit.get("title_es"):
            sample["name"] = hit["title_es"]; stats["filled_fields"] += 1; changed = True
        elif hit.get("title_fb"):
            sample["name"] = hit["title_fb"]; stats["filled_fields"] += 1; changed = True

    if should_set(sample.get("descripcion","")):
        if hit.get("overview_es"):
            sample["descripcion"] = hit["overview_es"]; stats["filled_fields"] += 1; changed = True
        elif hit.get("overview_fb"):
            sample["descripcion"] = hit["overview_fb"]; stats["filled_fields"] += 1; changed = True

    if should_set(sample.get("anio","")) and hit.get("year"):
        sample["anio"] = hit["year"]; stats["filled_fields"] += 1; changed = True

    if should_set(sample.get("genero","")) and hit.get("genres_str"):
        sample["genero"] = hit["genres_str"]; stats["filled_fields"] += 1; changed = True

    if should_set(sample.get("duracion","")) and hit.get("runtime"):
        sample["duracion"] = hit["runtime"]; stats["filled_fields"] += 1; changed = True

    # Imágenes: si ya es TMDB no tocar; si no es TMDB -> reemplazar si hay
    cur = sample.get("icono","") or ""
    if not cur.strip():
        if hit.get("poster"):
            sample["icono"] = hit["poster"]; stats["filled_images"] += 1; changed = True
    else:
        if (not is_tmdb_image(cur)) and hit.get("poster"):
            sample["icono"] = hit["poster"]; stats["replaced_images"] += 1; changed = True

    cur = sample.get("iconoHorizontal","") or ""
    if not cur.strip():
        if hit.get("backdrop"):
            sample["iconoHorizontal"] = hit["backdrop"]; stats["filled_images"] += 1; changed = True
    else:
        if (not is_tmdb_image(cur)) and hit.get("backdrop"):
            sample["iconoHorizontal"] = hit["backdrop"]; stats["replaced_images"] += 1; changed = True

    cur = sample.get("iconpng","") or ""
    if not cur.strip():
        if hit.get("logo"):
            sample["iconpng"] = hit["logo"]; stats["filled_images"] += 1; changed = True
    else:
        if (not is_tmdb_image(cur)) and hit.get("logo"):
            sample["iconpng"] = hit["logo"]; stats["replaced_images"] += 1; changed = True

    sample = ensure_sample_schema(sample)
    return sample, changed

# -----------------------
# Cargar JSON
# -----------------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

is_records = isinstance(data, list) and (len(data) == 0 or (isinstance(data[0], dict) and "samples" in data[0]))

stats = defaultdict(int)
processed = 0
changed_count = 0
hard_errors = 0

def _safe_json_dump(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

if is_records:
    for rec in data:
        samples = rec.get("samples") or []
        for i in range(len(samples)):
            processed += 1
            s = samples[i]
            before = _safe_json_dump(s)
            try:
                s2, changed = enrich_sample_in_place(s, stats)
            except Exception:
                hard_errors += 1
                stats["tmdb_errors"] += 1
                s2, changed = s, False
            after = _safe_json_dump(s2)
            samples[i] = s2
            if changed or (before != after):
                changed_count += 1
else:
    for i in range(len(data)):
        processed += 1
        s = data[i]
        before = _safe_json_dump(s)
        try:
            s2, changed = enrich_sample_in_place(s, stats)
        except Exception:
            hard_errors += 1
            stats["tmdb_errors"] += 1
            s2, changed = s, False
        after = _safe_json_dump(s2)
        data[i] = s2
        if changed or (before != after):
            changed_count += 1

# -----------------------
# Guardar output
# -----------------------
base = os.path.basename(JSON_PATH)
stem = re.sub(r"\.json$", "", base, flags=re.IGNORECASE)

out_json = os.path.join(OUTPUT_DIR, f"{stem}.tmdb_fixed.json")
out_report = os.path.join(OUTPUT_DIR, f"{stem}.tmdb_fixed.report.json")

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

report = {
    "input_json": JSON_PATH,
    "output_json": out_json,
    "language": LANGUAGE,
    "fallback_language": FALLBACK_LANGUAGE,
    "only_fill_empty": ONLY_FILL_EMPTY,
    "processed_samples": processed,
    "samples_changed": changed_count,
    "filled_fields": int(stats["filled_fields"]),
    "filled_images": int(stats["filled_images"]),
    "replaced_images_not_tmdb": int(stats["replaced_images"]),
    "cache_size": len(cache),
    "search_failures": int(stats["search_failures"]),
    "details_failures": int(stats["details_failures"]),
    "tmdb_errors": int(stats["tmdb_errors"]),
    "hard_errors_caught": hard_errors,
    "notes": "No se cae con 500/429: se reintenta y si falla se mantiene el item. '/' se normaliza para evitar queries raras."
}

with open(out_report, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
