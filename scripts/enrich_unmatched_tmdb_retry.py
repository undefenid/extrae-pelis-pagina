import os
import re
import json
import time
import math
import unicodedata
import requests
from urllib.parse import urlparse
from collections import defaultdict, OrderedDict

API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
JSON_URL = os.environ.get("JSON_URL", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "tmdb_retry").strip()
CHUNK_SIZE = int((os.environ.get("CHUNK_SIZE") or "2000").strip())
LANGUAGE = (os.environ.get("LANGUAGE") or "es-MX").strip()
FALLBACK_LANGUAGE = (os.environ.get("FALLBACK_LANGUAGE") or "en-US").strip()
ONLY_FILL_EMPTY = (os.environ.get("ONLY_FILL_EMPTY", "true").strip().lower() == "true")
MIN_DELAY_MS = int((os.environ.get("MIN_DELAY_MS") or "150").strip())
SCORE_THRESHOLD = float((os.environ.get("SCORE_THRESHOLD") or "0.38").strip())
USE_WIKIPEDIA = (os.environ.get("USE_WIKIPEDIA", "false").strip().lower() == "true")

if not API_KEY:
    raise SystemExit("Falta secrets.TMDB_API_KEY")
if not JSON_URL:
    raise SystemExit("Falta json_url")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/original"

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

def tmdb_get(path, params=None, retries=4):
    url = TMDB_BASE + path
    params = dict(params or {})
    params["api_key"] = API_KEY

    last_exc = None
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.0 + attempt)
            continue

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "2") or "2")
            time.sleep(retry_after + (attempt * 1.5))
            continue

        if 500 <= r.status_code < 600:
            time.sleep(1.0 + attempt)
            continue

        r.raise_for_status()
        return r.json()

    if last_exc:
        raise last_exc
    r.raise_for_status()
    return r.json()

def search_movie(query: str, year: str, lang: str):
    params = {"language": lang, "query": query, "include_adult": "false"}
    if year:
        params["year"] = year
    res = tmdb_get("/search/movie", params=params)
    return res.get("results", []) or []

def fetch_details(movie_id: int, lang: str):
    det = tmdb_get(f"/movie/{movie_id}", params={"language": lang})
    imgs = tmdb_get(f"/movie/{movie_id}/images", params={"include_image_language": "es,en,null"})
    return det, imgs

def to_img_url(path: str) -> str:
    return (IMG_BASE + path) if path else ""

def pick_logo_url(imgs: dict) -> str:
    logos = (imgs or {}).get("logos") or []
    def score(x):
        iso = x.get("iso_639_1")
        if iso == "es": return 0
        if iso == "en": return 1
        if iso in (None, "", "null"): return 2
        return 3
    logos_sorted = sorted(logos, key=score)
    if not logos_sorted:
        return ""
    fp = logos_sorted[0].get("file_path") or ""
    return (IMG_BASE + fp) if fp else ""

# -----------------------
# Load input JSON
# -----------------------
raw = requests.get(JSON_URL, timeout=90)
raw.raise_for_status()
data = raw.json()

def safe_basename_from_url(u: str) -> str:
    try:
        p = urlparse(u)
        name = os.path.basename(p.path) or "input.json"
    except Exception:
        name = "input.json"
    if not name.lower().endswith(".json"):
        name += ".json"
    return name

in_name = safe_basename_from_url(JSON_URL)
stem = re.sub(r"\.json$", "", in_name, flags=re.IGNORECASE)

# -----------------------
# Normalizers & heuristics
# -----------------------
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

def simple_clean(s: str) -> str:
    # quita basura típica, pero sin destruir el título
    t = (s or "").strip()
    t = t.replace("_", " ").replace("-", " ")
    t = QUALITY_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_year_from_any(text: str) -> str:
    yrs = YEAR_RE.findall(text or "")
    return yrs[-1] if yrs else ""

ROMAN = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",7:"VII",8:"VIII",9:"IX",10:"X"}
def romanize(n: int) -> str:
    return ROMAN.get(n, "")

def tokenize_for_score(s: str):
    t = strip_accents((s or "").lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = norm_spaces(t)
    return [x for x in t.split(" ") if x]

def jaccard(a_tokens, b_tokens) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    A = set(a_tokens)
    B = set(b_tokens)
    inter = len(A & B)
    uni = len(A | B)
    return (inter / uni) if uni else 0.0

ES_CHARS_RE = re.compile(r"[áéíóúñü¿¡]", re.IGNORECASE)
ES_WORDS_RE = re.compile(r"\b(el|la|los|las|un|una|unos|unas|de|del|y|en|para|con|sin|por|mi|tu|su)\b", re.IGNORECASE)
def looks_spanish(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    return bool(ES_CHARS_RE.search(t) or ES_WORDS_RE.search(t))

def should_replace_title_with_es(orig_title: str, tmdb_title_es: str) -> bool:
    if not orig_title or not tmdb_title_es:
        return False
    if looks_spanish(orig_title):
        return False
    if strip_accents(orig_title).strip().lower() == strip_accents(tmdb_title_es).strip().lower():
        return False
    if looks_spanish(tmdb_title_es):
        return True
    return False

def should_set(current_val: str) -> bool:
    if not ONLY_FILL_EMPTY:
        return True
    return not (current_val or "").strip()

def ensure_schema(sample: dict) -> dict:
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

# -----------------------
# Wikipedia optional helper (no scraping)
# eswiki search -> langlinks en
# -----------------------
def wiki_en_title_from_es_query(q: str, year: str = "") -> str:
    if not q:
        return ""
    try:
        # 1) buscar en eswiki
        sr = (q + (f" {year}" if year else "")).strip()
        r = requests.get(
            "https://es.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch": sr, "format":"json"},
            timeout=15
        )
        r.raise_for_status()
        js = r.json()
        hits = (js.get("query", {}).get("search", []) or [])
        if not hits:
            return ""
        title_es = hits[0].get("title") or ""
        if not title_es:
            return ""

        # 2) pedir langlinks a en
        r2 = requests.get(
            "https://es.wikipedia.org/w/api.php",
            params={
                "action":"query",
                "prop":"langlinks",
                "titles": title_es,
                "lllang":"en",
                "format":"json"
            },
            timeout=15
        )
        r2.raise_for_status()
        js2 = r2.json()
        pages = (js2.get("query", {}).get("pages", {}) or {})
        for _, p in pages.items():
            ll = p.get("langlinks") or []
            if ll:
                return ll[0].get("*") or ""
        return ""
    except Exception:
        return ""

# -----------------------
# Build query variants (permisivo)
# -----------------------
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")

def build_query_variants(raw_name: str):
    """
    Devuelve: (queries: list[str], guessed_year: str, sequel_num: int or 0)
    """
    name = (raw_name or "").strip()
    if not name:
        return [], "", 0

    # 1) detectar año en cualquier lado
    year = extract_year_from_any(name)

    # 2) separar por pipes
    parts = [p.strip() for p in PIPE_SPLIT_RE.split(name) if p.strip()]
    # sacar basura de cada parte
    parts_clean = [norm_spaces(simple_clean(p)) for p in parts if p]

    # 3) detectar secuela numérica en partes: "1", "2", "03"
    sequel = 0
    for p in parts_clean:
        if re.fullmatch(r"0?\d{1,2}", p):
            n = int(p)
            if 1 <= n <= 20:
                sequel = n

    # 4) escoger “mejor” parte como título base:
    # - prioriza parte con más letras (no solo números)
    title_candidates = [p for p in parts_clean if not re.fullmatch(r"0?\d{1,4}", p)]
    base = title_candidates[0] if title_candidates else (parts_clean[0] if parts_clean else name)

    # 5) limpiar base más agresivo: quitar tokens de calidad/idioma, y símbolos
    base2 = simple_clean(base)
    base2 = base2.replace(".", "")  # E.T. -> ET
    base2 = norm_spaces(base2)

    # 6) variantes principales
    variants = set()

    # sin pipes
    variants.add(base2)

    # si hay parte extra tipo "El Extraterrestre", también probarla sola
    if len(title_candidates) >= 2:
        variants.add(simple_clean(title_candidates[1]).replace(".", ""))

    # quitar año explícito del string (si estaba pegado)
    if year:
        variants.add(norm_spaces(re.sub(rf"\b{re.escape(year)}\b", " ", base2)))

    # versión sin acentos
    variants.add(strip_accents(base2))

    # 7) secuela: si sequel>1 añadir
    if sequel >= 2:
        variants.add(f"{base2} {sequel}")
        rn = romanize(sequel)
        if rn:
            variants.add(f"{base2} {rn}")

    # 8) “No Maches” -> intenta corregir con variantes mínimas (sin diccionario grande)
    # (esto evita pedir web para faltas comunes)
    if "maches" in strip_accents(base2).lower():
        variants.add(re.sub(r"(?i)\bmaches\b", "manches", base2))

    # 9) Limpiar finales raros
    final = []
    for v in variants:
        v = norm_spaces(v)
        v = re.sub(r"\b(latino|castellano|subtitulado|dual|multi|esp|eng|vose)\b", " ", v, flags=re.IGNORECASE)
        v = QUALITY_RE.sub(" ", v)
        v = norm_spaces(v)
        if v and len(v) >= 2:
            final.append(v)

    # orden estable
    final = list(OrderedDict.fromkeys(final).keys())
    return final, year, sequel

# -----------------------
# Scoring candidates
# -----------------------
def candidate_year(res):
    rd = (res.get("release_date") or "").strip()
    return rd[:4] if rd else ""

def score_candidate(query: str, year: str, res: dict) -> float:
    qt = tokenize_for_score(query)
    # usar title y original_title para comparar
    t1 = res.get("title") or ""
    t2 = res.get("original_title") or ""
    rt = tokenize_for_score(t1 + " " + t2)

    sim = jaccard(qt, rt)  # 0..1

    # bonus por año (suave)
    yb = 0.0
    ry = candidate_year(res)
    if year and ry and year.isdigit() and ry.isdigit():
        d = abs(int(year) - int(ry))
        if d == 0:
            yb = 0.18
        elif d == 1:
            yb = 0.10
        elif d <= 3:
            yb = 0.05
        elif d <= 6:
            yb = 0.0
        else:
            yb = -0.10  # no lo descarta, solo baja score

    pop = float(res.get("popularity") or 0.0)
    vc = float(res.get("vote_count") or 0.0)

    # popularidad/votos (pequeño empujón)
    popb = min(0.08, math.log1p(pop) / 20.0)
    vcb = min(0.06, math.log1p(vc) / 30.0)

    # score final
    return (0.78 * sim) + yb + popb + vcb

# -----------------------
# Retry matcher
# -----------------------
cache_best = {}  # (name_norm, year) -> best_tmdb_id or None
cache_details = {}  # id -> (det_es, det_fb, imgs)

def pick_best_tmdb(raw_name: str, year_from_sample: str):
    """
    Devuelve: (best_id, best_score, debug_info)
    """
    queries, year_guess, sequel = build_query_variants(raw_name)
    year = year_from_sample.strip() if (year_from_sample or "").strip() else year_guess

    key = (strip_accents(raw_name).lower().strip(), year)
    if key in cache_best:
        return cache_best[key], 1.0 if cache_best[key] else 0.0, {"cached": True}

    candidates = {}  # id -> best_score

    # Búsquedas: es-MX y en-US, con y sin año
    langs = [LANGUAGE, "en-US"]
    for q in queries[:8]:  # limit variantes por performance
        for lang in langs:
            # con año
            try:
                res1 = search_movie(q, year, lang) if year else search_movie(q, "", lang)
            except Exception:
                res1 = []
            for r in (res1[:12] or []):
                mid = r.get("id")
                if not mid:
                    continue
                sc = score_candidate(q, year, r)
                candidates[mid] = max(candidates.get(mid, 0.0), sc)

            # sin año (si año no ayudó o está mal)
            try:
                res2 = search_movie(q, "", lang)
            except Exception:
                res2 = []
            for r in (res2[:12] or []):
                mid = r.get("id")
                if not mid:
                    continue
                sc = score_candidate(q, year, r)
                candidates[mid] = max(candidates.get(mid, 0.0), sc)

    # Wikipedia fallback: obtener título en inglés y reintentar
    if USE_WIKIPEDIA and not candidates:
        q0 = queries[0] if queries else raw_name
        en_title = wiki_en_title_from_es_query(q0, year)
        if en_title:
            for lang in langs:
                try:
                    resw = search_movie(en_title, year, lang) if year else search_movie(en_title, "", lang)
                except Exception:
                    resw = []
                for r in (resw[:15] or []):
                    mid = r.get("id")
                    if not mid:
                        continue
                    sc = score_candidate(en_title, year, r)
                    candidates[mid] = max(candidates.get(mid, 0.0), sc)

    if not candidates:
        cache_best[key] = None
        return None, 0.0, {"queries": queries, "year": year, "candidates": 0}

    # elegir mejor score
    best_id, best_sc = max(candidates.items(), key=lambda kv: kv[1])

    # threshold de aceptación
    if best_sc < SCORE_THRESHOLD:
        cache_best[key] = None
        return None, best_sc, {"queries": queries, "year": year, "best_score": best_sc, "threshold": SCORE_THRESHOLD}

    cache_best[key] = best_id
    return best_id, best_sc, {"queries": queries, "year": year, "best_score": best_sc, "threshold": SCORE_THRESHOLD}

def get_details(mid: int):
    if mid in cache_details:
        return cache_details[mid]
    det_es, imgs = fetch_details(mid, LANGUAGE)
    det_fb = None
    if FALLBACK_LANGUAGE and FALLBACK_LANGUAGE != LANGUAGE:
        try:
            det_fb, _ = fetch_details(mid, FALLBACK_LANGUAGE)
        except Exception:
            det_fb = None
    cache_details[mid] = (det_es, det_fb, imgs)
    return cache_details[mid]

def enrich_sample(sample: dict):
    """
    Retorna: (sample_enriched, primary_genre, matched_bool)
    """
    sample = ensure_schema(sample)
    if (sample.get("type") or "").upper().strip() != "PELICULA":
        return sample, "", False

    raw_name = (sample.get("name") or "").strip()
    if not raw_name:
        return sample, "", False

    year_from_sample = (sample.get("anio") or "").strip()

    best_id, best_sc, dbg = pick_best_tmdb(raw_name, year_from_sample)
    if not best_id:
        return sample, "", False

    det_es, det_fb, imgs = get_details(best_id)

    title_es = (det_es.get("title") or "").strip()
    overview_es = (det_es.get("overview") or "").strip()
    release_date = (det_es.get("release_date") or "").strip()
    tmdb_year = release_date[:4] if release_date else ""
    runtime = det_es.get("runtime") or 0
    genres_list = [g.get("name","").strip() for g in (det_es.get("genres") or []) if g.get("name")]
    genres_str = ", ".join([g for g in genres_list if g])

    # fallback overview si es-MX vacío
    if det_fb:
        if not overview_es:
            overview_es = (det_fb.get("overview") or "").strip()

    poster = to_img_url(det_es.get("poster_path") or "")
    backdrop = to_img_url(det_es.get("backdrop_path") or "")
    logo = pick_logo_url(imgs)

    # TITULO con tu regla
    orig_name = raw_name
    if should_set(sample.get("name","")):
        if title_es:
            sample["name"] = title_es
    else:
        if should_replace_title_with_es(orig_name, title_es):
            sample["name"] = title_es

    if should_set(sample.get("descripcion","")) and overview_es:
        sample["descripcion"] = overview_es

    if should_set(sample.get("anio","")) and tmdb_year:
        sample["anio"] = tmdb_year

    if should_set(sample.get("genero","")) and genres_str:
        sample["genero"] = genres_str

    if should_set(sample.get("duracion","")) and runtime:
        sample["duracion"] = f"{int(runtime)} min"

    if should_set(sample.get("icono","")) and poster:
        sample["icono"] = poster
    if should_set(sample.get("iconoHorizontal","")) and backdrop:
        sample["iconoHorizontal"] = backdrop
    if should_set(sample.get("iconpng","")) and logo:
        sample["iconpng"] = logo

    sample = ensure_schema(sample)
    primary_genre = genres_list[0] if genres_list else "Sin información"
    return sample, primary_genre, True

# -----------------------
# Process input records
# -----------------------
# input esperado: [{name, samples:[...]}]
all_movie_samples = []
for rec in (data or []):
    for s in (rec.get("samples") or []):
        all_movie_samples.append(s)

genre_groups = defaultdict(list)
still_unmatched = []
matched = 0

for s in all_movie_samples:
    s2, g, ok = enrich_sample(s)
    if ok:
        matched += 1
        genre_groups[g or "Sin información"].append(s2)
    else:
        still_unmatched.append(s2)
        genre_groups["Sin información"].append(s2)

def ordered_genres(groups_dict):
    keys = list(groups_dict.keys())
    def ksort(x):
        lx = x.strip().lower()
        if lx in ("sin información", "sin informacion"):
            return (1, lx)
        return (0, lx)
    return sorted(keys, key=ksort)

records = [{"name": g, "samples": genre_groups[g]} for g in ordered_genres(genre_groups)]

# -----------------------
# Split + save
# -----------------------
def split_records(records_list, max_samples):
    parts = []
    cur = []
    cur_count = 0

    for r in records_list:
        samples = r.get("samples", []) or []
        if len(samples) > max_samples:
            for i in range(0, len(samples), max_samples):
                chunk = samples[i:i+max_samples]
                if cur and (cur_count + len(chunk) > max_samples):
                    parts.append(cur)
                    cur = []
                    cur_count = 0
                cur.append({"name": r.get("name",""), "samples": chunk})
                cur_count += len(chunk)
            continue

        if cur and (cur_count + len(samples) > max_samples):
            parts.append(cur)
            cur = []
            cur_count = 0
        cur.append(r)
        cur_count += len(samples)

    if cur:
        parts.append(cur)
    return parts

def save_parts(parts_list, base_filename):
    files = []
    if len(parts_list) == 1:
        fn = f"{base_filename}.json"
        with open(os.path.join(OUTPUT_DIR, fn), "w", encoding="utf-8") as f:
            json.dump(parts_list[0], f, indent=2, ensure_ascii=False)
        files.append(fn)
        return files

    for i, part in enumerate(parts_list, start=1):
        fn = f"{base_filename}_part{i:03d}.json"
        with open(os.path.join(OUTPUT_DIR, fn), "w", encoding="utf-8") as f:
            json.dump(part, f, indent=2, ensure_ascii=False)
        files.append(fn)

    manifest = {
        "total_parts": len(parts_list),
        "chunk_size_samples": CHUNK_SIZE,
        "files": files,
    }
    with open(os.path.join(OUTPUT_DIR, f"{base_filename}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return files

parts = split_records(records, CHUNK_SIZE)
out_files = save_parts(parts, f"{stem}.retry_enriched_by_genre")

still_unmatched_records = [{"name":"Sin información", "samples": still_unmatched}] if still_unmatched else []
still_files = []
if still_unmatched_records:
    still_parts = split_records(still_unmatched_records, CHUNK_SIZE)
    still_files = save_parts(still_parts, f"{stem}.still_unmatched")

report = {
    "source_url": JSON_URL,
    "language": LANGUAGE,
    "fallback_language": FALLBACK_LANGUAGE,
    "only_fill_empty": ONLY_FILL_EMPTY,
    "score_threshold": SCORE_THRESHOLD,
    "use_wikipedia": USE_WIKIPEDIA,
    "samples_in": len(all_movie_samples),
    "matched_tmdb": matched,
    "still_unmatched": len(still_unmatched),
    "output_dir": OUTPUT_DIR,
    "enriched_files": out_files,
    "still_unmatched_files": still_files,
    "notes": [
        "Se intenta match con variantes: split por '|', sin acentos, sin puntos, sin calidad/idioma, secuelas (2/II).",
        "El año se usa como bonus suave (no descarta si está mal).",
        "Wikipedia es opcional y usa API (no scraping) para conseguir título en inglés."
    ]
}

with open(os.path.join(OUTPUT_DIR, f"{stem}.retry_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
