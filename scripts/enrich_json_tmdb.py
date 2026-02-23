import os
import re
import json
import time
import requests
import unicodedata
import difflib
from urllib.parse import urlparse
from collections import OrderedDict, defaultdict

API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
JSON_URL = os.environ.get("JSON_URL", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "tmdb_enriched").strip()
CHUNK_SIZE = int((os.environ.get("CHUNK_SIZE") or "2000").strip())
LANGUAGE = (os.environ.get("LANGUAGE") or "es-MX").strip()
FALLBACK_LANGUAGE = (os.environ.get("FALLBACK_LANGUAGE") or "en-US").strip()
ONLY_FILL_EMPTY = (os.environ.get("ONLY_FILL_EMPTY", "true").strip().lower() == "true")
MIN_DELAY_MS = int((os.environ.get("MIN_DELAY_MS") or "150").strip())

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

# -----------------------
# Descarga JSON por URL
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
# Normalizacion del titulo
# -----------------------
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# tokens basura (calidad, idioma, flags, etc)
JUNK_RE = re.compile(
    r"(\b(1080p|720p|2160p|4k|uhd|hdr|sdr|webrip|web\-?dl|bluray|brrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|ac3|dts|"
    r"latino|castellano|subtitulado|sub|dual|multi|esp|eng|vose|español|english)\b)",
    re.IGNORECASE,
)

# elimina TODO lo que esté entre (), [], {}
BRACKETS_RE = re.compile(r"[\[\(\{].*?[\]\)\}]")
SEP_RE = re.compile(r"[._\-]+")
MULTISPACE_RE = re.compile(r"\s{2,}")

# regla: remover la palabra "parte/part" SOLO si va pegada a un número (parte 1, part 2)
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
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def _fix_broken_words_safe(t: str) -> str:
    """
    Arreglos MUY conservadores para no romper títulos:
    - 'salvaci n' -> 'salvacion'
    - 'rebeli n' -> 'rebelion'
    - 'g nesis' -> 'genesis' (solo este caso exacto)
    """
    if not t:
        return ""

    # caso específico: g nesis -> genesis
    t = re.sub(r"\b[gG]\s+nesis\b", "genesis", t)

    # casos tipo "...ci n" / "...li n" al final de palabra -> "...cion" / "...lion"
    # (sin acento, pero TMDB matchea igual)
    t = re.sub(r"\b(\w+?)ci\s+n\b", r"\1cion", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\w+?)li\s+n\b", r"\1lion", t, flags=re.IGNORECASE)

    return t

def clean_title(name: str) -> str:
    t = (name or "").strip()
    if not t:
        return ""

    # primero arreglos de “palabras rotas” (antes de borrar separadores)
    t = _fix_broken_words_safe(t)

    # quitar cualquier cosa entre brackets (incluye (2024)[dual], etc)
    t = BRACKETS_RE.sub(" ", t)

    # normalizar separadores
    t = SEP_RE.sub(" ", t)

    # quitar tokens de calidad/idioma/etc
    t = JUNK_RE.sub(" ", t)

    # quitar "parte/part" si precede a un número (mantiene el número)
    t = PART_WORD_BEFORE_NUM_RE.sub(" ", t)

    # si termina con año, quitarlo del query (el año se manda por param 'year')
    t = re.sub(r"\s+\b(19\d{2}|20\d{2})\b\s*$", "", t).strip()

    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def norm_cmp(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_for_match(s: str) -> str:
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

ES_CHARS_RE = re.compile(r"[áéíóúñü¿¡]", re.IGNORECASE)
ES_WORDS_RE = re.compile(r"\b(el|la|los|las|un|una|unos|unas|de|del|y|en|para|con|sin|por|mi|tu|su)\b", re.IGNORECASE)

def looks_spanish(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    if ES_CHARS_RE.search(t):
        return True
    if ES_WORDS_RE.search(t):
        return True
    return False

# Regla pedida:
# Si el titulo actual NO parece español y TMDB trae title en es-MX distinto => reemplazar.
# Si no hay title es-MX, se mantiene.
def should_replace_title_with_es(orig_title: str, tmdb_title_es: str) -> bool:
    if not orig_title or not tmdb_title_es:
        return False
    if looks_spanish(orig_title):
        return False
    if norm_cmp(orig_title) == norm_cmp(tmdb_title_es):
        return False
    if looks_spanish(tmdb_title_es):
        return True
    return False

# -----------------------
# TMDB Search + Details + Images
# -----------------------
def search_movie(query: str, year: str, lang: str):
    params = {"language": lang, "query": query, "include_adult": "false"}
    if year:
        params["year"] = year
    res = tmdb_get("/search/movie", params=params)
    return res.get("results", []) or []

def fetch_details(movie_id: int, lang: str):
    det = tmdb_get(f"/movie/{movie_id}", params={"language": lang})
    imgs = tmdb_get(
        f"/movie/{movie_id}/images",
        params={"include_image_language": "es,en,null"}
    )
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

def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def choose_best_result(results: list, query_clean: str, year: str):
    """
    Elige mejor candidato para evitar falsos positivos:
    - similitud entre query_clean y title/original_title
    - bonus si coincide el año
    """
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

    # umbral suave (no muy alto para no perder matches)
    if best and best_score >= 50.0:
        return best, best_score

    return None, best_score

# Cache para no repetir queries
cache = {}  # (clean_lower, year) -> hit dict or None

def should_set(current_val: str) -> bool:
    if not ONLY_FILL_EMPTY:
        return True
    return not (current_val or "").strip()

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

def enrich_movie_sample(sample: dict):
    """
    Devuelve: (sample_enriched, primary_genre, matched_bool)
    """
    sample = ensure_sample_schema(sample)

    tipo = (sample.get("type") or "").upper().strip()
    if tipo and tipo != "PELICULA":
        return sample, "", False

    orig_name = (sample.get("name") or "").strip()
    orig_year = (sample.get("anio") or "").strip()
    if not orig_name:
        return sample, "", False

    q_clean = clean_title(orig_name)
    year_for_search = orig_year or extract_year(orig_name)

    cache_key = (q_clean.lower(), year_for_search)
    if cache_key in cache:
        hit = cache[cache_key]
    else:
        hit = None

        # Variantes de query (sin arriesgar demasiado)
        variants = []
        if q_clean:
            variants.append(q_clean)

        q_ascii = _strip_accents(q_clean)
        if q_ascii and q_ascii.lower() != q_clean.lower():
            variants.append(q_ascii)

        # Buscar en LANGUAGE y fallback en-US y elegir el mejor match
        best_r = None
        best_score = 0.0

        for lang_try in (LANGUAGE, "en-US"):
            for qv in variants:
                results = search_movie(qv, year_for_search, lang_try)
                if not results and year_for_search:
                    results = search_movie(qv, "", lang_try)

                cand, score = choose_best_result(results, qv, year_for_search)
                if cand and score > best_score:
                    best_score = score
                    best_r = cand

        if best_r:
            mid = best_r.get("id")
            if mid:
                det_es, imgs = fetch_details(mid, LANGUAGE)

                det_fb = None
                if FALLBACK_LANGUAGE and FALLBACK_LANGUAGE != LANGUAGE:
                    try:
                        det_fb, _ = fetch_details(mid, FALLBACK_LANGUAGE)
                    except Exception:
                        det_fb = None

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

    if not hit:
        return sample, "", False

    # --- TITULO con tu regla ---
    if should_set(sample.get("name","")):
        if hit.get("title_es"):
            sample["name"] = hit["title_es"]
        elif hit.get("title_fb"):
            sample["name"] = hit["title_fb"]
    else:
        if should_replace_title_with_es(orig_name, hit.get("title_es","")):
            sample["name"] = hit["title_es"]

    # --- DESCRIPCION ---
    if should_set(sample.get("descripcion","")):
        if hit.get("overview_es"):
            sample["descripcion"] = hit["overview_es"]
        elif hit.get("overview_fb"):
            sample["descripcion"] = hit["overview_fb"]

    # --- RESTO (respetando ONLY_FILL_EMPTY como antes) ---
    if should_set(sample.get("anio","")) and hit.get("year"):
        sample["anio"] = hit["year"]

    if should_set(sample.get("genero","")) and hit.get("genres_str"):
        sample["genero"] = hit["genres_str"]

    if should_set(sample.get("duracion","")) and hit.get("runtime"):
        sample["duracion"] = hit["runtime"]

    # --- LOGOS: reemplazar SIEMPRE si TMDB trae (aunque ya exista en JSON) ---
    if hit.get("poster"):
        sample["icono"] = hit["poster"]

    if hit.get("backdrop"):
        sample["iconoHorizontal"] = hit["backdrop"]

    if hit.get("logo"):
        sample["iconpng"] = hit["logo"]

    sample = ensure_sample_schema(sample)
    primary_genre = (hit.get("genres_list") or [""])[0] if hit.get("genres_list") else ""
    return sample, (primary_genre or ""), True

# -----------------------
# Flatten y procesar
# -----------------------
movie_samples = []
non_movie_records = []

for rec in (data or []):
    samples = rec.get("samples") or []
    non_movies_here = []
    for s in samples:
        t = (s.get("type") or "").upper().strip()
        if t and t != "PELICULA":
            non_movies_here.append(s)
        else:
            movie_samples.append(s)

    if non_movies_here:
        non_movie_records.append({
            "name": (rec.get("name") or "OTROS"),
            "samples": non_movies_here
        })

genre_groups = defaultdict(list)
unmatched_samples = []
matched_count = 0
processed_count = 0

for s in movie_samples:
    processed_count += 1
    s2, primary_genre, matched = enrich_movie_sample(s)
    if matched:
        matched_count += 1
        g = primary_genre.strip() or "Sin información"
        genre_groups[g].append(s2)
    else:
        s2 = ensure_sample_schema(s2)
        genre_groups["Sin información"].append(s2)
        unmatched_samples.append(s2)

# -----------------------
# Build records (ordenado: genero ASC, "Sin información" al final)
# -----------------------
def ordered_genres(groups_dict):
    keys = list(groups_dict.keys())
    def ksort(x):
        if x.strip().lower() in ("sin información", "sin informacion"):
            return (1, x.lower())
        return (0, x.lower())
    return sorted(keys, key=ksort)

records = []
for g in ordered_genres(genre_groups):
    records.append({
        "name": g,
        "samples": genre_groups[g]
    })

unmatched_records = [{
    "name": "Sin información",
    "samples": unmatched_samples
}] if unmatched_samples else []

# -----------------------
# Split robusto
# -----------------------
def split_records(records_list, max_samples):
    parts = []
    cur_part = []
    cur_count = 0

    for r in records_list:
        name = r.get("name","")
        samples = r.get("samples", []) or []

        if len(samples) > max_samples:
            for i in range(0, len(samples), max_samples):
                chunk = samples[i:i+max_samples]
                if cur_part and (cur_count + len(chunk) > max_samples):
                    parts.append(cur_part)
                    cur_part = []
                    cur_count = 0
                cur_part.append({"name": name, "samples": chunk})
                cur_count += len(chunk)
            continue

        if cur_part and (cur_count + len(samples) > max_samples):
            parts.append(cur_part)
            cur_part = []
            cur_count = 0

        cur_part.append(r)
        cur_count += len(samples)

    if cur_part:
        parts.append(cur_part)

    return parts

parts = split_records(records, CHUNK_SIZE)
unmatched_parts = split_records(unmatched_records, CHUNK_SIZE) if unmatched_records else []
non_movie_parts = split_records(non_movie_records, CHUNK_SIZE) if non_movie_records else []

# -----------------------
# Save outputs
# -----------------------
def save_parts(parts_list, base_filename):
    if not parts_list:
        return []

    files = []
    if len(parts_list) == 1:
        fn = f"{base_filename}.json"
        out_path = os.path.join(OUTPUT_DIR, fn)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parts_list[0], f, indent=2, ensure_ascii=False)
        files.append(fn)
    else:
        for i, part in enumerate(parts_list, start=1):
            fn = f"{base_filename}_part{i:03d}.json"
            out_path = os.path.join(OUTPUT_DIR, fn)
            with open(out_path, "w", encoding="utf-8") as f:
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

enriched_files = save_parts(parts, f"{stem}.enriched_by_genre")
unmatched_files = save_parts(unmatched_parts, f"{stem}.unmatched") if unmatched_parts else []
non_movie_files = save_parts(non_movie_parts, f"{stem}.skipped_non_movies") if non_movie_parts else []

report = {
    "source_url": JSON_URL,
    "language": LANGUAGE,
    "fallback_language": FALLBACK_LANGUAGE,
    "only_fill_empty": ONLY_FILL_EMPTY,
    "records_in_input": len(data or []),
    "movie_samples_processed": processed_count,
    "movie_samples_matched_tmdb": matched_count,
    "movie_samples_unmatched": len(unmatched_samples),
    "non_movie_records": len(non_movie_records),
    "output_dir": OUTPUT_DIR,
    "enriched_files": enriched_files,
    "unmatched_files": unmatched_files,
    "non_movie_files": non_movie_files,
    "grouping_rule": "record.name se reemplaza por el genero principal (TMDB es-MX). Unmatched queda en 'Sin información' al final.",
    "title_rule": "Se limpia query (brackets/calidad/dual/etc) + 'parte' si antecede a número. Si el titulo actual NO parece español y TMDB trae title_es distinto, se reemplaza; si no, se mantiene.",
    "logo_rule": "icono/iconoHorizontal/iconpng se reemplazan siempre por TMDB si existen (poster/backdrop/logo).",
}

with open(os.path.join(OUTPUT_DIR, f"{stem}.report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
