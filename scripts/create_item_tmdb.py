import os
import re
import json
import time
import argparse
import requests
from datetime import datetime

TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/original"

# --- throttle suave ---
_last_call = 0.0
def _throttle(min_delay_ms=150):
    global _last_call
    now = time.time()
    wait = (min_delay_ms / 1000.0) - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def tmdb_get(api_key: str, path: str, params=None, retries=4):
    url = TMDB_BASE + path
    params = dict(params or {})
    params["api_key"] = api_key

    last_exc = None
    for attempt in range(retries):
        try:
            _throttle(150)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                ra = int(r.headers.get("Retry-After", "2") or "2")
                time.sleep(ra + 1 + attempt)
                continue
            if 500 <= r.status_code < 600:
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)

    raise last_exc if last_exc else RuntimeError("TMDB request failed")

# --- Normalización del nombre para buscar mejor ---
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
BRACKETS_RE = re.compile(r"[\[\(\{].*?[\]\)\}]")
SEP_RE = re.compile(r"[._\-]+")
MULTISPACE_RE = re.compile(r"\s{2,}")
JUNK_RE = re.compile(
    r"(\b(1080p|720p|2160p|4k|hdr|sdr|webrip|web-dl|bluray|brrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|ac3|dts|latino|castellano|subtitulado|sub|dual|multi|esp|eng|vose)\b)",
    re.IGNORECASE,
)

def extract_year(text: str) -> str:
    yrs = YEAR_RE.findall(text or "")
    return yrs[-1] if yrs else ""

def clean_title(name: str) -> str:
    t = (name or "").strip()
    t = BRACKETS_RE.sub(" ", t)
    t = SEP_RE.sub(" ", t)
    t = JUNK_RE.sub(" ", t)
    t = re.sub(r"\s+\b(19\d{2}|20\d{2})\b\s*$", "", t).strip()
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def search_movie(api_key: str, query: str, year: str, lang: str):
    params = {"language": lang, "query": query, "include_adult": "false"}
    if year:
        params["year"] = year
    res = tmdb_get(api_key, "/search/movie", params=params)
    return res.get("results", []) or []

def fetch_details(api_key: str, movie_id: int, lang: str):
    det = tmdb_get(api_key, f"/movie/{movie_id}", params={"language": lang})
    imgs = tmdb_get(api_key, f"/movie/{movie_id}/images", params={"include_image_language": "es,en,null"})
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

def norm_group(s: str) -> str:
    # normaliza espacios y compara sin mayúsculas/minúsculas
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

def load_items(items_path: str):
    if not os.path.exists(items_path):
        return []
    try:
        with open(items_path, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []

def save_items(items_path: str, items):
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

def consolidate_groups(items):
    """
    Une grupos duplicados por name (normalizado) y deduplica samples por url dentro del grupo.
    Mantiene el orden de aparición del primer grupo.
    """
    merged = {}
    order = []

    for g in items:
        k = norm_group(g.get("name"))
        if not k:
            continue
        if k not in merged:
            merged[k] = {"name": g.get("name") or "", "samples": []}
            order.append(k)
        merged[k]["samples"].extend(g.get("samples") or [])

    items_out = []
    for k in order:
        g = merged[k]
        seen = set()
        dedup_rev = []
        # último gana (por si hubo repetidos históricos)
        for sm in reversed(g["samples"]):
            u = (sm.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            dedup_rev.append(sm)
        g["samples"] = list(reversed(dedup_rev))
        items_out.append(g)

    return items_out

def upsert_sample(items, grupo: str, video_url: str, sample: dict):
    """
    - Si la URL existe en cualquier grupo: la elimina (mover/actualizar)
    - Inserta/actualiza en el grupo destino
    - IMPORTANTE: el sample nuevo/actualizado queda SIEMPRE en samples[0]
    - Si el grupo es nuevo, se crea en items[0]
    """
    target_key = norm_group(grupo)
    replaced = False

    # 1) eliminar URL existente de cualquier grupo
    found = None
    for gi, g in enumerate(items):
        smps = g.get("samples") or []
        for si, sm in enumerate(smps):
            if (sm.get("url") or "").strip() == video_url:
                found = (gi, si)
                break
        if found:
            break

    if found:
        old_gi, old_si = found
        old_group = items[old_gi]
        old_samples = old_group.get("samples") or []
        if 0 <= old_si < len(old_samples):
            old_samples.pop(old_si)
        old_group["samples"] = old_samples
        if not old_group["samples"]:
            items.pop(old_gi)
        replaced = True

    # 2) buscar grupo destino
    target_index = None
    for gi, g in enumerate(items):
        if norm_group(g.get("name")) == target_key:
            target_index = gi
            break

    # 3) insertar/actualizar en destino (SIEMPRE AL INICIO)
    if target_index is None:
        # Grupo no existe -> crear y ponerlo primero
        items.insert(0, {"name": grupo, "samples": [sample]})
    else:
        smps = items[target_index].get("samples") or []

        # ¿Ya estaba esta URL en este grupo?
        existing_index = None
        for si, sm in enumerate(smps):
            if (sm.get("url") or "").strip() == video_url:
                existing_index = si
                break

        if existing_index is None:
            # Nuevo sample -> al inicio
            smps.insert(0, sample)
        else:
            # Update -> reemplazar y mover arriba
            smps[existing_index] = sample
            if existing_index != 0:
                smps.pop(existing_index)
                smps.insert(0, sample)
            replaced = True

        items[target_index]["samples"] = smps

        # (OPCIONAL) Si querés que el grupo "actualizado" suba a items[0], descomentá:
        # g = items.pop(target_index)
        # items.insert(0, g)

    # 4) consolidar por si ya venía roto
    items = consolidate_groups(items)
    return items, replaced

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-url", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--grupo", required=True)
    ap.add_argument("--output-dir", default="tmdb_manual_create")
    ap.add_argument("--language", default="es-MX")
    ap.add_argument("--fallback-language", default="en-US")
    args = ap.parse_args()

    api_key = (os.environ.get("TMDB_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Falta TMDB_API_KEY en env (secrets.TMDB_API_KEY)")

    video_url = args.video_url.strip()
    nombre = args.nombre.strip()
    grupo = args.grupo.strip()
    output_dir = (args.output_dir or "tmdb_manual_create").strip()
    language = (args.language or "es-MX").strip()
    fallback_language = (args.fallback_language or "en-US").strip()

    if not video_url or not nombre or not grupo:
        raise SystemExit("Faltan parámetros: video_url, nombre, grupo")

    os.makedirs(output_dir, exist_ok=True)

    # --- Construir item base ---
    orig_year = extract_year(nombre)
    record = {
        "name": grupo,
        "samples": [{
            "name": nombre,
            "url": video_url,
            "icono": "",
            "iconoHorizontal": "",
            "iconpng": "",
            "type": "PELICULA",
            "descripcion": "",
            "anio": orig_year,
            "genero": "",
            "duracion": ""
        }]
    }

    # --- Enriquecer con TMDB ---
    q = clean_title(nombre)
    year_for_search = orig_year

    results = search_movie(api_key, q, year_for_search, language)
    if not results and year_for_search:
        results = search_movie(api_key, q, "", language)
    if not results:
        results = search_movie(api_key, q, year_for_search, "en-US")
    if not results and year_for_search:
        results = search_movie(api_key, q, "", "en-US")

    enriched = False
    if results:
        movie = results[0]
        mid = movie.get("id")
        if mid:
            det, imgs = fetch_details(api_key, mid, language)

            det_fb = None
            if fallback_language and fallback_language != language:
                try:
                    det_fb, _ = fetch_details(api_key, mid, fallback_language)
                except Exception:
                    det_fb = None

            title = (det.get("title") or "").strip()
            overview = (det.get("overview") or "").strip()
            if det_fb:
                if not title:
                    title = (det_fb.get("title") or "").strip()
                if not overview:
                    overview = (det_fb.get("overview") or "").strip()

            release_date = (det.get("release_date") or "").strip()
            tmdb_year = release_date[:4] if release_date else ""
            runtime = det.get("runtime") or 0
            genres = [g.get("name", "").strip() for g in (det.get("genres") or []) if g.get("name")]

            poster = to_img_url(det.get("poster_path") or "")
            backdrop = to_img_url(det.get("backdrop_path") or "")
            logo = pick_logo_url(imgs)

            s = record["samples"][0]
            if title:
                s["name"] = title
            if overview:
                s["descripcion"] = overview
            if tmdb_year:
                s["anio"] = tmdb_year
            if genres:
                s["genero"] = ", ".join(genres)
            if runtime:
                s["duracion"] = f"{int(runtime)} min"
            if poster:
                s["icono"] = poster
            if backdrop:
                s["iconoHorizontal"] = backdrop
            if logo:
                s["iconpng"] = logo

            enriched = True

    # --- Guardar/actualizar items.json (sin duplicar grupos) ---
    items_path = os.path.join(output_dir, "items.json")
    items = load_items(items_path)
    items = consolidate_groups(items)  # por si ya está roto de antes

    items, replaced = upsert_sample(
        items=items,
        grupo=grupo,
        video_url=video_url,
        sample=record["samples"][0]
    )
    save_items(items_path, items)

    # Guardar "ultimo creado"
    last_path = os.path.join(output_dir, "last_created.json")
    with open(last_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    samples_total = sum(len(g.get("samples") or []) for g in items)

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "input": {"url": video_url, "nombre": nombre, "grupo": grupo},
        "enriched": enriched,
        "language": language,
        "fallback_language": fallback_language,
        "groups_total": len(items),
        "samples_total": samples_total,
        "replaced_existing_url": replaced,
        "output_dir": output_dir
    }
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
