import os
import re
import json
import requests
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# ENV / CONFIG
# =========================
M3U_URL = os.environ.get("M3U_URL", "").strip()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "m3u_convert").strip()
CHUNK_SIZE = int((os.environ.get("CHUNK_SIZE") or "2000").strip())

VERIFY_URLS = (os.environ.get("VERIFY_URLS", "true").strip().lower() == "true")
VERIFY_TIMEOUT = float(os.environ.get("VERIFY_TIMEOUT") or "8")
VERIFY_WORKERS = int(os.environ.get("VERIFY_WORKERS") or "25")

if not M3U_URL:
    raise SystemExit("Falta M3U_URL")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Helpers: URL|headers
# =========================
def split_url_pipe_headers(raw_url: str):
    url = raw_url.strip()
    headers = {}
    if "|" not in url:
        return url, headers
    base, meta = url.split("|", 1)
    for part in meta.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                headers[k] = v
    return base.strip(), headers

# =========================
# Extract year
# =========================
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

def extract_year(title: str) -> str:
    if not title:
        return ""
    yrs = YEAR_RE.findall(title)
    return yrs[-1] if yrs else ""

# =========================
# Skip CAM only as standalone token
# =========================
CAM_TOKEN_RE = re.compile(r"\bCAM\b", re.IGNORECASE)

def should_skip_title(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    return bool(CAM_TOKEN_RE.search(t))

# =========================
# Parse EXTINF / EXTVLCOPT
# =========================
ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')

def parse_extinf(line: str):
    if "," in line:
        left, title = line.split(",", 1)
    else:
        left, title = line, ""
    attrs = dict(ATTR_RE.findall(left))
    return attrs, (title or "").strip()

def parse_extvlcopt(line: str):
    prefix = "#EXTVLCOPT:"
    if not line.startswith(prefix):
        return None, None
    rest = line[len(prefix):].strip()
    if "=" not in rest:
        return None, None
    k, v = rest.split("=", 1)
    return k.strip().lower(), v.strip()

# =========================
# Strict URL verification
# =========================
def build_verify_headers(user_agent: str, referer: str):
    headers = {
        "User-Agent": user_agent or "Mozilla/5.0",
        "Accept": "*/*",
        "Connection": "close",
    }
    if referer:
        headers["Referer"] = referer
    return headers

def _guess_kind_from_url(url: str) -> str:
    u = (url or "").lower()
    if ".m3u8" in u:
        return "m3u8"
    if ".mpd" in u:
        return "mpd"
    if any(ext in u for ext in [".mp4", ".m4v", ".mov"]):
        return "mp4"
    if ".mkv" in u:
        return "mkv"
    if ".webm" in u:
        return "webm"
    if ".ts" in u:
        return "ts"
    return "unknown"

def _looks_like_media_bytes(kind: str, b: bytes) -> bool:
    if not b:
        return False
    head = b[:256].lstrip()
    if kind == "m3u8":
        return head.startswith(b"#EXTM3U")
    if kind == "mpd":
        return head.startswith(b"<?xml") or head.startswith(b"<MPD") or (b"<MPD" in head[:220])
    if kind == "ts":
        return b[:1] == b"\x47"
    if kind == "mkv":
        return b.startswith(b"\x1A\x45\xDF\xA3")
    if kind == "mp4":
        return (b.find(b"ftyp", 0, 32) != -1)
    if kind == "webm":
        return b.startswith(b"\x1A\x45\xDF\xA3")
    return True

def _is_probably_html(content_type: str, b: bytes) -> bool:
    ct = (content_type or "").lower()
    if "text/html" in ct:
        return True
    if ("text/" in ct or "application/xhtml" in ct) and b:
        s = b[:200].lstrip().lower()
        return s.startswith(b"<!doctype html") or s.startswith(b"<html") or (b"<html" in s)
    if b:
        s = b[:200].lstrip().lower()
        if s.startswith(b"<!doctype html") or s.startswith(b"<html") or (b"<html" in s):
            return True
    return False

def is_url_online_strict(url: str, headers: dict) -> bool:
    if not url:
        return False

    kind = _guess_kind_from_url(url)
    headers = dict(headers or {})
    headers.setdefault("User-Agent", "Mozilla/5.0")
    headers.setdefault("Accept", "*/*")
    headers.setdefault("Connection", "close")

    try:
        r = requests.head(url, allow_redirects=True, timeout=VERIFY_TIMEOUT, headers=headers)
        code = r.status_code
        if code in (404, 410):
            return False
        if code in (401, 403, 405):
            return False
    except requests.RequestException:
        pass

    try:
        headers2 = dict(headers)
        headers2["Range"] = "bytes=0-1023"
        r2 = requests.get(url, allow_redirects=True, timeout=VERIFY_TIMEOUT, headers=headers2, stream=True)
        if r2.status_code not in (200, 206):
            return False

        ct = r2.headers.get("Content-Type", "")
        chunk = b""
        for part in r2.iter_content(chunk_size=1024):
            chunk = part or b""
            break

        if _is_probably_html(ct, chunk):
            return False

        if kind != "unknown":
            return _looks_like_media_bytes(kind, chunk)

        return True
    except requests.RequestException:
        return False

# =========================
# SERIES detection (improved, NO quality-only)
# =========================
QUALITY_TOKENS_RE = re.compile(
    r"\b(480p|720p|1080p|2160p|4k|hdr|sdr|webrip|web\-dl|bluray|brrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|ac3|dts|latino|castellano|subtitulado|sub|dual|multi|esp|eng|vose)\b",
    re.IGNORECASE
)

SERIES_GROUP_HINTS = [
    "temporada", "season", "serie", "series", "tv", "episodios", "capitulos", "capítulos",
    "capitulo", "capítulo"
]

# SEÑALES FUERTES (si aparece -> es serie)
STRONG_PATTERNS = [
    re.compile(r"\b\d{1,2}x\d{1,3}\b", re.IGNORECASE),                  # 1x01
    re.compile(r"\bs\d{1,2}\s*e\d{1,3}\b", re.IGNORECASE),              # S01E02
    re.compile(r"\bcap(?:itulo|ítulo)?\.?\s*\d+\b", re.IGNORECASE),     # capitulo 10
    re.compile(r"\bepisodio\.?\s*\d+\b", re.IGNORECASE),                # episodio 1
    re.compile(r"\bepisode\.?\s*\d+\b", re.IGNORECASE),                 # episode 1
    re.compile(r"\bep\.?\s*0*\d+\b", re.IGNORECASE),                    # ep01
    re.compile(r"\be\.?\s*0*\d+\b", re.IGNORECASE),                     # e01
    re.compile(r"\b(temporada|season|temp)\s*\d+\b", re.IGNORECASE),     # temporada 1
    re.compile(r"(^|[\s._-])([st])\s*0?\d{1,2}($|[\s._-])", re.IGNORECASE),  # S01 / T01 token
]

LEADING_EP_NUMBER = re.compile(r"^\s*\d{1,3}\s*[-–._)]\s+\S+", re.IGNORECASE)

# SEÑALES DÉBILES (solo sirven si el título base se repite)
YEAR_THEN_NUMBER_RE = re.compile(r"\(\s*(19\d{2}|20\d{2})\s*\)\s*0?\d{1,3}\b", re.IGNORECASE)
TRAILING_NUMBER_RE = re.compile(r"\s0?\d{1,3}\s*$", re.IGNORECASE)
NUMBER_BEFORE_QUALITY_RE = re.compile(
    r"\s0?\d{1,3}\s+(?=(?:480p|720p|1080p|2160p|4k|hdr|sdr|webrip|web\-dl|bluray|brrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|ac3|dts|latino|castellano|subtitulado|sub|dual|multi|esp|eng|vose)\b)",
    re.IGNORECASE
)

SEP_RE = re.compile(r"[._\-]+")
MULTISPACE_RE = re.compile(r"\s{2,}")

def episode_hints(title: str, group_title: str):
    t = (title or "").strip()
    gt = (group_title or "").strip().lower()
    if not t:
        return (False, False)

    # group-title con hints -> fuerte
    for hint in SERIES_GROUP_HINTS:
        if hint in gt:
            return (True, False)

    tl = t.lower()

    # fuerte por patrones
    if LEADING_EP_NUMBER.search(t):
        return (True, False)

    for p in STRONG_PATTERNS:
        if p.search(t):
            return (True, False)

    # debil: "(2023) 06 ..." o " ... 06 720p ..."
    if YEAR_THEN_NUMBER_RE.search(t):
        return (False, True)

    if NUMBER_BEFORE_QUALITY_RE.search(t):
        return (False, True)

    # debil: termina con numero (sin otras pistas)
    # OJO: esto solo se usa por repeticion
    if TRAILING_NUMBER_RE.search(t):
        # evitamos confundir años solos
        only = t.strip()
        if not re.fullmatch(r"(19\d{2}|20\d{2})", only):
            return (False, True)

    return (False, False)

def series_base_key(title: str) -> str:
    """
    Titulo base para agrupar repeticiones.
    Quita calidad/idioma, y quita marcadores típicos de episodios,
    pero deja el año (2024) para diferenciar remakes.
    """
    t = (title or "").strip().lower()
    if not t:
        return ""

    # quitar calidad/idioma
    t = QUALITY_TOKENS_RE.sub(" ", t)

    # normalizar separadores
    t = SEP_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()

    # quitar patrones fuertes de episodios
    t = re.sub(r"\b\d{1,2}x\d{1,3}\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bs\d{1,2}\s*e\d{1,3}\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(ep|e)\.?\s*0*\d+\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(episodio|episode|capitulo|capítulo|cap)\.?\s*\d+\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(temporada|season|temp)\s*\d+\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b([st])\s*0?\d{1,2}\b", " ", t, flags=re.IGNORECASE)

    # caso "(2023) 06" -> remover el 06 pero mantener el (2023)
    t = re.sub(r"(\(\s*(?:19\d{2}|20\d{2})\s*\))\s*0?\d{1,3}\b", r"\1", t, flags=re.IGNORECASE)

    # si termina con numero (débil), lo quitamos para base
    t = re.sub(r"\s0?\d{1,3}\s*$", " ", t)

    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

# =========================
# Download & parse M3U
# =========================
resp = requests.get(M3U_URL, timeout=120)
resp.raise_for_status()
text = resp.text

lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

all_entries = []
current_extinf = None
current_attrs = None
current_title = None
current_opts_lines = []
current_user_agent = ""
current_referer = ""

for ln in lines:
    if ln.startswith("#EXTM3U"):
        continue

    if ln.startswith("#EXTINF"):
        current_extinf = ln
        current_attrs, current_title = parse_extinf(ln)
        current_opts_lines = []
        current_user_agent = ""
        current_referer = ""
        continue

    if current_extinf and ln.startswith("#EXTVLCOPT:"):
        k, v = parse_extvlcopt(ln)
        if k and v:
            current_opts_lines.append(ln)
            if k == "http-user-agent":
                current_user_agent = v
            elif k == "http-referrer":
                current_referer = v
        continue

    if current_extinf and not ln.startswith("#"):
        raw_url = ln.strip()
        video_url, pipe_headers = split_url_pipe_headers(raw_url)

        if not current_user_agent and ("User-Agent" in pipe_headers or "user-agent" in pipe_headers):
            current_user_agent = pipe_headers.get("User-Agent") or pipe_headers.get("user-agent") or ""
        if not current_referer and ("Referer" in pipe_headers or "referer" in pipe_headers):
            current_referer = pipe_headers.get("Referer") or pipe_headers.get("referer") or ""

        group_title = (current_attrs.get("group-title") or "").strip()
        tvg_logo = (current_attrs.get("tvg-logo") or "").strip()

        title = (current_title or "").strip()
        if not title:
            title = (current_attrs.get("tvg-name") or current_attrs.get("tvg-id") or "").strip()

        # --- FILTRO CAM ---
        if should_skip_title(title):
            current_extinf = None
            current_attrs = None
            current_title = None
            current_opts_lines = []
            current_user_agent = ""
            current_referer = ""
            continue

        all_entries.append({
            "extinf": current_extinf,
            "opts_lines": list(current_opts_lines),
            "title": title,
            "url": video_url,
            "group_title": group_title,
            "tvg_logo": tvg_logo,
            "anio": extract_year(title),
            "user_agent": current_user_agent,
            "referer": current_referer,
        })

        current_extinf = None
        current_attrs = None
        current_title = None
        current_opts_lines = []
        current_user_agent = ""
        current_referer = ""

# =========================
# Verify URLs (optional)
# =========================
if VERIFY_URLS and all_entries:
    keys = []
    seen = set()
    for e in all_entries:
        key = (e["url"], e.get("user_agent") or "", e.get("referer") or "")
        if key not in seen:
            seen.add(key)
            keys.append(key)

    print(f"Verificando URLs: {len(keys)} unicas (workers={VERIFY_WORKERS}, timeout={VERIFY_TIMEOUT}s)")

    results = {}
    with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as ex:
        fut_map = {}
        for (url, ua, ref) in keys:
            headers = build_verify_headers(ua, ref)
            fut_map[ex.submit(is_url_online_strict, url, headers)] = (url, ua, ref)

        done = 0
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            ok = False
            try:
                ok = bool(fut.result())
            except Exception:
                ok = False
            results[key] = ok
            done += 1
            if done % 300 == 0:
                print(f"  progreso: {done}/{len(keys)}")

    filtered = []
    for e in all_entries:
        key = (e["url"], e.get("user_agent") or "", e.get("referer") or "")
        if results.get(key, False):
            filtered.append(e)

    dropped = len(all_entries) - len(filtered)
    print(f"URLs offline ignoradas: {dropped}")
    all_entries = filtered

# =========================
# Series vs Movies (2-pass)
# - Pass 1: STRONG hints -> series
# - Pass 2: repetition family + WEAK hints -> series
# =========================
base_counts = defaultdict(int)
base_has_any_hint = defaultdict(bool)   # strong OR weak
entry_meta = []  # (entry, strong, weak, base_key)

for e in all_entries:
    strong, weak = episode_hints(e["title"], e["group_title"])
    base_key = series_base_key(e["title"])
    base_counts[base_key] += 1
    if strong or weak:
        base_has_any_hint[base_key] = True
    entry_meta.append((e, strong, weak, base_key))

movies_entries = []
series_entries = []

for (e, strong, weak, base_key) in entry_meta:
    # fuerte => serie
    if strong:
        series_entries.append(e)
        continue

    # debil SOLO si el base se repite (01,02,03...)
    if weak and base_key and base_counts[base_key] >= 2 and base_has_any_hint[base_key]:
        series_entries.append(e)
        continue

    # si no, pelicula
    movies_entries.append(e)

# =========================
# Write M3U outputs
# =========================
def write_m3u(path, entries):
    out = ["#EXTM3U"]
    for e in entries:
        out.append(e["extinf"])
        for opt in e.get("opts_lines", []):
            out.append(opt)
        out.append(e["url"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

write_m3u(os.path.join(OUTPUT_DIR, "peliculas.m3u"), movies_entries)
write_m3u(os.path.join(OUTPUT_DIR, "series.m3u"), series_entries)

# =========================
# JSON outputs (all fields)
# =========================
def sample_from_entry(e, tipo: str):
    return {
        "name": e["title"],
        "url": e["url"],
        "icono": e["tvg_logo"] or "",
        "iconoHorizontal": "",
        "iconpng": "",
        "type": tipo,
        "descripcion": "",
        "anio": e.get("anio") or "",
        "genero": "",
        "duracion": ""
    }

def to_grouped_records(entries, tipo: str):
    groups = OrderedDict()
    for e in entries:
        g = e["group_title"]
        if g not in groups:
            groups[g] = []
        groups[g].append(sample_from_entry(e, tipo))
    return [{"name": g, "samples": samples} for g, samples in groups.items()]

movies_records = to_grouped_records(movies_entries, "PELICULA")
series_records = to_grouped_records(series_entries, "SERIE")

def split_records_by_samples(records, max_samples):
    parts = []
    cur = []
    cur_count = 0
    for r in records:
        s = r.get("samples", [])
        if cur and (cur_count + len(s) > max_samples):
            parts.append(cur)
            cur = []
            cur_count = 0
        cur.append(r)
        cur_count += len(s)
    if cur:
        parts.append(cur)
    return parts

parts = split_records_by_samples(movies_records, CHUNK_SIZE)

if len(parts) == 1:
    with open(os.path.join(OUTPUT_DIR, "peliculas.json"), "w", encoding="utf-8") as f:
        json.dump(parts[0], f, indent=2, ensure_ascii=False)
else:
    for i, part in enumerate(parts, start=1):
        with open(os.path.join(OUTPUT_DIR, f"peliculas_part{i:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(part, f, indent=2, ensure_ascii=False)
    manifest = {
        "total_parts": len(parts),
        "chunk_size_samples": CHUNK_SIZE,
        "files": [f"peliculas_part{i:03d}.json" for i in range(1, len(parts) + 1)]
    }
    with open(os.path.join(OUTPUT_DIR, "peliculas_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "series.json"), "w", encoding="utf-8") as f:
    json.dump(series_records, f, indent=2, ensure_ascii=False)

print(f"M3U URL: {M3U_URL}")
print(f"VERIFY_URLS: {VERIFY_URLS}")
print(f"Peliculas: {len(movies_entries)} entries")
print(f"Series: {len(series_entries)} entries")
print(f"Peliculas JSON parts: {len(parts)} (chunk_size={CHUNK_SIZE})")
print(f"Salida en: {OUTPUT_DIR}")
