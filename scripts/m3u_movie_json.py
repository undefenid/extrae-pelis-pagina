import os
import re
import json
import requests
from collections import OrderedDict
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
# Strict URL verification (más anti-falsos positivos)
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
    if ".m3u" in u:
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

    if ".avi" in u:
    return "avi"
    
    return "unknown"

def _is_probably_html(content_type: str, b: bytes) -> bool:
    ct = (content_type or "").lower()
    if "text/html" in ct:
        return True
    if ("text/" in ct or "application/xhtml" in ct) and b:
        s = b[:400].lstrip().lower()
        return s.startswith(b"<!doctype html") or s.startswith(b"<html") or (b"<html" in s)
    if b:
        s = b[:400].lstrip().lower()
        if s.startswith(b"<!doctype html") or s.startswith(b"<html") or (b"<html" in s):
            return True
    return False

def _detect_kind_from_bytes(b: bytes) -> str:
    if not b:
        return "unknown"
    head = b[:4096].lstrip()

    # playlist m3u8
    if head.startswith(b"#EXTM3U"):
        return "m3u8"

    # dash mpd
    if head.startswith(b"<?xml") or head.startswith(b"<MPD") or (b"<MPD" in head[:1200]):
        return "mpd"

    # mkv/webm (EBML)
    if head.startswith(b"\x1A\x45\xDF\xA3"):
        return "mkv"

     # AVI: "RIFF" .... "AVI "
    if head.startswith(b"RIFF") and b"AVI " in head[8:16]:
        return "avi"

    # ts sync byte
    if b[:1] == b"\x47":
        return "ts"

    # mp4 signature (ftyp en primeros 32)
    if b.find(b"ftyp", 0, 64) != -1:
        return "mp4"

    return "unknown"

def _looks_like_media_bytes(kind: str, b: bytes) -> bool:
    if not b:
        return False
    head = b[:4096].lstrip()

    if kind == "m3u8":
        return head.startswith(b"#EXTM3U")
    if kind == "mpd":
        return head.startswith(b"<?xml") or head.startswith(b"<MPD") or (b"<MPD" in head[:1200])
    if kind == "ts":
        return b[:1] == b"\x47"
    if kind in ("mkv", "webm"):
        return head.startswith(b"\x1A\x45\xDF\xA3")
    if kind == "mp4":
        return (b.find(b"ftyp", 0, 64) != -1)

    if kind == "avi":
    return b.startswith(b"RIFF") and (b"AVI " in b[8:16])

    # unknown: NO aceptar por defecto (evita falsos positivos)
    detected = _detect_kind_from_bytes(b)
    return detected != "unknown"

def is_url_online_strict(url: str, headers: dict):
    """
    Devuelve (ok: bool, info: dict)
    info incluye: reason, status_code, content_type, final_url, kind_guess, kind_detected
    """
    info = {
        "reason": "",
        "status_code": None,
        "content_type": "",
        "final_url": "",
        "kind_guess": _guess_kind_from_url(url),
        "kind_detected": "unknown",
    }

    if not url:
        info["reason"] = "empty_url"
        return False, info

    headers = dict(headers or {})
    headers.setdefault("User-Agent", "Mozilla/5.0")
    headers.setdefault("Accept", "*/*")
    headers.setdefault("Connection", "close")

    # 1) HEAD (rápido)
    try:
        r = requests.head(url, allow_redirects=True, timeout=VERIFY_TIMEOUT, headers=headers)
        info["status_code"] = r.status_code
        info["final_url"] = getattr(r, "url", "") or ""
        ct = r.headers.get("Content-Type", "") if hasattr(r, "headers") else ""
        info["content_type"] = ct or ""

        if r.status_code in (404, 410):
            info["reason"] = f"head_{r.status_code}"
            return False, info
        if r.status_code in (401, 403):
            info["reason"] = f"head_{r.status_code}"
            return False, info
        # 405: método no permitido -> seguimos con GET
    except requests.RequestException:
        # sin head, seguimos con GET
        pass

    # 2) GET con Range (validación de bytes)
    try:
        headers2 = dict(headers)
        headers2["Range"] = "bytes=0-4095"
        r2 = requests.get(url, allow_redirects=True, timeout=VERIFY_TIMEOUT, headers=headers2, stream=True)

        info["status_code"] = r2.status_code
        info["final_url"] = getattr(r2, "url", "") or info["final_url"]
        info["content_type"] = r2.headers.get("Content-Type", "") or info["content_type"]

        if r2.status_code not in (200, 206):
            info["reason"] = f"get_status_{r2.status_code}"
            return False, info

        chunk = b""
        for part in r2.iter_content(chunk_size=4096):
            chunk = part or b""
            break

        if not chunk:
            info["reason"] = "empty_body"
            return False, info

        # anti-html
        if _is_probably_html(info["content_type"], chunk):
            info["reason"] = "html_detected"
            return False, info

        # detect por bytes (más estricto)
        info["kind_detected"] = _detect_kind_from_bytes(chunk)

        # si el tipo es conocido por URL, exigimos match por bytes
        kind = info["kind_guess"]
        if kind != "unknown":
            ok = _looks_like_media_bytes(kind, chunk)
            info["reason"] = "" if ok else f"bytes_not_{kind}"
            return ok, info

        # si es unknown por URL, NO aceptamos por defecto: solo si detectamos algo real
        ok = _looks_like_media_bytes("unknown", chunk)
        info["reason"] = "" if ok else "unknown_bytes"
        return ok, info

    except requests.RequestException as ex:
        info["reason"] = f"request_error"
        return False, info

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
            "title": title,                   # ✅ nombre completo tal cual
            "url": video_url,
            "group_title": group_title,       # ✅ grupo tal cual
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
# Verify URLs (optional) + OFFLINE REPORT
# =========================
offline_items = []
verify_details = {}  # key -> info dict

if VERIFY_URLS and all_entries:
    keys = []
    seen = set()
    for e in all_entries:
        key = (e["url"], e.get("user_agent") or "", e.get("referer") or "")
        if key not in seen:
            seen.add(key)
            keys.append(key)

    print(f"Verificando URLs: {len(keys)} unicas (workers={VERIFY_WORKERS}, timeout={VERIFY_TIMEOUT}s)")

    results_ok = {}
    with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as ex:
        fut_map = {}
        for (url, ua, ref) in keys:
            headers = build_verify_headers(ua, ref)
            fut_map[ex.submit(is_url_online_strict, url, headers)] = (url, ua, ref)

        done = 0
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            ok = False
            info = {"reason": "unknown_error"}
            try:
                ok, info = fut.result()
            except Exception:
                ok = False
                info = {"reason": "exception_in_worker"}

            results_ok[key] = bool(ok)
            verify_details[key] = info
            done += 1
            if done % 300 == 0:
                print(f"  progreso: {done}/{len(keys)}")

    filtered = []
    for e in all_entries:
        key = (e["url"], e.get("user_agent") or "", e.get("referer") or "")
        if results_ok.get(key, False):
            filtered.append(e)
        else:
            info = verify_details.get(key, {}) or {}
            offline_items.append({
                "name": e["title"],
                "group": e["group_title"],
                "url": e["url"],
                "user_agent": e.get("user_agent") or "",
                "referer": e.get("referer") or "",
                "reason": info.get("reason", ""),
                "status_code": info.get("status_code", None),
                "content_type": info.get("content_type", ""),
                "final_url": info.get("final_url", ""),
                "kind_guess": info.get("kind_guess", ""),
                "kind_detected": info.get("kind_detected", ""),
            })

    dropped = len(all_entries) - len(filtered)
    print(f"URLs offline ignoradas (pero reportadas): {dropped}")
    all_entries = filtered

# siempre escribir reporte (aunque esté vacío o verify=false)
offline_json_path = os.path.join(OUTPUT_DIR, "offline_urls.json")
offline_txt_path  = os.path.join(OUTPUT_DIR, "offline_urls.txt")

with open(offline_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "m3u_url": M3U_URL,
        "verify_urls": VERIFY_URLS,
        "verify_timeout": VERIFY_TIMEOUT,
        "verify_workers": VERIFY_WORKERS,
        "offline_count": len(offline_items),
        "items": offline_items
    }, f, indent=2, ensure_ascii=False)

with open(offline_txt_path, "w", encoding="utf-8") as f:
    f.write(f"M3U: {M3U_URL}\n")
    f.write(f"VERIFY_URLS: {VERIFY_URLS}\n")
    f.write(f"OFFLINE_COUNT: {len(offline_items)}\n\n")
    for it in offline_items:
        f.write(f"- [{it.get('group','')}] {it.get('name','')}\n")
        f.write(f"  url: {it.get('url','')}\n")
        reason = it.get("reason","")
        sc = it.get("status_code", None)
        ct = it.get("content_type","")
        f.write(f"  reason: {reason} | status: {sc} | ct: {ct}\n\n")

# =========================
# Write M3U output (TODO junto, sin series)
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

write_m3u(os.path.join(OUTPUT_DIR, "peliculas.m3u"), all_entries)

# =========================
# JSON outputs (todo junto, nombre/grupo tal cual)
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

records = to_grouped_records(all_entries, "PELICULA")

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

parts = split_records_by_samples(records, CHUNK_SIZE)

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

print(f"M3U URL: {M3U_URL}")
print(f"VERIFY_URLS: {VERIFY_URLS}")
print(f"Items (online): {sum(len(r['samples']) for r in records)}")
print(f"JSON parts: {len(parts)} (chunk_size={CHUNK_SIZE})")
print(f"Offline report: {offline_json_path} / {offline_txt_path}")
print(f"Salida en: {OUTPUT_DIR}")
