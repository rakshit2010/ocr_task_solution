# app_patched_final.py
# Full-text-first OCR pipeline with heuristic barcode recovery integrated.

import streamlit as st
import os
import tempfile
import json
import traceback
import cv2
import numpy as np
import re
import math

from src.preprocessing import (to_gray, clahe_eq, denoise_bilateral, fast_denoise,
                           binarize_adaptive, binarize_otsu, morphological_process, upscale as do_upscale)
from src.ocr_engine import ocr_on_crop_variants
from src.utils import save_json

st.set_page_config(layout='wide', page_title='Shipping Label OCR - Full-text scan')
st.title("Shipping Label OCR — scan all text, then search for `_1` pattern")

uploaded = st.file_uploader("Upload a shipping label image", type=["png","jpg","jpeg","tif"])
show_debug = st.checkbox("Show debug images & OCR candidates", value=False)
upscale_preview = st.sidebar.slider("Upscale factor for preview/preprocessing", 1, 3, 2)
st.sidebar.info("Increase if suffix is tiny. Larger values slower.")

# ----------------- helpers -----------------
def rotate_image(img, angle_clockwise):
    if img is None:
        return None
    angle_ccw = -angle_clockwise
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_ccw, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos)); new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]; M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def save_cv2_image(img, path):
    ext = os.path.splitext(path)[1]
    if ext == "": path = path + ".jpg"
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or '.jpg', img)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return cv2.imwrite(path, img)

# safer show_image that downscales for display to avoid Pillow/Streamlit issues
MAX_DISPLAY_DIM = 2200  # max side length to display (pixels)

def show_image(img, caption=None, width=420):
    if img is None:
        return
    out = img
    # grayscale image (2D)
    if len(img.shape) == 2:
        H, W = img.shape[:2]
        if max(H, W) > MAX_DISPLAY_DIM:
            scale = MAX_DISPLAY_DIM / float(max(H, W))
            out = cv2.resize(img, (max(1, int(W*scale)), max(1, int(H*scale))), interpolation=cv2.INTER_AREA)
        st.image(out, caption=caption, width=width, clamp=True)
    else:
        H, W = img.shape[:2]
        if max(H, W) > MAX_DISPLAY_DIM:
            scale = MAX_DISPLAY_DIM / float(max(H, W))
            out = cv2.resize(img, (max(1, int(W*scale)), max(1, int(H*scale))), interpolation=cv2.INTER_AREA)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=caption, width=width)

# Orientation detection
def detect_orientation_with_tesseract(img_bgr):
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(img_bgr)
        m = re.search(r'Rotate:\s+(\d+)', osd)
        if m: return int(m.group(1)) % 360
    except Exception:
        return None
    return None

def detect_orientation_by_contours(gray):
    try:
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        candidates = [c for c in contours if cv2.contourArea(c) > 200]
        if not candidates: return None
        c = max(candidates, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        if angle < -45:
            rot = -(90 + angle)
        else:
            rot = -angle
        if abs(rot) < 2: return 0.0
        return rot % 360
    except Exception:
        return None

def ensure_upright(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    rot = detect_orientation_with_tesseract(img_bgr)
    if rot is not None:
        if rot != 0:
            return rotate_image(img_bgr, rot), rot, f"tesseract"
        return img_bgr, 0, "tesseract"

    rot2 = detect_orientation_by_contours(gray)
    if rot2 is not None:
        if abs(rot2) >= 1:
            return rotate_image(img_bgr, rot2), rot2, f"contours"
        return img_bgr, 0, "contours"

    h,w = img_bgr.shape[:2]
    if h > w:
        return rotate_image(img_bgr, 90), 90, "fallback_90"
    return img_bgr, 0, "none"

def normalize_text_candidate(s):
    if not s: return ""
    s2 = s.strip()
    # early normalizations (keep digits/letters/underscore only)
    s2 = s2.replace("I","1").replace("l","1").replace("O","0").replace("o","0")
    s2 = s2.replace("-","_").replace(" ","_").replace("|","1").replace("[","1").replace("]","1")
    s2 = re.sub(r"\s+","",s2)
    s2 = re.sub(r"[^0-9A-Za-z_]", "", s2)
    # collapse multiple underscores and strip leading/trailing underscores
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return s2

# helper: discard obviously-ambiguous single-letter suffixes
def is_ambiguous_single_letter(s):
    if not s: return False
    s = s.strip()
    if len(s) != 1:
        return False
    # ambiguous set: digit/letter shapes that OCR mixes up
    return bool(re.match(r'^[1Il|]$', s, flags=re.IGNORECASE))
# ===== suffix selection blacklist (words that are not valid suffixes) =====
SUFFIX_BLACKLIST = {
    "VENDOR","VEN","ven","vendor", "SELLER", "SHADOWFAX", "FLYER", "RVP", "SHIPPING", "TRACKING", "ORDER",
    "INVOICE", "ADDRESS", "PHONE", "FAX", "GSTIN"
}
# helper: check if a candidate suffix is derived from or starts-with a blacklisted word
def is_blacklisted_suffix(s):
    if not s:
        return False
    su = s.upper().strip()
    # direct equality or starts-with a blacklisted word (e.g. 'VENDOR' -> 'VEN' or 'VEND')
    for bad in SUFFIX_BLACKLIST:
        if su == bad:
            return True
        # also treat 'VEN', 'VEND' etc derived from 'VENDOR' as blacklisted
        if bad.startswith(su) or su.startswith(bad[:3]) and bad.startswith(su):
            return True
        # if suffix is prefix of a blacklist word (rare), also treat as blacklisted
        if bad.startswith(su):
            return True
    return False

import math

def pick_best_match(matches):
    """
    Strong scoring:
      - huge bonus for explicit `_1_<suffix>` (suffix must not be blacklisted),
      - large penalty if suffix is in SUFFIX_BLACKLIST,
      - prefer sources fusion/easyocr for suffix,
      - then conf, numeric prefix len, token len, lexicographic.
    """
    if not matches:
        return None

    def conf_val(m):
        c = m.get('conf', -1)
        try:
            c = float(c)
            if 0 <= c <= 1:
                c *= 100.0
        except:
            c = -1.0
        return c

    def numeric_prefix_len(m):
        norm = (m.get('norm') or '').upper()
        mnum = re.match(r'^(\d+)_1', norm)
        if mnum:
            return len(mnum.group(1))
        mm = re.match(r'^(\d+)', norm)
        return len(mm.group(1)) if mm else 0

    def extract_suffix(norm):
        if not norm:
            return ""
        norm = str(norm).upper()
        m = re.search(r'_1_([A-Z0-9]{1,3})$', norm)
        return m.group(1) if m else ""

    scored = []
    for m in matches:
        norm = (m.get('norm') or "").upper()
        src = (m.get('source') or "").lower()
        suffix = extract_suffix(norm)
        score = 0.0
        if len(suffix)>3:
            score -= 1000  # invalid suffix length penalty

        
        if len(norm)<14:
            score -= 500  # too short overall penalty   

        # confidence scaled
        score += conf_val(m)

        # numeric prefix length small weight
        score += numeric_prefix_len(m) * 0.4

        # token length small weight
        score += len(norm) * 0.05
        if norm.count("_") > 2:
            score -= 100
        if norm.count("_") == 0:
            score -= 100

        scored.append((score, m, suffix, src))

    # debug table
    if show_debug:
        st.write("DEBUG: scoring candidates (score, norm, suffix, source, conf):")
        for s, m, suf, src in sorted(scored, key=lambda x: -x[0]):
            st.write({
                "score": round(s, 2),
                "norm": m.get('norm'),
                "suffix": suf,
                "source": src,
                "conf": m.get('conf')
            })

    scored_sorted = sorted(scored, key=lambda x: -x[0])
    best = scored_sorted[0][1] if scored_sorted[0][0]>0 else None
    return best

# ----------------- UI logic -----------------
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

# save uploaded bytes to temp file and decode correctly
tmp_dir = tempfile.mkdtemp()
uploaded_path = os.path.join(tmp_dir, uploaded.name)
with open(uploaded_path, "wb") as f:
    f.write(uploaded.getbuffer())

# --- Robust image decoding + auto-downscale to avoid Pillow DecompressionBombError ---
# tune this if you have more memory; 25M pixels is safe default
MAX_PIXELS = 25_000_000  # ~25 megapixels

# decode uploaded bytes into OpenCV image
arr = np.frombuffer(open(uploaded_path, "rb").read(), np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Failed to load image.")
    st.stop()

# check total pixels and downscale if necessary
h, w = img_bgr.shape[:2]
total_pixels = int(h) * int(w)
if total_pixels > MAX_PIXELS:
    scale = math.sqrt(MAX_PIXELS / float(total_pixels))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    st.info(f"Input was very large ({w}x{h}), downscaled to {new_w}x{new_h} for processing.")

# auto-orient
img_bgr, rot_deg, method = ensure_upright(img_bgr)
st.info(f"Orientation fixed using: {method}, rotated: {rot_deg}°")

colL, colR = st.columns([1,1])
with colL:
    if st.button("Rotate left 90°"):
        img_bgr = rotate_image(img_bgr, -90)
with colR:
    if st.button("Rotate right 90°"):
        img_bgr = rotate_image(img_bgr, 90)

st.subheader("Image used for OCR")
show_image(img_bgr, width=720)

# preprocess
try:
    gray = to_gray(img_bgr)
except:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ---------- Improved OCR Preprocessing ----------
gray = clahe_eq(gray)

# strong denoise
gray = cv2.fastNlMeansDenoising(gray, h=12)

# upscale x2 for better readability
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# light sharpen
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
gray = cv2.filter2D(gray, -1, kernel)


if upscale_preview > 1:
    try:
        gray_up = do_upscale(gray, upscale_preview)
    except:
        gray_up = cv2.resize(gray, (gray.shape[1]*upscale_preview, gray.shape[0]*upscale_preview), interpolation=cv2.INTER_CUBIC)
else:
    gray_up = gray

bin1 = binarize_adaptive(gray_up)
bin2 = binarize_otsu(gray_up)
bin1 = morphological_process(bin1)
bin2 = morphological_process(bin2)
preprocessed = bin1 if cv2.countNonZero(bin1) > cv2.countNonZero(bin2) else bin2

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Preprocessed binary (rotated)")
    show_image(preprocessed, width=520)

# ------------------ ONLY GRAYSCALE SHOWN -------------------
with col2:
    st.subheader("Grayscale (OCR input)")
    show_image(gray_up, caption="Grayscale upscaled/CLAHE", width=300)

# Save rotated image
rotated_path = os.path.join(tmp_dir, "rotated_" + uploaded.name)
save_cv2_image(img_bgr, rotated_path)

# extraction trigger
run_btn = st.button("Scan all text and search for pattern")


if not (run_btn ):
    st.stop()

# regex pattern
pattern_text = st.text_input(
    "Regex to search for:",
    value=r"\d{11,}(?:[\s\-_\.]+)1(?:[\s\-_\.]*[A-Za-z]{1,3})?"
)

try:
    pattern = re.compile(pattern_text)
except:
    st.error("Invalid regex.")
    st.stop()

all_candidates = []

# ------------------- OCR: pytesseract full -------------------
try:
    import pytesseract
    # use the enhanced grayscale (gray_up) as Tesseract input
    full_raw = pytesseract.image_to_string(gray_up, config='--oem 3 --psm 6')
    tokens = re.split(r"[\s,;:/]+", full_raw)
    for t in tokens:
        norm = normalize_text_candidate(t)
        if norm:
            all_candidates.append({
                "norm": norm,
                "raw": t,
                "conf": -1,
                "source": "tesseract"
            })
except:
    full_raw = ""

# ------------------- OCR: EasyOCR (optional) -------------------
easy_res = []
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)

    # Prepare a clean natural grayscale input for EasyOCR (avoid over-sharpen/denoise)
    easyocr_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Slight resize to help small text without introducing artifacts
    easyocr_input = cv2.resize(easyocr_input, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

    easy_res = reader.readtext(easyocr_input, detail=0)
    for t in easy_res:
        norm = normalize_text_candidate(t)
        if norm:
            all_candidates.append({
                "norm": norm,
                "raw": t,
                "conf": None,
                "source": "easyocr"
            })
except Exception:
    easy_res = []

# make easy raw tokens list (raw strings from easyocr) - used by repair/fusion logic
easy_raw_tokens = [str(t) for t in easy_res] if easy_res else []

def fuse_text(tess_list, easy_list):
    """
    Character-level fusion:
     - prefer numeric prefix from Tesseract tokens (longest match),
     - prefer short alphabetic suffix from raw EasyOCR tokens,
     - discard suffixes that are in SUFFIX_BLACKLIST or ambiguous single-letter,
     - return fused tokens like '<digits>_1_<SUF>' or '<digits>_1'.
    """
    global easy_raw_tokens

    if not tess_list and not easy_list:
        return []

    fused = []
    all_tokens = set(tess_list + easy_list)

    for token in all_tokens:
        if "_1" not in token:
            continue

        prefix, _, suffix = token.partition("_1")
        prefix = prefix or ""
        suffix = re.sub(r"[^A-Za-z0-9]", "", (suffix or ""))

        # pick a best prefix from tess_list if available (longest)
        best_prefix = max([t for t in tess_list if t.startswith(prefix)], default=prefix)

        # candidate suffix from token itself (clean)
        candidate_suffix = re.sub(r"[^A-Za-z0-9]", "", suffix)
        candidate_suffix = candidate_suffix.upper()

        # try to replace with a short raw-easy suffix if possible
                # try to replace with a short raw-easy suffix if possible
        for raw in easy_raw_tokens:
            raw_letters = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
            if not raw_letters:
                continue
            if 1 <= len(raw_letters) <= 3:
                # accept if it appears near token text or is similar
                if raw_letters in token.upper() or raw_letters in suffix.upper():
                    # reject ambiguous single-letter and blacklist-derived suffixes
                    if is_ambiguous_single_letter(raw_letters) or is_blacklisted_suffix(raw_letters):
                        if show_debug:
                            st.write(f"DEBUG: fuse_text rejected raw suffix '{raw_letters}' (ambiguous/blacklisted)")
                        continue
                    candidate_suffix = raw_letters
                    break

        # final accept: ensure suffix is either empty or 1..3 alpha/numeric but not blacklisted
        if candidate_suffix and (len(candidate_suffix) > 3 or is_blacklisted_suffix(candidate_suffix) or is_ambiguous_single_letter(candidate_suffix)):
            if show_debug:
                st.write(f"DEBUG: fuse_text rejected token-internal suffix '{candidate_suffix}' (len/blacklist/ambig)")
            candidate_suffix = ""

        if candidate_suffix:
            final = best_prefix + "_1_" + candidate_suffix
        else:
            final = best_prefix + "_1"

        fused.append(final)

    return fused


def stabilize_suffix(token):
    """Fix noisy suffix after _1: keep clean letters only and ensure `_1` separator only when suffix exists."""
    if "_1" not in token:
        return token

    prefix, _, suffix = token.partition("_1")

    # clean suffix: keep letters only
    suffix = re.sub(r"[^A-Za-z0-9]+", "", suffix)

    if suffix:
        return prefix + "_1_" + suffix
    else:
        # no suffix found -> keep just `_1` with no trailing underscore
        return prefix + "_1"

# ------------------- Deduplicate -------------------
ded = {}
for c in all_candidates:
    key=c["norm"]
    if not key: continue
    if key not in ded:
        ded[key]=c
    else:
        try:
            if float(c.get("conf",-1)) > float(ded[key].get("conf",-1)):
                ded[key]=c
        except:
            pass

ded_list = list(ded.values())
# ---------- compute fusion IMMEDIATELY so we never read stale fused_tokens ----------
# make sure fused_tokens exists and is fresh for this image
fused_tokens = []

# prepare tokens for fusion
tess_tokens = [d["norm"] for d in ded_list if d.get("source") == "tesseract"]
easy_tokens = [d["norm"] for d in ded_list if d.get("source") == "easyocr"]

# compute fused tokens (fresh)
fused_tokens = fuse_text(tess_tokens, easy_tokens) if (tess_tokens or easy_tokens) else []

if show_debug:
    st.write("DEBUG: fused tokens added (fusion output):", fused_tokens)

# add fused tokens as high-confidence candidates immediately so candidate_pool is consistent
for ft in fused_tokens:
    cleaned = stabilize_suffix(ft)
    if re.match(r"^\d{11,}_1(?:_[A-Za-z]{1,3})?$", cleaned) is None:
        continue
    ded_list.append({
        "norm": cleaned,
        "raw": cleaned,
        "conf": None,
        "source": "fusion"
    })
# -------------------------------------------------------------------------------

# ---------- DEBUG: show ded_list candidates that look barcode-like ----------
if show_debug:
    st.write("DEBUG: ded_list entries (potential barcode-like tokens):")
    for d in ded_list:
        norm = d.get("norm","")
        if re.search(r"\d{10,}", norm):  # show any with long digit runs
            st.write({"norm": norm, "raw": d.get("raw"), "conf": d.get("conf"), "source": d.get("source")})

# initialize extracted_value
extracted_value = None

# ------------------- Targeted + repaired extraction: barcode -> _1_ suffix (STRICT length check) -------------------
found_pair_repaired = None

# Build combined full text from Tesseract/EasyOCR if available
combined_full_text = ""
if 'full_raw' in globals() and full_raw:
    combined_full_text = full_raw
if 'easy_res' in locals() and easy_res:
    try:
        combined_full_text += "\n" + "\n".join([str(x) for x in easy_res])
    except:
        pass
# ---------- DEBUG: combined_full_text & tokens_seq_full ----------
if show_debug:
    try:
        st.write("DEBUG: combined_full_text (first 1200 chars):")
        st.text(combined_full_text[:1200])
    except Exception as e:
        st.write("DEBUG: failed to show combined_full_text:", e)
# small heuristic to fix common OCR digit->letter confusions (0->O, 1->I, 5->S, 2->Z, 7->T, 6->G, 8->B, 4->A)
def fix_ocr_letters(s: str) -> str:
    """
    Try to convert noisy tokens that mix digits/letters to a letters-only candidate.
    - Uppercase input, then apply multiple likely digit->letter mappings.
    - Also attempt an alternate mapping for cases where '7' looks like 'R' (common on some fonts).
    - Returns the best letters-only candidate (or empty string if none).
    """
    if not s:
        return ""

    s_up = s.upper()

    # Primary mapping (safe)
    primary_subs = str.maketrans({
        "0": "O",
        "1": "I",
        "5": "S",
        "2": "Z",
        "7": "T",
        "6": "G",
        "8": "B",
        "4": "A",
        "3": "E",   # occasionally 3->E
        "9": "G"    # sometimes 9->G
    })

    # Alternate mapping (for fonts where 7 looks like R)
    alt_subs = str.maketrans({
        "7": "R",
        "0": "O",
        "1": "I",
        "5": "S",
        "2": "Z",
        "6": "G",
        "8": "B",
        "4": "A",
        "3": "E",
        "9": "G"
    })

    # try primary
    cand_primary = re.sub(r"[^A-Z]", "", s_up.translate(primary_subs))
    if 1 <= len(cand_primary) <= 3:
        return cand_primary

    # try alternate (7->R) if primary failed
    cand_alt = re.sub(r"[^A-Z]", "", s_up.translate(alt_subs))
    if 1 <= len(cand_alt) <= 3:
        return cand_alt

    # final fallback: just keep letters that already exist (strip digits)
    cand_letters = re.sub(r"[^A-Z]", "", s_up)
    if 1 <= len(cand_letters) <= 3:
        return cand_letters

    return ""


# Heuristic scan helper: find digit runs and attempt to extend them by looking at neighbors
# This helps when OCR splits a leading digit off or tokenizes digits incorrectly.
def find_best_barcode_candidate(text):
    """Search text for digit runs >=11 and try to extend them by looking at neighbors.
    Returns tuple (barcode_digits, suffix_or_empty) or (None, None).

    Safety improvements:
     - Remove suffix marker `_1...` before digit extraction so `_1` is not absorbed.
     - Only extend from neighbor tokens if digits look like a continuation:
       - Left extension: neighbor must have digits at the *end* of token (e.g. 'X432').
       - Right extension: neighbor must start with digits or be all-digits.
     - Separator tokens like '1', '1AHR', '1_AHR', etc. are treated as separators.
    """
    if not text:
        return None, None

    toks = re.split(r"(\s+|[,;:/|]+)", text)  # keeps separators as separate tokens
    candidates = []
    n = len(toks)

    # detect separator marker tokens like "1", "1AHR", "1_AHR", "1 AHR", "1-AHR", "1.AHR"
    sep_marker_re = re.compile(r"^\s*1(?:[\s\-_\.]*[A-Za-z]{1,3})?\s*$", flags=re.IGNORECASE)

    def is_separator_token(tok):
        if re.fullmatch(r"(\s+|[,;:/|]+)", tok):
            return True
        if sep_marker_re.match(tok):
            return True
        return False

    def next_nonsep_index(start, direction):
        i = start + direction
        while 0 <= i < n:
            if not is_separator_token(toks[i]):
                return i
            i += direction
        return -1

    for i, t in enumerate(toks):
        # keep underscores when cleaning so we can remove patterns like `_1_ABC`
        # remove the suffix marker `_1...` *before* extracting digits
        t_clean = re.sub(r"[^0-9A-Za-z_]", "", t)
        t_clean = re.sub(r"_1.*$", "", t_clean)  # IMPORTANT: strip suffix marker
        digits = re.sub(r"[^0-9]", "", t_clean)

        if len(digits) >= 11:
            extended = digits  # base candidate

            if show_debug:
                st.write(f"DBG: base token idx={i}, token='{t}', cleaned='{t_clean}', digits='{digits}'")

            # ----- SAFE LEFT EXTENSION -----
            li = next_nonsep_index(i, -1)
            if li != -1:
                left_raw = toks[li]
                left_alnum = re.sub(r"[^0-9A-Za-z_]", "", left_raw)
                left_digits = re.sub(r"[^0-9]", "", left_raw)
                # require digits to be present and appear at the END of the token (continuation from left)
                left_digits_at_end = bool(re.search(r"\d+$", left_alnum))
                if left_digits and len(left_digits) <= 3 and left_digits_at_end and not sep_marker_re.match(left_raw):
                    # take only the last digit (best-effort continuation)
                    ext = left_digits[-1] + extended
                    if len(ext) > len(extended):
                        if show_debug:
                            st.write(f"DBG: extending LEFT with '{left_digits[-1]}' from token idx={li} ('{left_raw}') -> '{ext}'")
                        extended = ext
                else:
                    if show_debug:
                        st.write(f"DBG: skipped left token idx={li} ('{left_raw}') as separator/invalid for extension")

            # ----- SAFE RIGHT EXTENSION -----
            ri = next_nonsep_index(i, +1)
            if ri != -1:
                right_raw = toks[ri]
                right_alnum = re.sub(r"[^0-9A-Za-z_]", "", right_raw)
                right_digits = re.sub(r"[^0-9]", "", right_raw)
                # require that neighbor digits are a leading run (starts with digits) or the token is all digits
                right_digits_at_start = bool(re.match(r"^\d+", right_alnum))
                right_is_all_digits = bool(right_alnum.isdigit())
                if right_digits and len(right_digits) <= 3 and (right_digits_at_start or right_is_all_digits) and not sep_marker_re.match(right_raw):
                    ext = extended + right_digits[0]
                    if len(ext) > len(extended):
                        if show_debug:
                            st.write(f"DBG: extending RIGHT with '{right_digits[0]}' from token idx={ri} ('{right_raw}') -> '{ext}'")
                        extended = ext
                else:
                    if show_debug:
                        st.write(f"DBG: skipped right token idx={ri} ('{right_raw}') as separator/invalid for extension")

            # find a short suffix adjacent (letters only)
            # find a short suffix adjacent (letters only)
            # find a short suffix adjacent (letters only) — STRICter rules:
            suffix = ""
            # keep letters+digits to allow later correction, but prefer same-token trailing letters
            same_token_alnum = re.sub(r"[^A-Za-z0-9]", "", t)
            # if the barcode and suffix are inside the same token (like '..._1AHR' or '_1_AHR'), prefer trailing letters
            m_same = re.search(r"_1[_\-\.]?([A-Za-z]{1,3})$", same_token_alnum, flags=re.IGNORECASE)
            if m_same:
                cand = m_same.group(1).upper()
                if cand not in SUFFIX_BLACKLIST and not is_ambiguous_single_letter(cand):
                    suffix = cand
            else:
                # examine immediate neighbor tokens (only accept all-letter tokens length 1..3)
                ri1 = next_nonsep_index(i, +1)
                ri2 = next_nonsep_index(i + (1 if ri1 == -1 else 0), +1)
                for j in (i, ri1, ri2):
                    if j is None or j == -1:
                        continue
                    if 0 <= j < n:
                        tok = toks[j]
                        letters_only = re.sub(r"[^A-Za-z]", "", tok)
                        # accept only if token is 1..3 letters and not in blacklist
                        if 1 <= len(letters_only) <= 3:
                            letters_only_up = letters_only.upper()
                            if letters_only_up not in SUFFIX_BLACKLIST and not is_ambiguous_single_letter(letters_only_up):
                                suffix = letters_only_up
                                break
                        # If token is short but contains digits/letters (e.g. 'i70'), try to fix obvious confusions
                        alnum = re.sub(r"[^A-Za-z0-9]", "", tok)
                        if 1 <= len(alnum) <= 3:
                            fixed = fix_ocr_letters(alnum)  # returns letters-only uppercased
                            if 1 <= len(fixed) <= 3 and fixed not in SUFFIX_BLACKLIST and not is_ambiguous_single_letter(fixed):
                                suffix = fixed
                                break


            # If `extended` ends with '1' and the immediate next non-sep token is a separator marker (like '1' or '1AHR'),
            # then that trailing '1' was probably borrowed from the separator and should be removed.
            if extended.endswith("1"):
                rn = next_nonsep_index(i, +1)
                if rn != -1:
                    rn_tok = toks[rn]
                    if sep_marker_re.match(rn_tok):
                        if show_debug:
                            st.write(f"DBG: removing trailing '1' from extended because next non-sep token idx={rn} ('{rn_tok}') looks like separator")
                        if len(extended) > 0:
                            extended = extended[:-1]

            candidates.append((extended, suffix, i))
            if show_debug:
                st.write(f"DBG: candidate appended: (extended='{extended}', suffix='{suffix}', idx={i})")

    if not candidates:
        return None, None

    # Rank candidates so that ones with a suffix win over suffix-less candidates,
    # then by digit length, then by suffix length.
    # This avoids selecting a suffix-less heuristic candidate over a slightly-shorter
    # candidate that includes a short suffix (which is usually the more-correct match).
    candidates.sort(key=lambda x: (
        0 if x[1] else 1,   # has suffix -> better
        -len(x[0]),         # longer digit run -> better
        -len(x[1])          # longer suffix -> better
    ))
    best = candidates[0]
    if show_debug:
        st.write(f"DBG: best candidate chosen: {best}")
    return best[0], best[1]

# Run heuristic before strict regex/adacency checks
# define valid_barcode before using heuristic
# Helper to validate barcode length: STRICT > 10 digits (i.e. 11+)
def valid_barcode(b):
    return b is not None and len(b) > 10

# Helper to validate suffix length (must be 1..3 letters)
def valid_suffix(s):
    if s is None:
        return False
    s = s.strip()
    return 1 <= len(s) <= 3 and s.isalpha()

# Helper to format barcode+suffix consistently
def format_pair(barcode, suffix):
    """Return formatted pair. If suffix is empty or falsy, return 'barcode_1' without trailing underscore."""
    if suffix:
        return f"{barcode}_1_{suffix}"
    return f"{barcode}_1"

hb_barcode, hb_suffix = find_best_barcode_candidate(combined_full_text)

# If heuristic found a strong barcode + suffix, accept immediately (short-circuit)
HEURISTIC_FINALIZED = False
if hb_barcode and valid_barcode(hb_barcode):
    if hb_suffix and valid_suffix(hb_suffix):
        found_pair_repaired = format_pair(hb_barcode, hb_suffix)
        extracted_value = found_pair_repaired.upper()
        HEURISTIC_FINALIZED = True
    else:
        HEURISTIC_FINALIZED = False

# If heuristic did not finalize, try to find an exact-match candidate in OCR/fusion pool.
# NOTE: use exact numeric equality (no substring matching) to avoid wrong replacements.
if not HEURISTIC_FINALIZED and hb_barcode and valid_barcode(hb_barcode):
    candidate_pool = [d.get("norm","") for d in ded_list if d.get("norm")]
    try:
        candidate_pool += fused_tokens if 'fused_tokens' in globals() or 'fused_tokens' in locals() else []
    except:
        pass
    candidate_pool = [str(x).upper() for x in candidate_pool if x]

    better_found = None
    for c in candidate_pool:
        if "_1" not in c:
            continue
        m = re.match(r'^(\d+)_1(?:_([A-Z0-9]{1,3}))?$', c, flags=re.IGNORECASE)
        if not m:
            continue
        num = m.group(1)
        suf = (m.group(2) or "").upper()

        # Require exact numeric equality — do NOT use substring checks here.
        if num == hb_barcode:
            # prefer valid, non-blacklisted suffixes
            if valid_suffix(suf) and suf not in SUFFIX_BLACKLIST and not is_ambiguous_single_letter(suf):
                better_found = c
                break
            # otherwise keep as fallback (only if no better)
            if not better_found:
                better_found = c

    if better_found:
        found_pair_repaired = better_found
    else:
        # accept heuristic only if it had a valid suffix (we already short-circuited that case),
        # or if there are absolutely no _1 candidates in the pool.
        any__1_in_pool = any("_1" in c for c in candidate_pool)
        if not any__1_in_pool:
            found_pair_repaired = format_pair(hb_barcode, hb_suffix)

# Regex: digits (11 or more) then whitespace, then '1' optionally with separators and a 1..3-letter suffix
barcode_suffix_re = re.compile(r"(\d{11,})[\s\-_\.]*1[\s\-_\.]*([A-Za-z]{1,3})", flags=re.IGNORECASE)


# 1) Try direct regex on combined full text (Tesseract + EasyOCR text)
# Build a token sequence for proximity checks (used below)
tokens_seq_full = []
if combined_full_text:
    for line in combined_full_text.splitlines():
        parts = re.split(r"[\t,;:|]+|\s{2,}", line.strip())
        for part in parts:
            if not part:
                continue
            subtoks = re.split(r"[\s,;:/|]+", part.strip())
            for stt in subtoks:
                if stt:
                    tokens_seq_full.append(stt)

    # ---------- DEBUG: tokens_seq_full ----------
if show_debug:
    try:
        st.write("DEBUG: tokens_seq_full (first 80 tokens):")
        st.write(tokens_seq_full[:80])
    except Exception as e:
        st.write("DEBUG: failed to show tokens_seq_full:", e)


# helper: find index of a token in tokens_seq_full that contains the given alphanumeric snippet
def find_token_index(tokens, snippet):
    """
    Return the first index i such that tokens[i] (after removing punctuation) contains snippet (case-insensitive),
    or -1 if not found.
    """
    if not tokens or not snippet:
        return -1
    sn = re.sub(r"[^0-9A-Za-z]", "", snippet).lower()
    for i, tk in enumerate(tokens):
        tk_clean = re.sub(r"[^0-9A-Za-z]", "", tk).lower()
        if not tk_clean:
            continue
        if sn.isdigit():
            if sn in tk_clean:
                return i
        else:
            if sn in tk_clean:
                return i
    return -1

# helper: obtain cleaned token string for a token index
def token_clean_at(tokens, idx):
    if not tokens or idx < 0 or idx >= len(tokens):
        return ""
    return re.sub(r"[^0-9A-Za-z]", "", tokens[idx])

if combined_full_text:
    m = barcode_suffix_re.search(combined_full_text)
    if m:
        barcode = re.sub(r"[^0-9]", "", m.group(1))
        suffix_raw = m.group(2) or ""
        # keep letters only (no digits, no punctuation)
        suffix = re.sub(r"[^A-Za-z0-9]", "", suffix_raw)

        # find token index of barcode in the tokenized combined string
        barcode_token_idx = find_token_index(tokens_seq_full, m.group(1))
        barcode_token_clean = token_clean_at(tokens_seq_full, barcode_token_idx).lower()

        # DEBUG — show what we matched (remove or comment out after testing)
        if show_debug:
            st.write("DEBUG: regex match token index:", barcode_token_idx, "token_clean:", barcode_token_clean, "barcode:", barcode)

        # If the token that contains the barcode also contains letters that precede the digit-run,
        # treat those letters as prefix and never use them as suffix.
        # Example: token_clean == "lgp03251056557" -> letters before digits -> 'lgp' is prefix, disallow as suffix.
        if barcode_token_clean:
            # find where the barcode digits start within the token, if possible
            pos_digits = barcode_token_clean.find(barcode)
            if pos_digits > 0:
                # there are letters before digits in same token -> drop any suffix that equals that prefix
                prefix_letters = barcode_token_clean[:pos_digits]
                # if our current regex-derived suffix equals that prefix, clear it
                if suffix and suffix.lower() == prefix_letters:
                    if show_debug:
                        st.write("DEBUG: found left-side prefix in same token; ignoring that as suffix:", prefix_letters)
                    suffix = ""

        # prefer easyocr detection of nearby short letter-only suffix, but ONLY if the easy token's
        # token index is at or after the barcode's token index AND the letters occur at/after digits
        if suffix and easy_raw_tokens and combined_full_text:
            s_lower = suffix.lower()
            for et in easy_raw_tokens:
                et_letters = re.sub(r"[^A-Za-z0-9]", "", et)
                if not et_letters:
                    continue
                if not (1 <= len(et_letters) <= 3):
                    continue

                # find token index of this easy token in the combined string
                et_idx = find_token_index(tokens_seq_full, et_letters)
                if et_idx == -1:
                    # maybe easy token included digits+letters; try token search using the whole raw
                    et_idx = find_token_index(tokens_seq_full, et)

                # accept only if easy token exists and occurs at/after the barcode token index
                if et_idx != -1 and (barcode_token_idx == -1 or et_idx >= barcode_token_idx):
                    # If easy token is in the same token index as the barcode, ensure it occurs at/after digits
                    if et_idx == barcode_token_idx and barcode_token_clean:
                        # find positions inside the token string
                        et_letters_low = et_letters.lower()
                        # find position of the digit run within the cleaned token
                        pos_digits = barcode_token_clean.find(barcode)
                        pos_letters = barcode_token_clean.find(et_letters_low)
                        # if the easy-letter-run occurs before the digits, reject it (it's a prefix)
                        if pos_letters != -1 and pos_digits != -1 and pos_letters < pos_digits:
                            # skip this easy token because it appears before digits
                            continue
                    # require that the easy token contains / matches our suffix (substring or equal)
                    if s_lower in et_letters.lower() or et_letters.lower() in s_lower:
                        candidate = et_letters
                        if valid_suffix(candidate) and not is_ambiguous_single_letter(candidate):
                            suffix = candidate
                            break

        # reject ambiguous single-letter suffixes (like 'I' that are likely OCR noise)
        if suffix and is_ambiguous_single_letter(suffix):
            if show_debug:
                st.write(f"DEBUG: rejected ambiguous single-letter suffix '{suffix}'")
            suffix = ""

        if valid_barcode(barcode) and valid_suffix(suffix):
            found_pair_repaired = format_pair(barcode, suffix)

# 2) Fallback: token adjacency in combined_full_text or ded_list (best-effort)
if not found_pair_repaired:
    tokens_seq = []
    if combined_full_text:
        for line in combined_full_text.splitlines():
            # split preserving tokens; allow multiple whitespace separators
            parts = re.split(r"[\t,;:|]+|\s{2,}", line.strip())
            for part in parts:
                if not part:
                    continue
                subtoks = re.split(r"[\s,;:/|]+", part.strip())
                for stt in subtoks:
                    if stt:
                        tokens_seq.append(stt)
    else:
        tokens_seq = [d.get("raw", "") for d in ded_list]

    for i in range(len(tokens_seq) - 1):
        left = tokens_seq[i]
        right = tokens_seq[i + 1]

        left_digits = re.sub(r"[^0-9]", "", str(left))
        right_clean = re.sub(r"[^0-9A-Za-z\-_\. ]", "", str(right)).strip()

        # require barcode-like left (STRICT >10 digits)
        if valid_barcode(left_digits):
            # case 1: right starts with '1' followed by 1..3 letters: "1kuu" or "1-kuu" or "1 kuu"
            m2 = re.match(r"^1[\s\-_\.]*([A-Za-z]{1,3})$", right_clean, flags=re.IGNORECASE)
            if m2:
                suffix_raw = m2.group(1) or ""
                                # clean candidate suffix to letters only and reject obvious non-suffix words
                candidate_suffix = re.sub(r"[^A-Za-z]", "", suffix_raw).upper()
                if candidate_suffix and (len(candidate_suffix) < 1 or len(candidate_suffix) > 3 or candidate_suffix in SUFFIX_BLACKLIST or is_ambiguous_single_letter(candidate_suffix)):
                    candidate_suffix = ""

                # Prefer short-letter tokens to the right (i+1..i+2), enforced same-token check
                if easy_raw_tokens and tokens_seq_full:
                    left_token_idx_in_full = find_token_index(tokens_seq_full, left)
                    for j in range(i+1, min(i+3, len(tokens_seq))):
                        right_candidate = re.sub(r"[^A-Za-z]", "", str(tokens_seq[j])).upper()
                        if 1 <= len(right_candidate) <= 3 and right_candidate not in SUFFIX_BLACKLIST and not is_ambiguous_single_letter(right_candidate):
                            right_idx_in_full = find_token_index(tokens_seq_full, right_candidate)
                            if right_idx_in_full == -1:
                                right_idx_in_full = find_token_index(tokens_seq_full, str(tokens_seq[j]))
                            if right_idx_in_full != -1 and (left_token_idx_in_full == -1 or right_idx_in_full >= left_token_idx_in_full):
                                # avoid picking prefixes from same token (ensure occurrence after digits if same token)
                                token_clean = token_clean_at(tokens_seq_full, right_idx_in_full).lower()
                                left_token_clean = token_clean_at(tokens_seq_full, left_token_idx_in_full).lower() if left_token_idx_in_full != -1 else ""
                                if right_idx_in_full == left_token_idx_in_full and left_token_clean and left_digits:
                                    pos_digits = left_token_clean.find(left_digits)
                                    pos_letters = left_token_clean.find(right_candidate.lower())
                                    if pos_letters != -1 and pos_digits != -1 and pos_letters < pos_digits:
                                        continue
                                candidate_suffix = right_candidate
                                break


                # discard ambiguous single-letter suffixes
                if is_ambiguous_single_letter(candidate_suffix):
                    if show_debug:
                        st.write("DEBUG: adjacency rejected ambiguous suffix:", candidate_suffix)
                    candidate_suffix = ""

                if valid_suffix(candidate_suffix):
                    found_pair_repaired = format_pair(left_digits, candidate_suffix)
                    break
            # case 2: right is exactly '1' and next token provides suffix
            if re.fullmatch(r"^1$", right_clean) and i + 2 < len(tokens_seq):
                next_tok = re.sub(r"[^A-Za-z0-9]", "", str(tokens_seq[i + 2]))
                if next_tok and valid_suffix(next_tok):
                    left_token_idx_in_full = find_token_index(tokens_seq_full, left) if tokens_seq_full else -1
                    next_idx_in_full = find_token_index(tokens_seq_full, next_tok) if tokens_seq_full else -1
                    if tokens_seq_full:
                        if next_idx_in_full != -1 and (left_token_idx_in_full == -1 or next_idx_in_full >= left_token_idx_in_full):
                            # ensure not a left-side prefix in same token
                            token_clean = token_clean_at(tokens_seq_full, next_idx_in_full).lower()
                            left_token_clean = token_clean_at(tokens_seq_full, left_token_idx_in_full).lower() if left_token_idx_in_full != -1 else ""
                            if next_idx_in_full == left_token_idx_in_full and left_token_clean and left_digits:
                                pos_digits = left_token_clean.find(left_digits)
                                pos_letters = left_token_clean.find(next_tok.lower())
                                if pos_letters != -1 and pos_digits != -1 and pos_letters < pos_digits:
                                    # it's a left-side prefix -> skip
                                    continue
                            if not is_ambiguous_single_letter(next_tok):
                                found_pair_repaired = format_pair(left_digits, next_tok)
                                break
                    else:
                        if not is_ambiguous_single_letter(next_tok):
                            found_pair_repaired = format_pair(left_digits, next_tok)
                            break

# If found, set extracted_value (normalized repaired form)
if found_pair_repaired:
    extracted_value = found_pair_repaired.upper()
# Patch: Add heuristic candidate to ded_list so pick_best_match can score it
# Insert this block after the HEURISTIC_FINALIZED section and before the strict matching
# (i.e., just before the line: "# ------------------- STRICT MATCHING (RUN ONLY IF NO REPAIRED VALUE) --")

# --- BEGIN PATCH ---
if hb_barcode and valid_barcode(hb_barcode):
    # create normalized candidate from heuristic (HB = heuristic barcode)
    cand_norm = format_pair(hb_barcode, hb_suffix).upper()
    # only add if not already present in ded_list
    already=False
    if not already:
        # append with high confidence and a distinct source so pick_best_match can score it
        ded_list.append({
            "norm": cand_norm,
            "raw": cand_norm,
            "conf": None,
            "source": "heuristic"
        })
        if show_debug:
            st.write(f"DEBUG: added heuristic candidate to ded_list: {cand_norm}")
# --- END PATCH ---

# ------------------- STRICT MATCHING (RUN ONLY IF NO REPAIRED VALUE) -------------------
if not extracted_value:
    if show_debug:
        candidate_pool = [d.get("norm", "") for d in ded_list if d.get("norm")]
        st.write("DEBUG: candidate_pool (post-fusion):", [c for c in candidate_pool])

    strict_matches = []
    for d in ded_list:
        norm = stabilize_suffix(d.get("norm", ""))
        norm = norm.rstrip("_")
        if "_1" not in norm:
            continue
        if not pattern.search(norm):
            continue
        suffix_now = re.search(r'_1_([A-Z0-9]{1,3})$', norm)
        if suffix_now and is_blacklisted_suffix(suffix_now.group(1)):
            if show_debug:
                st.write(f"DEBUG: strict matching skipping blacklisted suffix candidate: {norm}")
            continue

        mnum = re.match(r'^(\d+)_1', norm)
        if mnum:
            prefix_len = len(mnum.group(1))
        else:
            mm = re.match(r'^(\d+)', norm)
            prefix_len = len(mm.group(1)) if mm else 0

        if prefix_len <= 10:
            continue

        strict_matches.append({
            "norm": norm,
            "raw": d.get("raw"),
            "conf": d.get("conf"),
            "source": d.get("source"),
        })

    # merge strict matches
    ded_list.extend(strict_matches)

    if show_debug:
        st.write("DEBUG: STRICT matches (before scoring):", strict_matches)

best_match = pick_best_match(ded_list) if ded_list else None

if best_match:
    extracted_value = best_match["norm"].upper()
else:
    extracted_value = ""  # explicit: no match found

# ------------------- AUTO-SAVE FINAL RESULT (SINGLE JSON) -------------------
# Save the image name + extracted value into a single JSON file (results/single_results.json).
# The file will be a JSON array of objects: [{"file": "name.jpg", "extracted": "..."}, ...]
try:
    os.makedirs("results", exist_ok=True)
    single_path = os.path.join("results", "single_results.json")
    entry = {"file": uploaded.name, "extracted": (extracted_value or "").strip().upper()}

    if os.path.exists(single_path):
        try:
            with open(single_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None

        if isinstance(existing, list):
            existing.append(entry)
            out_data = existing
        elif isinstance(existing, dict):
            out_data = [existing, entry]
        else:
            out_data = [entry]
    else:
        out_data = [entry]

    with open(single_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    st.info(f"Auto-saved result to {single_path}")
except Exception as e:
    st.warning(f"Failed to auto-save result: {e}")

# ------------------- DISPLAY ONLY THE FINAL ANSWER -------------------
if extracted_value:
    st.success(extracted_value.upper())
else:
    st.error("No `_1` match found.")

# ------------------- Save / Edit -------------------
st.subheader("Save / Edit extracted value")
final_value = st.text_input("Final Value", value=extracted_value)

if st.button("Save extracted value"):
    if not final_value.strip():
        st.error("Value empty.")
    else:
        os.makedirs("results", exist_ok=True)
        out = {
            "file": uploaded.name,
            "extracted": final_value.strip().upper(),
            "source": "full_text_scan"
        }
        save_json(out, os.path.join("results", uploaded.name + ".json"))
        st.success("Saved.")
