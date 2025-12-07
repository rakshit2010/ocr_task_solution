# text_extraction.py  (REPLACE existing file)
import re
from Levenshtein import distance as levdist

# pattern: something_1_something - underscores required but we'll attempt to repair missing underscores
TARGET_RE = re.compile(r'\b\S*_1_\S*\b')

# mapping for common confusions
CONFUSABLES = str.maketrans({'I':'1','l':'1','|':'1','O':'0','o':'0','S':'5','Z':'2'})

def normalize_text(s):
    if s is None:
        return ""
    s2 = s.strip()
    # remove weird whitespace inside
    s2 = re.sub(r'[\s\-]+', '', s2)
    # apply char map
    s2 = s2.translate(CONFUSABLES)
    # keep only allowed chars
    s2 = re.sub(r'[^0-9A-Za-z_]', '', s2)
    return s2

def find_exact(lines):
    for item in lines:
        txt = normalize_text(item['text'])
        m = TARGET_RE.search(txt)
        if m:
            return m.group(0), item.get('conf', None)
    return None, None

def repair_insert_underscore(s):
    # if there's '1' but underscores missing, try to insert underscores around the first '1'
    if '_' in s:
        return s
    if '1' not in s:
        return s
    idx = s.find('1')
    # insert underscores if not at edges
    left = s[:idx]
    right = s[idx+1:]
    # ensure we keep some characters on both sides
    if len(left) >= 1 and len(right) >= 1:
        return left + '_1_' + right
    return s

def fuzzy_search(lines, max_candidates=10):
    """
    lines: list of {'text':..., 'conf':...} from crops / whole-image OCR.
    Returns best candidate string or None.
    """
    candidates = []
    for item in lines:
        raw = item.get('text','')
        norm = normalize_text(raw)
        # produce repairs
        repaired = repair_insert_underscore(norm)
        for cand in [norm, repaired]:
            if not cand: continue
            # score: lower is better
            # prefer strings that contain '_1_'
            contains = 0 if '_1_' in cand else 5
            # distance to a template to prefer strings with reasonable length and structure
            templ = 'A'*max(3, min(10, len(cand)//2)) + '_1_' + 'B'*max(3, min(10, len(cand)//2))
            d = levdist(cand, templ)
            total = contains + d - (item.get('conf',0)/100.0)  # higher conf reduces score
            candidates.append((total, cand, item.get('conf',0), raw))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    # return best candidate
    best = candidates[0]
    return best[1], {'score': best[0], 'conf': best[2], 'raw': best[3]}

def extract_from_ocr_results(all_engine_results):
    """
    all_engine_results: list of lists of {'text','conf', 'bbox'(optional)}
    We'll flatten, try exact matches first, else fuzzy search across candidates.
    """
    flattened = []
    seen = set()
    for engine in all_engine_results:
        for item in engine:
            txt = item.get('text','').strip()
            if not txt:
                continue
            key = txt
            if key in seen: continue
            seen.add(key)
            flattened.append({'text': txt, 'conf': item.get('conf', 0)})
    # try exact
    exact, c = find_exact(flattened)
    if exact:
        return exact, c, 'exact'
    # fuzzy across flattened
    fuzzy, meta = fuzzy_search(flattened)
    if fuzzy:
        return fuzzy, meta, 'fuzzy'
    return None, None, 'none'

