# ocr_engine.py
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import easyocr


import numpy as np
import cv2

# initialize easyocr reader (set gpu=True if GPU available)
reader = easyocr.Reader(['en'], gpu=False)

# helper to normalize image for OCR engines (tesseract prefers grayscale or color)
def img_for_tess(img):
    # if already single-channel, convert to BGR for stability
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def tesseract_ocr_variants(img):
    """
    Run Tesseract with multiple PSMs to catch different layouts.
    Returns list of dicts: {'text':..., 'conf':..., 'bbox':(x,y,w,h)}
    """
    img_bgr = img_for_tess(img)
    results = []
    # try multiple psm values that are useful for lines / single-line recognition
    psm_list = [6, 7, 11]  # 6 = assume a single uniform block of text, 7 = treat as single line, 11 sparse text
    whitelist = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    for psm in psm_list:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
        data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT, config=config)
        n = len(data['level'])
        for i in range(n):
            text = data['text'][i].strip()
            if not text:
                continue
            conf = -1
            try:
                conf = float(data['conf'][i])
            except:
                conf = -1
            left = int(data['left'][i]); top = int(data['top'][i]); w = int(data['width'][i]); h = int(data['height'][i])
            results.append({'text': text, 'conf': conf, 'bbox': (left, top, w, h), 'psm': psm})
    return results

def easyocr_ocr(img):
    """
    Returns list of dicts: {'text':..., 'conf':..., 'bbox':(x1,y1,x2,y2)}
    """
    # EasyOCR expects color or gray; it returns bbox as list of 4 points
    try:
        res = reader.readtext(img, detail=1)
    except Exception as e:
        # if easyocr fails (sometimes for weird images), return empty
        return []
    out = []
    for bbox, text, conf in res:
        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [int(p[0]) for p in bbox]; ys = [int(p[1]) for p in bbox]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        out.append({'text': text.strip(), 'conf': float(conf), 'bbox': (left, top, right-left, bottom-top)})
    return out

def ocr_on_crop_variants(img_crop):
    """
    Run both engines on a crop (numpy image). Returns deduped list of candidate texts with confidences.
    """
    candidates = []
    t_out = tesseract_ocr_variants(img_crop)
    e_out = easyocr_ocr(img_crop)
    # collect texts from both
    for t in t_out:
        candidates.append({'text': t['text'], 'conf': t['conf']})
    for e in e_out:
        candidates.append({'text': e['text'], 'conf': e['conf']*100.0})  # easyocr conf is 0-1
    # deduplicate by text normalised
    uniq = {}
    for c in candidates:
        k = c['text'].strip()
        if not k: continue
        if k in uniq:
            uniq[k]['conf'] = max(uniq[k]['conf'], c['conf'])
        else:
            uniq[k] = {'text': k, 'conf': c['conf']}
    return list(uniq.values())
