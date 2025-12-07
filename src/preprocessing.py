# preprocessing.py  (REPLACE existing file)
import cv2
import numpy as np

def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_eq(gray, clipLimit=3.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def denoise_bilateral(gray, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

def fast_denoise(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)

def deskew(gray):
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def morphological_process(bin_img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def binarize_adaptive(gray, blocksize=31, C=9):
    if blocksize % 2 == 0:
        blocksize += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)

def binarize_otsu(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def upscale(img, scale=2):
    h,w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def preprocess_master(path, upscale_factor=1, return_variants=False):
    """
    Returns a strong preprocessed binary image and optional variants (different thresholds)
    """
    img = load_image(path)
    gray = to_gray(img)
    gray = deskew(gray)
    gray = clahe_eq(gray)
    gray = denoise_bilateral(gray, d=9)
    gray = fast_denoise(gray, h=8)
    if upscale_factor > 1:
        gray = upscale(gray, upscale_factor)
    # two binarization variants to try
    bin1 = binarize_adaptive(gray, blocksize=31, C=9)
    bin2 = binarize_otsu(gray)
    bin1 = morphological_process(bin1, kernel_size=3)
    bin2 = morphological_process(bin2, kernel_size=2)
    main = bin1 if cv2.countNonZero(bin1) > cv2.countNonZero(bin2) else bin2
    if return_variants:
        return main, [bin1, bin2], gray
    return main
