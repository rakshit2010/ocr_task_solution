# Shipping Label OCR — Full-Text First Extraction System

## 1. Project Overview

This project is an advanced OCR-based system designed to extract shipping label barcodes following the pattern:

```
<11+ digit barcode>_1_<1–3 letter suffix>
```

The system performs full-text scanning, applies multi-engine OCR fusion (Tesseract + EasyOCR), and uses heuristic repair logic to reconstruct barcodes even when OCR splits or distorts characters. The UI is built using Streamlit, and OpenCV handles the preprocessing pipeline. Extracted results are auto-saved into a unified JSON file.

---

## 2. Installation Instructions

### Prerequisites

* Python 3.8+
* Tesseract OCR installed
* Pip dependencies

### Step-by-Step Setup

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rakshit2010/ocr_task_solution.git
cd shipping-label-ocr
```

#### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

Typical requirements:

```
streamlit
opencv-python
numpy
pytesseract
easyocr
```

#### 3️⃣ Install Tesseract (Windows)

Download from:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Add to PATH:

```
C:\Program Files\Tesseract-OCR
```
Or change the path in ocr_engine.py to your tesseract path
#### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 3. Usage Guide

### Upload Image

Upload PNG/JPG/TIFF shipping label images.

### Adjust Preprocessing

Use the sidebar slider to adjust upscale factor.

### Perform OCR

Click:

* **Scan all text and search for pattern**

The system executes:

* Auto-rotation
* Enhancement + denoising
* Full-text OCR
* Heuristic reconstruction
* Pattern matching & engine fusion

### View & Edit

* Extracted barcode is displayed at the bottom.
* Auto-saved to:

```
results/single_results.json
```

* Manual editing also allowed.

---

## 4. Technical Approach

### OCR Methods Used

#### Tesseract OCR

* Strong with numeric strings
* Offers orientation detection (OSD)

#### EasyOCR

* Better for alphabetic suffixes

---

### Preprocessing Techniques

* Grayscale conversion
* CLAHE enhancement
* Non-local means denoising
* ×2 upscaling
* Sharpening
* Adaptive + Otsu thresholding
* Morphological cleanup
* Auto-orientation (Tesseract OSD + contour-based)

---

### Text Extraction Logic

The system uses **five extraction stages**:

#### 1️⃣ Full-text OCR (Tesseract + EasyOCR)

Tokens collected → normalized → deduplicated.

#### 2️⃣ Heuristic Barcode Reconstruction

* Detects digit runs ≥ 11
* Extends digits from neighbors
* Extracts suffix candidates

#### 3️⃣ Regex Matching

Ensures:

* 11+ digits
* `_1`
* Optional 1–3 letter suffix

#### 4️⃣ Token Adjacency Logic

Handles split tokens.

#### 5️⃣ OCR Fusion

Combines:

* Tesseract digits
* EasyOCR alphabetic suffix

---

## 5. Challenges & Solutions

### OCR splits digits incorrectly

**Solution:** Heuristic stitching + reconstruction + fusion.

### Suffix letters misread as digits

**Solution:** Prefer EasyOCR for suffix detection.

### Rotation inconsistencies

**Solution:** Combine Tesseract OSD with contour-based correction.

### Small or low-resolution text

**Solution:** Upscaling, sharpening, and advanced denoising.

### Short (invalid) digit prefixes appearing

**Solution:** Enforce strict 11+ digit rule.

---

## 6. Future Improvements

* Integrate deep-learning OCR (PaddleOCR, TrOCR)
* YOLO-based barcode localization
* Automatic label region cropping
* Confidence scoring & heatmaps
* Batch processing mode
* Docker deployment
* Cloud API version


