import cv2
import numpy as np
import os


def _deskew(gray):
    """Correct slight tilt before thresholding."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR processor for handwritten code
#
# EasyOCR uses a CRNN (CNN + BiLSTM + CTC) architecture that is significantly
# better than TrOCR and Tesseract at recognising syntax characters like:
#   ( ) = . " ' % : { } [ ] + - * / < > ! #
#
# It does NOT hallucinate prose the way TrOCR does, and it handles mixed
# handwriting + symbols better than Tesseract's character segmentation.
# ─────────────────────────────────────────────────────────────────────────────


def _remove_ruled_lines(gray, original_bgr=None):
    """Bug D fix: colour-aware line removal — see image_preprocessor.py."""
    if original_bgr is not None:
        hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        line_mask = cv2.bitwise_and(
            cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)[1],
            cv2.threshold(val, 100, 255, cv2.THRESH_BINARY)[1],
        )
        h, w = line_mask.shape
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
        detected = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, h_kernel, iterations=2)
        repair = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        dilated = cv2.dilate(detected, repair, iterations=1)
        result = gray.copy()
        result[dilated > 0] = 255
        return result

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    lines_dilated = cv2.dilate(detected_lines, repair_kernel, iterations=1)
    cleaned = cv2.subtract(binary, lines_dilated)
    return cv2.bitwise_not(cleaned)


def _preprocess_variants(image_path):
    """
    Return multiple preprocessed BGR numpy arrays for EasyOCR to try.
    EasyOCR accepts BGR numpy arrays directly.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    # Crop dark borders
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_full, 160, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(bright_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 20
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(img.shape[1] - x, w + pad * 2)
        h = min(img.shape[0] - y, h + pad * 2)
        img = img[y:y+h, x:x+w]

    # Bug A fix: raise threshold 1500 → 2500, scale 3× for small images
    h, w = img.shape[:2]
    if max(h, w) < 2500:
        scale = 3 if max(h, w) < 1000 else 2
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bug B fix: lower CLAHE to avoid grain amplification
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Bug C fix: gentler denoise to preserve thin syntax characters
    denoised = cv2.fastNlMeansDenoising(
        enhanced, None, h=7, templateWindowSize=5, searchWindowSize=21
    )

    variants = []

    # V1: colour original — clean baseline (Bug F)
    variants.append(('original_first', img))

    # V2: colour-aware line removal + deskew AFTER (Bug D + E)
    line_free = _remove_ruled_lines(denoised, original_bgr=img)
    line_free_deskewed = _deskew(line_free)
    variants.append(('line_free', cv2.cvtColor(line_free_deskewed, cv2.COLOR_GRAY2BGR)))

    # V3: adaptive threshold on clean line-free image
    adaptive = cv2.adaptiveThreshold(
        line_free_deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    variants.append(('adaptive', cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # V4: sharpened + colour-aware line removal
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    sharp_lf = _remove_ruled_lines(sharpened, original_bgr=img)
    variants.append(('sharpened', cv2.cvtColor(sharp_lf, cv2.COLOR_GRAY2BGR)))

    return variants


def _score_text(text):
    """
    Score EasyOCR output quality.
    Delegates to the shared ocr_utils.ocr_quality_score() to avoid
    duplication across OCR processor modules.
    """
    from ocr_utils import ocr_quality_score
    return ocr_quality_score(text)


class EasyOCRProcessor:
    """
    Handwriting/code recognition using EasyOCR.
    Speciality: syntax characters ( ) = . " % : that TrOCR/Tesseract miss.

    Install: pip install easyocr
    First run downloads ~100MB of model weights automatically.
    """

    def __init__(self):
        print("=" * 50)
        print("LOADING EasyOCR MODEL...")
        print("=" * 50)
        try:
            import easyocr
            # gpu=False for CPU-only machines; set gpu=True if CUDA is available
            # paragraph=False keeps individual text region results separate
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.available = True
            print("✅ EasyOCR loaded successfully")
        except ImportError:
            print("❌ EasyOCR not installed. Run: pip install easyocr")
            self.reader = None
            self.available = False
        except Exception as e:
            print(f"❌ Failed to load EasyOCR: {e}")
            self.reader = None
            self.available = False

    def extract_text(self, image_path):
        """
        Extract text from image using EasyOCR.
        Tries multiple preprocessing variants and picks the best result
        by quality score.
        """
        if not self.available or self.reader is None:
            print("⚠️  EasyOCR not available")
            return None

        if not os.path.exists(image_path):
            print(f"❌ Image path does not exist: {image_path}")
            return None

        try:
            variants = _preprocess_variants(image_path)
            if not variants:
                return None

            best_text = None
            best_score = -1

            for name, img_variant in variants:
                try:
                    # Bug G fix: use paragraph=True on the colour-original
                    # variant only.  Code lines like "for ( int i = 0 , i<5 ;"
                    # have natural word gaps that paragraph=False splits into
                    # many tiny fragments, each scoring near zero.
                    # paragraph=True merges same-baseline regions into full
                    # lines, recovering the complete statement in one token.
                    # All other variants keep paragraph=False so individual
                    # word confidences remain available for noise filtering.
                    use_paragraph = (name == 'original_first')
                    results = self.reader.readtext(
                        img_variant, detail=1,
                        paragraph=use_paragraph,
                        width_ths=0.9, ycenter_ths=0.5, min_size=10
                    )

                    if not results:
                        continue

                    # paragraph=True  -> results are (bbox, text)       — no conf
                    # paragraph=False -> results are (bbox, text, conf)  — filter low conf
                    if use_paragraph:
                        lines = [text for (_, text) in results if text.strip()]
                    else:
                        lines = [
                            text for (_, text, conf) in results
                            if text.strip() and conf > 0.1
                        ]

                    if not lines:
                        continue

                    candidate = '\n'.join(lines).strip()
                    score = _score_text(candidate)

                    print(f"  EasyOCR variant '{name}': score={score} | {repr(candidate[:80])}")

                    if score > best_score:
                        best_score = score
                        best_text = candidate

                except Exception as e:
                    print(f"  EasyOCR variant '{name}' error: {e}")
                    continue

            if best_text:
                print(f"  ✅ EasyOCR best variant score={best_score}")

            return best_text

        except Exception as e:
            print(f"EasyOCR extraction error: {e}")
            return None

    def process(self, image_path):
        """Entry point — same interface as RapidOCRProcessor and TrOCRHandwritingProcessor."""
        return self.extract_text(image_path)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_instance = None


def get_easyocr_processor():
    """Get or create EasyOCR processor singleton."""
    global _easyocr_instance
    if _easyocr_instance is None:
        _easyocr_instance = EasyOCRProcessor()
    return _easyocr_instance
