import cv2
import numpy as np
import os
from rapidocr_onnxruntime import RapidOCR


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


def _crop_to_content(img):
    """
    Crop away dark borders (e.g. table/desk background in phone photos).
    Finds the largest bright rectangular region and crops to it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Threshold to find bright (paper) region — use 160 to better separate
    # white notebook paper from dark desk/table backgrounds in phone photos
    _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Find bounding box of the bright region
    coords = cv2.findNonZero(bright_mask)
    if coords is None:
        return img  # can't crop, return as-is

    x, y, w, h = cv2.boundingRect(coords)

    # Add small padding
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + pad * 2)
    h = min(img.shape[0] - y, h + pad * 2)

    return img[y:y+h, x:x+w]


def _remove_ruled_lines(gray, original_bgr=None):
    """
    Remove horizontal ruled lines.  Bug D fix: uses HSV saturation channel
    when colour image is available so blue ink strokes survive.
    """
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

    # Grayscale fallback
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
    Return multiple preprocessed versions of the image for RapidOCR to try.
    Each variant is a BGR numpy array (RapidOCR expects BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    # Step 1: crop dark borders
    img = _crop_to_content(img)

    # Bug A fix: raise threshold 1500 → 2500, scale 2× → 3× for small images
    h, w = img.shape[:2]
    if max(h, w) < 2500:
        scale = 3 if max(h, w) < 1000 else 2
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bug B fix: lower CLAHE aggressiveness to avoid grain amplification
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Bug C fix: gentle denoise — preserve thin syntax characters
    denoised = cv2.fastNlMeansDenoising(
        enhanced, None, h=7, templateWindowSize=5, searchWindowSize=21
    )

    # Bug E fix: deskew AFTER line removal (see _deskew call below)
    variants = []

    # V1: colour original — no destructive ops (Bug F)
    variants.append(('colour_original', img))

    # V2: colour-aware line removal + deskew after (Bug D + E)
    line_free = _remove_ruled_lines(denoised, original_bgr=img)
    line_free_deskewed = _deskew(line_free)
    variants.append(('line_free', cv2.cvtColor(line_free_deskewed, cv2.COLOR_GRAY2BGR)))

    # V3: plain Otsu — sometimes better when lines are faint
    _, otsu = cv2.threshold(line_free_deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(('otsu', cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))

    # V4: adaptive threshold on line-free deskewed image
    adaptive = cv2.adaptiveThreshold(
        line_free_deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    variants.append(('adaptive', cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # V5: sharpen + colour-aware line removal
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    sharp_lf = _remove_ruled_lines(sharpened, original_bgr=img)
    variants.append(('sharpened', cv2.cvtColor(sharp_lf, cv2.COLOR_GRAY2BGR)))

    return variants


def _score_text(text):
    """Score OCR output — delegates to shared ocr_utils.ocr_quality_score()."""
    from ocr_utils import ocr_quality_score
    return ocr_quality_score(text)


class RapidOCRProcessor:
    """Handwriting recognition using RapidOCR with ruled-line-aware preprocessing."""

    def __init__(self):
        print("=" * 50)
        print("LOADING RapidOCR MODEL...")
        print("=" * 50)
        try:
            self.engine = RapidOCR()
            print("✅ RapidOCR loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load RapidOCR: {e}")
            print("⚠️  Make sure you have installed: pip install rapidocr-onnxruntime")
            self.engine = None

    def preprocess_image(self, image_path: str):
        """Return best single preprocessed image (kept for backwards compat)."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = _crop_to_content(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Keep in sync with _preprocess_variants: Bug B + C params
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(
                enhanced, None, h=7, templateWindowSize=5, searchWindowSize=21
            )
            line_free = _remove_ruled_lines(denoised, original_bgr=img)
            return cv2.cvtColor(line_free, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def extract_text(self, image_path: str) -> str | None:
        """
        Extract text using RapidOCR engine.
        Tries multiple preprocessing variants and picks the best result
        by quality score.
        """
        if self.engine is None:
            print("⚠️  RapidOCR model not loaded")
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
                    results, _ = self.engine(img_variant)
                    if not results:
                        continue

                    lines = [line[1] for line in results if line and line[1].strip()]
                    candidate = '\n'.join(lines).strip()
                    if not candidate:
                        continue

                    score = _score_text(candidate)
                    print(f"  RapidOCR variant '{name}': score={score} | {repr(candidate[:80])}")

                    if score > best_score:
                        best_score = score
                        best_text = candidate

                except Exception as e:
                    print(f"  RapidOCR variant '{name}' error: {e}")
                    continue

            if best_text:
                print(f"  ✅ RapidOCR best variant score={best_score}")

            return best_text

        except Exception as e:
            print(f"RapidOCR extraction error: {e}")
            return None


# Singleton instance
_rapidocr_instance: RapidOCRProcessor | None = None


def get_rapidocr_processor() -> RapidOCRProcessor:
    """Get or create RapidOCR processor singleton"""
    global _rapidocr_instance
    if _rapidocr_instance is None:
        _rapidocr_instance = RapidOCRProcessor()
    return _rapidocr_instance
