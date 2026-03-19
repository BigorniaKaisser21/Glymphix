import cv2
import numpy as np


def _deskew(gray):
    """Correct slight tilt.

    Bug E fix: always call AFTER remove_ruled_lines.  Running deskew before
    line removal causes cv2.minAreaRect to compute the angle of the ruled
    lines rather than the handwriting, producing a wrong correction.
    """
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


def remove_ruled_lines(gray, original_bgr=None):
    """
    Detect and remove horizontal ruled lines from notebook paper.

    Bug D fix: when the original colour image is available, detection runs on
    the HSV saturation channel.  Blue ruled lines are low-saturation (pale
    blue/grey-blue); handwriting ink is higher saturation even when it looks
    blue to the eye.  Grayscale cannot separate them — the old morphology
    deleted letter crossbars ('t', 'f', '=', '-') alongside the lines.

    Falls back to grayscale morphology when no colour image is supplied.
    """
    if original_bgr is not None:
        hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]   # 0 = grey, 255 = vivid
        val = hsv[:, :, 2]   # brightness

        # Ruled lines: low saturation AND well-lit (bright enough to be paper)
        line_mask = cv2.bitwise_and(
            cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)[1],
            cv2.threshold(val, 100, 255, cv2.THRESH_BINARY)[1],
        )
        h, w = line_mask.shape
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
        detected = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, h_kernel, iterations=2)
        repair = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        dilated = cv2.dilate(detected, repair, iterations=1)

        # Erase line pixels (set to white / paper colour)
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


def preprocess_for_handwriting(image_path):
    """
    Preprocessing pipeline optimised for handwritten code on ruled paper.
    Returns list of (name, image) tuples — each tried separately by Tesseract.

    Bug fixes vs previous version
    ───────────────────────────────
    A  Upscale threshold 2000 → 2500px; factor 2× → 3× for small images.
       1456px-wide phone photos no longer skipped without upscaling.
    B  CLAHE clipLimit 3.0 → 1.5; tile grid 8×8 → 4×4.
       Stops paper grain being amplified into fake ink speckles.
    C  NLM denoise h 15 → 7; templateWindowSize 7 → 5.
       h=15 blurred syntax characters (;  ,  (  )  <  >) into nothing.
    D  Ruled-line detection uses HSV saturation channel of the colour image.
       Blue ruled lines and blue ink look identical in grayscale.
    E  Deskew runs AFTER line removal, not before.
    F  Colour-original is always the first variant — clean, unmodified path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bug A fix
    h, w = gray.shape
    if max(h, w) < 2500:
        scale = 3 if max(h, w) < 1000 else 2
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        img  = cv2.resize(img,  (img.shape[1] * scale, img.shape[0] * scale),
                          interpolation=cv2.INTER_CUBIC)

    # Bug B fix
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Bug C fix
    denoised = cv2.fastNlMeansDenoising(
        enhanced, None, h=7, templateWindowSize=5, searchWindowSize=21
    )

    results = []

    # V1: colour original — no destructive ops (Bug F)
    results.append(("colour_original", img))

    # V2: colour-aware line removal + deskew after (Bug D + E)
    line_free = remove_ruled_lines(denoised, original_bgr=img)
    line_free_deskewed = _deskew(line_free)
    results.append(("line_free_otsu", line_free_deskewed))

    # V3: adaptive threshold on clean line-free image
    results.append(("line_free_adaptive", cv2.adaptiveThreshold(
        line_free_deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )))

    # V4: sharpen then colour-aware line removal
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    results.append(("sharpened_line_free",
                    remove_ruled_lines(sharpened, original_bgr=img)))

    # V5: plain adaptive, no line removal (for images without ruled lines)
    results.append(("adaptive_plain", cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )))

    return results
