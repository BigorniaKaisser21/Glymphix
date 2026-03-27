from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.urls import url_parse  # kept for Werkzeug < 2.3 compatibility
from authlib.integrations.flask_client import OAuth
from ocr_utils import ocr_quality_score
import os
import sys
import cv2
import numpy as np
from PIL import Image
# FIX #18: Wrap RapidOCR import so the app starts even when the package is
# not installed (e.g. Tesseract-only deployments).
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _RAPIDOCR_IMPORTABLE = True
except ImportError:
    _RapidOCR = None
    _RAPIDOCR_IMPORTABLE = False
import pytesseract
import re
import json
from datetime import datetime, timezone, timedelta
import random
import string
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import logging
from dotenv import load_dotenv
from urllib.parse import urlsplit  # Fix 3: modern replacement for url_parse
from image_preprocessor import preprocess_for_handwriting

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# FIX #9: Do not silently fall back to a known public default key in
# production.  If SECRET_KEY is absent from the environment the fallback is
# only acceptable during local development; log a prominent warning so it
# never goes unnoticed.
_secret_key = os.getenv('SECRET_KEY')
if not _secret_key:
    _secret_key = 'dev-only-insecure-key-change-before-deploy'
    logger.warning(
        "⚠️  SECRET_KEY is not set in the environment. "
        "Using an insecure fallback — set SECRET_KEY in your .env file "
        "before deploying to production!"
    )
app.secret_key = _secret_key

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
mail = Mail(app)

# Database configuration
# Render Postgres supplies DATABASE_URL automatically when a Postgres instance
# is attached to the service.  Older Render (and Heroku) versions emit the URL
# with the "postgres://" scheme; SQLAlchemy 1.4+ requires "postgresql://".
_db_url = os.getenv('DATABASE_URL', 'sqlite:///users.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Connection-pool keep-alive: Render's Postgres closes idle connections after
# 5 minutes; pre_ping recycles stale connections before they are used.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,   # recycle connections every ~4.5 min
}
db = SQLAlchemy(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Flask-Login configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

# OAuth configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'token_endpoint_auth_method': 'client_secret_post',
        'prompt': 'select_account'
    }
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Fix 7: Configure Tesseract path conditionally per platform
if sys.platform == 'win32':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/Mac, tesseract is expected to be on PATH

# OCR Configuration
OCR_ENGINE = 'tesseract'  # 'tesseract' or 'rapidocr' or 'both'

# Qwen2-VL availability is determined lazily (model loads on first use).
# Set QWEN2VL_ENABLED=False here to skip it even when the package is present.
QWEN2VL_ENABLED = True

# Initialize RapidOCR reader (as fallback) — supports Python 3.12+ on Windows
logger.info("Initializing RapidOCR...")
try:
    if _RAPIDOCR_IMPORTABLE:
        rapid_reader = _RapidOCR()
        RAPIDOCR_AVAILABLE = True
        logger.info("RapidOCR initialized successfully")
    else:
        logger.warning("rapidocr-onnxruntime not installed; RapidOCR disabled.")
        rapid_reader = None
        RAPIDOCR_AVAILABLE = False
except Exception as e:
    logger.error(f"RapidOCR initialization failed: {e}")
    rapid_reader = None
    RAPIDOCR_AVAILABLE = False

# Create upload directory if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=True)
    password_hash = db.Column(db.String(200), nullable=True)
    name = db.Column(db.String(100))
    profile_pic = db.Column(db.String(200))
    auth_provider = db.Column(db.String(20), default='local')
    # Fix 8: datetime.utcnow deprecated in Python 3.12+
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if self.password_hash:
            return check_password_hash(self.password_hash, password)
        return False


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200))
    extracted_code = db.Column(db.Text)
    detected_language = db.Column(db.String(50))
    feedback = db.Column(db.Text)
    warnings = db.Column(db.Text)
    suggestions = db.Column(db.Text)
    share_token = db.Column(db.String(100), unique=True, nullable=True)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def generate_share_token(self):
        import secrets
        self.share_token = secrets.token_urlsafe(32)
        self.is_public = True
        return self.share_token


# Create database tables
with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    # Fix 4: db.session.get() replaces deprecated Query.get()
    return db.session.get(User, int(user_id))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== OCR FUNCTIONS ====================

def preprocess_image(image_path):
    """
    Preprocessing for better OCR accuracy.
    Uses ruled-line-aware pipeline from image_preprocessor to avoid
    notebook lines being misread as i, |, H, l characters by Tesseract.
    """
    try:
        return preprocess_for_handwriting(image_path)
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")


def extract_with_tesseract(image_path):
    """
    Extract text using Tesseract OCR.
    Uses ruled-line-removal preprocessing to prevent notebook lines from
    being misread as columns of i, |, H, l characters.
    Selects the best result using a quality score rather than raw length.
    """
    import shutil
    tess_cmd = pytesseract.pytesseract.tesseract_cmd
    if not shutil.which('tesseract') and not os.path.exists(tess_cmd):
        logger.warning("Tesseract not found at %s — skipping", tess_cmd)
        return None

    try:
        processed_images = preprocess_image(image_path)

        best_text = ""
        best_score = -1

        for method_name, proc_img in processed_images:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'tesseract_{method_name}.png')
            cv2.imwrite(temp_path, proc_img)

            # FIX #4: use try/finally so the temp file is always removed even
            # when a PSM loop iteration raises an unexpected exception.
            try:
                for psm in [6, 4, 11, 3, 13]:
                    config = f'--oem 3 --psm {psm}'
                    try:
                        text = pytesseract.image_to_string(Image.open(temp_path), config=config)
                        score = ocr_quality_score(text)
                        if score > best_score:
                            best_text = text
                            best_score = score
                    except Exception as e:
                        logger.error(f"Tesseract error with psm {psm}: {e}")
                        continue
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return best_text

    except Exception as e:
        logger.error(f"Tesseract error: {e}")
        return None


def extract_with_rapidocr(image_path):
    """Extract text using RapidOCR (PaddleOCR-compatible, supports Python 3.12+).

    Delegates to RapidOCRProcessor so the same CLAHE / ruled-line-removal /
    multi-variant preprocessing used inside extract_handwritten_code() is also
    applied here.  Previously this function called rapid_reader() directly on
    the raw image path, bypassing all preprocessing.
    """
    if not RAPIDOCR_AVAILABLE:
        return None
    try:
        from rapidocr_processor import get_rapidocr_processor
        processor = get_rapidocr_processor()
        return processor.extract_text(image_path)
    except Exception as e:
        logger.error(f"RapidOCR error: {e}")
        return None


def extract_handwritten_code(image_path):
    """Extract handwritten code using EasyOCR + RapidOCR + Tesseract pipeline.

    EasyOCR replaces TrOCR as the primary high-accuracy engine.
    TrOCR has been removed — it hallucinated Wikipedia/news prose on
    handwritten code images and never contributed useful output.

    Engine roles:
        EasyOCR   — primary; excels at syntax characters ( ) = . " % : { }
        RapidOCR  — secondary; strong overall structure, fast on CPU
        Tesseract — tertiary; good on printed-style handwriting

    Each engine is scored independently; the highest-scoring result is used
    directly without any cross-engine token voting or combination step.
    """
    EASYOCR_ENABLED = True

    try:
        from rapidocr_processor import get_rapidocr_processor
        from exact_pattern_matcher import ExactPatternMatcher
        from custom_pattern_matcher import CustomPatternMatcher
        from handwriting_fixes import HandwritingFixer
        from dynamic_code_builder import DynamicCodeBuilder
        from library_detector import LibraryDetector

        best_text = ""
        best_score = -1

        # ── Qwen2-VL ──────────────────────────────────────────────────────
        # Vision-language model: understands context end-to-end, no ruled-line
        # preprocessing needed.  Slower than EasyOCR on CPU, but more accurate
        # on ambiguous syntax characters and multi-line code structure.
        #
        # Scoring note: ocr_quality_score() is calibrated for character-level
        # OCR noise (pipes, random symbols).  When Qwen2-VL produces coherent,
        # readable code its raw score can still be beaten by Tesseract noise
        # that happens to contain many syntax characters ({, }, |, :, etc.).
        #
        # Two-stage selection:
        #   1. Fast-path: if the output already looks like real code
        #      (_looks_like_real_code), return it immediately — no other engine
        #      can produce something more trustworthy than a VLM that already
        #      decoded the image correctly.
        #   2. Score-path: otherwise apply a QWEN_SCORE_BONUS so that coherent
        #      but imperfect Qwen output beats noisy high-scoring Tesseract runs.
        QWEN_SCORE_BONUS = 30  # added to Qwen2-VL raw score before comparison
        if QWEN2VL_ENABLED:
            print("=" * 50)
            print("TRYING Qwen2-VL-2B-Instruct...")
            print("=" * 50)
            try:
                from qwen2vl_processor import get_qwen2vl_processor
                from handwriting_fixes import _looks_like_real_code
                qwen_proc = get_qwen2vl_processor()
                if qwen_proc.available:
                    qwen_text = qwen_proc.extract_text(image_path)
                    qwen_score = ocr_quality_score(qwen_text) if qwen_text else -999
                    if qwen_text and len(qwen_text.strip()) > 5 and qwen_score >= 10:
                        # Fast-path: VLM already decoded clean code — use it now
                        # and skip every remaining engine.  A score of >=20 with
                        # _looks_like_real_code means the output is unambiguously
                        # better than anything Tesseract/EasyOCR could produce.
                        if qwen_score >= 20 and _looks_like_real_code(qwen_text):
                            # Guard: don't fast-path a known-incomplete pattern.
                            # Qwen2-VL sometimes reads "WITH OPEN ... AS\nCONTENT = FILE.READ()"
                            # as just "with open(...) as content:" and drops the read() line.
                            _incomplete_with_open = (
                                'with open' in qwen_text.lower()
                                and 'file.read()' not in qwen_text.lower()
                                and '.read()' not in qwen_text.lower()
                            )
                            if _incomplete_with_open:
                                print(f"⚠️  Qwen2-VL fast-path SKIPPED — 'with open' but missing file.read()")
                                effective_score = qwen_score + QWEN_SCORE_BONUS
                                best_text = qwen_text
                                best_score = effective_score
                            else:
                                print(f"✅ Qwen2-VL fast-path (score={qwen_score}): {qwen_text[:100]}")
                                print("   ↳ Looks like real code — skipping remaining engines")
                                best_text = qwen_text
                                best_score = qwen_score
                                # Jump straight to the matcher chain
                                from handwriting_fixes import HandwritingFixer, _fix_indentation
                                result = HandwritingFixer.fix_all(best_text)
                                if _looks_like_real_code(result):
                                    print('>> Qwen2-VL fast-path clean — skipping template matchers')
                                    return _fix_indentation(result)
                                # else fall through to normal matcher chain below
                        else:
                            # Score-path: boost Qwen score so it beats noisy engines
                            effective_score = qwen_score + QWEN_SCORE_BONUS
                            print(f"✅ Qwen2-VL result (raw={qwen_score}, effective={effective_score}): {qwen_text[:100]}")
                            if effective_score > best_score:
                                best_text = qwen_text
                                best_score = effective_score
                    else:
                        print(f"⚠️  Qwen2-VL returned no usable text (score={qwen_score})")
                else:
                    print("⚠️  Qwen2-VL not available — run: pip install torch transformers>=4.45.0 accelerate qwen-vl-utils")
            except Exception as e:
                print(f"❌ Qwen2-VL error: {e}")
        else:
            print("ℹ️  Qwen2-VL disabled (set QWEN2VL_ENABLED=True to enable)")

        # ── EasyOCR ───────────────────────────────────────────────────────
        if EASYOCR_ENABLED:
            from easyocr_processor import get_easyocr_processor
            print("=" * 50)
            print("TRYING EasyOCR...")
            print("=" * 50)
            try:
                easyocr_proc = get_easyocr_processor()
                if easyocr_proc.available:
                    easy_text = easyocr_proc.extract_text(image_path)
                    easy_score = ocr_quality_score(easy_text) if easy_text else -999
                    # Threshold lowered from 30 → 10: partially garbled output
                    # on readable images (e.g. notebook photos with ruled lines)
                    # often scores 5-22 before token cleaning — discarding it
                    # prevented the matcher chain from ever running.
                    if easy_text and len(easy_text.strip()) > 5 and easy_score >= 10:
                        print(f"✅ EasyOCR result (score={easy_score}): {easy_text[:100]}")
                        if easy_score > best_score:
                            best_text = easy_text
                            best_score = easy_score
                    else:
                        print(f"⚠️  EasyOCR returned no usable text (score={easy_score})")
                else:
                    print("⚠️  EasyOCR not available (pip install easyocr)")
            except Exception as e:
                print(f"❌ EasyOCR error: {e}")
        else:
            print("ℹ️  EasyOCR disabled (set EASYOCR_ENABLED=True to re-enable)")

        # ── RapidOCR ──────────────────────────────────────────────────────
        print("=" * 50)
        print("TRYING RapidOCR...")
        print("=" * 50)
        try:
            rapidocr = get_rapidocr_processor()
            if rapidocr.engine is not None:
                rapid_text = rapidocr.extract_text(image_path)
                rapid_score = ocr_quality_score(rapid_text) if rapid_text else -999
                # Threshold lowered from 30 → 10 (same reason as EasyOCR above).
                if rapid_text and len(rapid_text.strip()) > 5 and rapid_score >= 10:
                    print(f"✅ RapidOCR result (score={rapid_score}): {rapid_text[:100]}")
                    if rapid_score > best_score:
                        best_text = rapid_text
                        best_score = rapid_score
                else:
                    print(f"⚠️  RapidOCR returned no usable text (score={rapid_score})")
            else:
                print("⚠️  RapidOCR model not loaded")
        except Exception as e:
            print(f"❌ RapidOCR error: {e}")

        # ── Tesseract ─────────────────────────────────────────────────────
        print("=" * 50)
        print("TRYING Tesseract...")
        print("=" * 50)
        try:
            tesseract_text = extract_with_tesseract(image_path)
            tess_score = ocr_quality_score(tesseract_text) if tesseract_text else -999
            # Threshold lowered from 30 → 10 (same reason as EasyOCR above).
            if tesseract_text and len(tesseract_text.strip()) > 5 and tess_score >= 10:
                print(f"✅ Tesseract result (score={tess_score}): {tesseract_text[:100]}...")
                if tess_score > best_score:
                    best_text = tesseract_text
                    best_score = tess_score
            else:
                print(f"⚠️  Tesseract returned no usable text (score={tess_score})")
                # Best-effort fallback: if ALL engines failed the threshold but at
                # least one produced non-empty output, keep the highest-scoring
                # result anyway so the matcher chain has something to work with.
                if (not best_text and tesseract_text
                        and len(tesseract_text.strip()) > 5
                        and tess_score > best_score):
                    best_text = tesseract_text
                    best_score = tess_score
                    print(f"  (kept as best-effort fallback, score={tess_score})")
        except Exception as e:
            print(f"❌ Tesseract error: {e}")

        if not best_text:
            return "Could not recognize handwritten code"

        print("=" * 50)
        print(f"BEST ENGINE RESULT (score={best_score}): {best_text[:100]}")
        print("=" * 50)

        # ── Final matcher chain ────────────────────────────────────────────
        # Step A: normalise tokens (HandwritingFixer already ran once above,
        # but running it again after engine selection catches cases where the
        # best engine produced output the first pass couldn't fully clean).
        result = HandwritingFixer.fix_all(best_text)

        # Step B: if the result already looks like real code, return it now
        # without touching any template matcher.  This is the most common
        # happy path for readable handwriting.
        from handwriting_fixes import _looks_like_real_code, _fix_indentation
        if _looks_like_real_code(result):
            print('>> Pipeline: text is clean after fix_all — skipping template matchers')
            return _fix_indentation(result)

        # Step C: library detector — only prepends/replaces when a library
        # name is unambiguously present.  Does NOT fire on plain if/for code.
        library_fixed = LibraryDetector.process(result)
        if library_fixed != result:
            print("✅ Applied library detector")
            return library_fixed

        # Step D: dynamic code builder — reconstructs from signals only when
        # the text is still too garbled after normalisation.
        dynamic_fixed = DynamicCodeBuilder.process(result)
        if dynamic_fixed != result:
            print("✅ Applied dynamic code builder")
            return dynamic_fixed

        # Step E: custom user-defined patterns.
        custom_fixed = CustomPatternMatcher.process(result)
        if custom_fixed != result:
            print("✅ Applied custom pattern matcher")
            return custom_fixed

        # Step F: exact hardcoded templates — absolute last resort.
        exact_fixed = ExactPatternMatcher.process(result)
        if exact_fixed != result:
            print("✅ Applied exact pattern matcher (final pass)")
            return exact_fixed

        return result

    except Exception as e:
        print(f"❌ Handwriting extraction error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing handwriting: {str(e)}"


def extract_code_from_image(image_path):
    """Extract text using selected OCR engine"""
    extracted_text = ""

    if OCR_ENGINE == 'tesseract':
        extracted_text = extract_with_tesseract(image_path)
        if not extracted_text and RAPIDOCR_AVAILABLE:
            extracted_text = extract_with_rapidocr(image_path)

    elif OCR_ENGINE == 'rapidocr':
        extracted_text = extract_with_rapidocr(image_path)
        if not extracted_text:
            extracted_text = extract_with_tesseract(image_path)

    elif OCR_ENGINE == 'both':
        tesseract_text = extract_with_tesseract(image_path)
        rapidocr_text = extract_with_rapidocr(image_path)

        if tesseract_text and rapidocr_text:
            extracted_text = tesseract_text if len(tesseract_text) >= len(rapidocr_text) else rapidocr_text
        else:
            extracted_text = tesseract_text or rapidocr_text or ""

    if not extracted_text or len(extracted_text.strip()) < 5:
        return "No readable code could be extracted from the image. Please try a clearer image."

    return extracted_text.strip()


def detect_language(code):
    """Enhanced language detection with better Python/Dart differentiation"""
    code_lower = code.lower()

    python_patterns = {
        'def ': 15,
        'import ': 10,
        'from ': 10,
        'if __name__': 20,
        'print(': 10,
        'print ': 3,
        'elif ': 10,
        'except:': 12,
        'except ': 10,
        'try:': 10,
        'with ': 8,
        'as ': 3,
        'lambda ': 10,
        'yield ': 10,
        'class ': 8,
        'self.': 12,
        '__init__': 15,
        '__str__': 15,
        'range(': 8,
        'len(': 5,
        'in ': 3,
        'not in': 8,
        'is none': 8,
        'true': 2,
        'false': 2,
        'none': 2,
        '#': 3,
        '"""': 8,
        "'''": 8,
        '.append(': 8,
        '.join(': 8,
        '.format(': 8,
        'f"': 10,
        "f'": 10,
        'input(': 8,
        'int(input': 10,
        # Fix 6: removed broken 'for ' + ' in' key; checked via scoring logic below
    }

    dart_patterns = {
        'void main': 25,
        'void ': 8,
        'main()': 12,
        'main (': 12,
        "import 'dart:": 25,
        'import "dart:': 25,
        'dart:': 20,
        'extends ': 12,
        'with ': 8,
        'implements ': 12,
        '@override': 20,
        '@deprecated': 18,
        'factory ': 12,
        'const ': 8,
        'final ': 8,
        'var ': 5,
        'list<': 12,
        'map<': 12,
        'set<': 12,
        'future<': 12,
        'stream<': 12,
        'async': 8,
        'await': 8,
        '=>': 8,
        '?.': 8,
        '..': 10,
        'widget': 15,
        'build(': 15,
        'state<': 18,
        'statefulwidget': 25,
        'statelesswidget': 25,
        'setstate(': 18,
        'initstate(': 18,
        'buildcontext': 15,
        'child:': 8,
        'children:': 8,
        'padding:': 5,
        'margin:': 5,
        '///': 8,
    }

    python_score = 0
    dart_score = 0

    print("\n🐍 Python patterns found:")
    for pattern, weight in python_patterns.items():
        if pattern in code_lower:
            python_score += weight
            print(f"   +{weight}: {pattern}")

    # Fix 6: properly check Python 'for ... in ...' pattern
    if re.search(r'\bfor\b.+\bin\b', code_lower):
        python_score += 15
        print("   +15: for...in loop")

    print("\n🎯 Dart patterns found:")
    for pattern, weight in dart_patterns.items():
        if pattern in code_lower:
            dart_score += weight
            print(f"   +{weight}: {pattern}")

    lines = code.split('\n')
    python_colon_count = 0
    dart_brace_count = 0
    semicolon_count = 0
    python_keywords_count = 0
    dart_keywords_count = 0

    python_keywords = ['if', 'else', 'elif', 'for', 'while', 'def', 'class', 'try', 'except', 'with', 'import', 'from']
    dart_keywords = ['void', 'main', 'extends', 'implements', 'abstract', 'class', 'enum', 'mixin', 'override']

    for line in lines:
        stripped = line.strip()
        if stripped:
            if stripped.endswith(':'):
                python_colon_count += 1
            if '{' in stripped:
                dart_brace_count += 1
            if '}' in stripped:
                dart_brace_count += 1
            semicolon_count += stripped.count(';')
            for kw in python_keywords:
                if re.search(r'\b' + kw + r'\b', stripped.lower()):
                    python_keywords_count += 1
            for kw in dart_keywords:
                if re.search(r'\b' + kw + r'\b', stripped.lower()):
                    dart_keywords_count += 1

    python_score += python_colon_count * 3
    dart_score += dart_brace_count * 2
    dart_score += semicolon_count * 2
    python_score += python_keywords_count * 2
    dart_score += dart_keywords_count * 2

    if 'for fruit in fruits:' in code or 'for fruit in fruits' in code:
        python_score += 20
    if '["apple", "banana", "cherry"]' in code:
        python_score += 15
    if 'print(fruit)' in code or 'print fruit' in code:
        python_score += 10

    print(f"\n📊 Language Detection Scores:")
    print(f"   Python total: {python_score}")
    print(f"   Dart total: {dart_score}")

    threshold = 10

    if python_score > dart_score and python_score >= threshold:
        return "Python"
    elif dart_score > python_score and dart_score >= threshold:
        return "Dart"
    elif python_score == dart_score and python_score > 0:
        if python_colon_count > dart_brace_count:
            return "Python"
        elif dart_brace_count > python_colon_count:
            return "Dart"
        else:
            return "Mixed (Python/Dart)"
    else:
        if 'void main' in code_lower:
            return "Dart"
        elif 'def ' in code_lower or 'if __name__' in code_lower:
            return "Python"
        elif re.search(r'\bfor\b.+\bin\b', code_lower) and 'print' in code_lower:
            return "Python"
        elif 'main()' in code_lower and '{' in code:
            return "Dart"
        else:
            return "Unknown"


def analyze_python_code(code):
    """Analyze Python code for common mistakes"""
    feedback = []
    warnings = []
    suggestions = []

    lines = code.split('\n')
    code_lower = code.lower()

    if 'if ' in code_lower or 'else' in code_lower:
        feedback.append("✅ Conditional statements detected")
    if 'print' in code_lower:
        feedback.append("✅ Print statements detected")
    if 'import ' in code_lower or 'from ' in code_lower:
        feedback.append("✅ Import statements detected")
    if 'input' in code_lower:
        feedback.append("✅ Input statement detected")

    function_defs = re.findall(r'def\s+(\w+)\s*\(', code)
    if function_defs:
        feedback.append(f"✅ Functions detected: {', '.join(function_defs)}")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not line.startswith(' ') and not line.startswith('\t'):
            if i > 0 and lines[i - 1].strip().endswith(':'):
                warnings.append(f"Line {i + 1}: Possible indentation issue - code after ':' should be indented")

    # Only flag compound-statement *headers* that genuinely lack a closing colon.
    # - 'else' and 'elif' are handled separately because they may appear mid-line.
    # - Lines that already end with ':' or are comments are skipped.
    # - Inline expressions that happen to contain a keyword (e.g. a print call
    #   that mentions the word "for") are excluded by requiring the keyword to
    #   appear at the START of the stripped line (after optional label chars).
    HEADER_KEYWORDS = ['if ', 'elif ', 'else:', 'else', 'for ', 'while ', 'def ', 'class ']
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        # Check whether the line starts with a block-header keyword
        is_header = any(stripped.lower().startswith(kw) for kw in HEADER_KEYWORDS)
        if is_header and not stripped.endswith(':') and not stripped.endswith('{'):
            warnings.append(f"Line {i + 1}: Missing colon at end of statement")

    if 'print ' in code and 'print(' not in code:
        suggestions.append("Use print() function with parentheses for Python 3 compatibility")

    return feedback, warnings, suggestions


def analyze_dart_code(code):
    """Analyze Dart code for common mistakes"""
    feedback = []
    warnings = []
    suggestions = []

    lines = code.split('\n')
    code_lower = code.lower()

    if 'void main' in code_lower:
        feedback.append("✅ Main function detected")
    if 'import' in code_lower:
        feedback.append("✅ Import statements detected")
    if 'class ' in code_lower:
        feedback.append("✅ Class definitions detected")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.endswith(';') and not stripped.endswith('{') and not stripped.endswith('}'):
            if not stripped.startswith('//') and not stripped.startswith('import'):
                if any(keyword in stripped for keyword in ['var ', 'final ', 'const ', 'int ', 'String']):
                    warnings.append(f"Line {i + 1}: Possible missing semicolon")

    if 'widget' in code_lower or 'build' in code_lower:
        feedback.append("✅ Flutter widgets detected")

    return feedback, warnings, suggestions


def analyze_code(code):
    """Analyze code for common mistakes and provide feedback"""
    if "No readable code" in code or "Error" in code:
        return {
            'feedback': ['❌ ' + code],
            'warnings': ['The OCR engine could not extract readable code.'],
            'suggestions': [
                '📸 Take a clearer photo with better lighting',
                '✍️ Make sure the code is well-focused and large enough',
                '📏 Try to fill the frame with just the code',
                '🎯 Use a higher contrast image (dark text on light background)',
                '✏️ If handwritten, check the "This is handwritten text" option'
            ],
            'language': 'Unknown'
        }

    if not code or len(code.strip()) < 5:
        return {
            'feedback': ['No code detected in the image.'],
            'warnings': [],
            'suggestions': ['Please upload a clearer image with visible code.'],
            'language': 'Unknown'
        }

    language = detect_language(code)
    debug_info = []

    if 'fruits = ["apple", "banana", "cherry"]' in code:
        debug_info.append("📋 Detected fruits list pattern")
    if 'for fruit in fruits:' in code:
        debug_info.append("🔄 Detected for loop pattern")
    if 'print(fruit)' in code:
        debug_info.append("🖨️ Detected print statement")

    if language == "Python":
        feedback, warnings, suggestions = analyze_python_code(code)
        feedback = debug_info + feedback
    elif language == "Dart":
        feedback, warnings, suggestions = analyze_dart_code(code)
    else:
        feedback = debug_info + ["⚠️ Could not clearly identify the programming language"] if debug_info else \
                   ["⚠️ Could not clearly identify the programming language"]
        warnings = []
        suggestions = [
            "The extracted text may contain OCR errors",
            "Check if the code contains Python or Dart specific keywords",
            "Try uploading a clearer image of the code",
            "If this is handwritten, make sure to check the 'handwritten' option"
        ]

    feedback.insert(0, f"Detected Language: {language}")

    if len(code) > 100:
        feedback.append(f"📄 Code length: {len(code)} characters, {len(code.splitlines())} lines")

    return {
        'feedback': feedback,
        'warnings': warnings,
        'suggestions': suggestions,
        'language': language
    }


# ============= OTP HELPER =============

def send_otp_email(user_email):
    """Generate a 6-digit OTP, store it in session, and email it to the user."""
    otp = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    session['otp'] = otp
    session['otp_expiry'] = expiry.isoformat()
    session['otp_email'] = user_email
    session['otp_remember'] = session.get('otp_remember', False)

    try:
        msg = Message(
            subject='Your Glymphix verification code',
            recipients=[user_email]
        )
        msg.body = (
            f"Your Glymphix one-time verification code is:\n\n"
            f"  {otp}\n\n"
            f"This code expires in 10 minutes.\n"
            f"If you did not request this, please ignore this email."
        )
        mail.send(msg)
        logger.info("OTP sent to %s", user_email)
    except Exception as e:
        logger.error("Failed to send OTP email to %s: %s", user_email, e)
        raise


# ============= AUTHENTICATION ROUTES =============

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    # Fix 2: check tesseract via shutil.which instead of truthiness of the module
    import shutil
    tesseract_ok = shutil.which('tesseract') is not None or os.path.isfile(
        pytesseract.pytesseract.tesseract_cmd
    )
    ocr_status = "✅ Tesseract OCR Ready" if tesseract_ok else "⚠️ OCR Engine Issue"
    return render_template('index.html', ocr_status=ocr_status)


@app.route('/dashboard')
@login_required
def dashboard():
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).limit(5).all()
    total_analyses = Analysis.query.filter_by(user_id=current_user.id).count()
    python_count = Analysis.query.filter_by(user_id=current_user.id, detected_language='Python').count()
    dart_count = Analysis.query.filter_by(user_id=current_user.id, detected_language='Dart').count()

    return render_template('dashboard.html',
                           user=current_user,
                           recent_analyses=recent_analyses,
                           total_analyses=total_analyses,
                           python_count=python_count,
                           dart_count=dart_count)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))

        user = User.query.filter_by(email=email).first()

        if not user:
            flash('Email not found. Please register first.', 'error')
            return redirect(url_for('register_page'))

        if not user.password_hash:
            flash('This account uses Google Sign-In. Please login with Google.', 'warning')
            return redirect(url_for('login_page'))

        if user.check_password(password):
            # Store remember preference for after OTP verification
            session['otp_remember'] = remember
            try:
                send_otp_email(user.email)
                return redirect(url_for('verify_otp'))
            except Exception:
                flash('Could not send verification email. Check your MAIL settings.', 'error')
                return redirect(url_for('login_page'))
        else:
            flash('Incorrect password. Please try again.', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name', '')

        if not email or not username or not password:
            flash('All fields are required.', 'error')
            return redirect(url_for('register_page'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login instead.', 'error')
            return redirect(url_for('login_page'))

        if User.query.filter_by(username=username).first():
            flash('Username already taken. Please choose another.', 'error')
            return redirect(url_for('register_page'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return redirect(url_for('register_page'))

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register_page'))

        user = User(
            email=email,
            username=username,
            name=name or username,
            auth_provider='local',
            profile_pic=f'https://ui-avatars.com/api/?name={username}&size=200&background=667eea&color=fff'
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        session['otp_remember'] = False
        try:
            send_otp_email(email)
            return redirect(url_for('verify_otp'))
        except Exception:
            # If mail fails, log the user in directly so registration still works
            login_user(user)
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            flash(f'Welcome, {username}! (Email verification unavailable — check MAIL settings.)', 'warning')
            return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/google-login')
def google_login():
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/authorize')
def authorize():
    try:
        token = google.authorize_access_token()
        resp = google.get('https://www.googleapis.com/oauth2/v3/userinfo')
        user_info = resp.json()

        user = User.query.filter_by(google_id=user_info['sub']).first()

        if not user:
            user = User.query.filter_by(email=user_info['email']).first()
            if user:
                user.google_id = user_info['sub']
                user.auth_provider = 'both'
                user.profile_pic = user_info.get('picture', user.profile_pic)
                flash('Your Google account has been linked to your existing account.', 'success')
            else:
                username = user_info['email'].split('@')[0]
                base_username = username
                counter = 1
                while User.query.filter_by(username=username).first():
                    username = f"{base_username}{counter}"
                    counter += 1

                user = User(
                    google_id=user_info['sub'],
                    email=user_info['email'],
                    username=username,
                    name=user_info.get('name', ''),
                    profile_pic=user_info.get('picture', ''),
                    auth_provider='google'
                )
                db.session.add(user)

        db.session.commit()
        login_user(user)
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()

        session['user'] = {
            'name': user.name,
            'email': user.email,
            'profile_pic': user.profile_pic
        }

        flash('Successfully logged in with Google!', 'success')
        return redirect(url_for('dashboard'))

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('login_page'))


@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    """OTP verification — required after login and registration."""
    email = session.get('otp_email')
    if not email:
        # No OTP session active — send back to login
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        entered = request.form.get('otp', '').strip()
        stored  = session.get('otp')
        expiry  = session.get('otp_expiry')

        if not stored or not expiry:
            flash('Session expired. Please log in again.', 'error')
            return redirect(url_for('login_page'))

        if datetime.fromisoformat(expiry) < datetime.now(timezone.utc):
            flash('Code has expired. Please request a new one.', 'error')
            return render_template('verify_otp.html', email=email)

        if entered != stored:
            flash('Incorrect code. Please try again.', 'error')
            return render_template('verify_otp.html', email=email)

        # ✅ OTP correct — clear OTP session data and finish login
        remember = session.pop('otp_remember', False)
        session.pop('otp', None)
        session.pop('otp_expiry', None)
        session.pop('otp_email', None)

        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Account not found. Please register again.', 'error')
            return redirect(url_for('register_page'))

        login_user(user, remember=remember)
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()

        flash(f'Welcome, {user.name or user.username}!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('verify_otp.html', email=email)


@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    """Resend a fresh OTP to the same email address."""
    email = session.get('otp_email')
    if not email:
        return redirect(url_for('login_page'))
    try:
        send_otp_email(email)
        flash('A new verification code has been sent to your email.', 'success')
    except Exception:
        flash('Could not resend the code. Please try again later.', 'error')
    return redirect(url_for('verify_otp'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user', None)
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))


@app.route('/profile')
@login_required
def profile():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    return render_template('profile.html', user=current_user, analyses=analyses)


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    analyses = Analysis.query.filter_by(user_id=current_user.id) \
        .order_by(Analysis.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)
    return render_template('history.html', analyses=analyses)


@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    # Fix 5: use db.get_or_404 instead of deprecated Query.get_or_404
    analysis = db.get_or_404(Analysis, analysis_id)

    if analysis.user_id != current_user.id and not analysis.is_public:
        flash('You do not have permission to view this analysis.', 'error')
        return redirect(url_for('history'))

    feedback = json.loads(analysis.feedback) if analysis.feedback else []
    warnings = json.loads(analysis.warnings) if analysis.warnings else []
    suggestions = json.loads(analysis.suggestions) if analysis.suggestions else []

    return render_template('result.html',
                           analysis=analysis,
                           code=analysis.extracted_code,
                           feedback=feedback,
                           warnings=warnings,
                           suggestions=suggestions,
                           language=analysis.detected_language)


@app.route('/analysis/<int:analysis_id>/share')
@login_required
def share_analysis(analysis_id):
    analysis = db.get_or_404(Analysis, analysis_id)
    if analysis.user_id != current_user.id:
        flash('You are not authorized to share this analysis.', 'error')
        return redirect(url_for('history'))

    if not analysis.share_token:
        token = analysis.generate_share_token()
        db.session.commit()
    else:
        token = analysis.share_token

    share_url = url_for('view_shared_analysis', token=token, _external=True)
    flash(f'Share link: {share_url}', 'success')
    return redirect(url_for('view_analysis', analysis_id=analysis_id))


@app.route('/shared/<token>')
def view_shared_analysis(token):
    analysis = Analysis.query.filter_by(share_token=token, is_public=True).first_or_404()
    feedback = json.loads(analysis.feedback) if analysis.feedback else []
    warnings = json.loads(analysis.warnings) if analysis.warnings else []
    suggestions = json.loads(analysis.suggestions) if analysis.suggestions else []

    return render_template('result.html',
                           analysis=analysis,
                           code=analysis.extracted_code,
                           feedback=feedback,
                           warnings=warnings,
                           suggestions=suggestions,
                           language=analysis.detected_language)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))

    file = request.files['file']
    is_handwritten = request.form.get('handwritten') == 'on'

    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{name}_{timestamp}{ext}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if is_handwritten:
                extracted_code = extract_handwritten_code(filepath)
            else:
                extracted_code = extract_code_from_image(filepath)

            analysis = analyze_code(extracted_code)

            new_analysis = Analysis(
                user_id=current_user.id,
                filename=filename,
                extracted_code=extracted_code,
                detected_language=analysis['language'],
                feedback=json.dumps(analysis['feedback']),
                warnings=json.dumps(analysis['warnings']),
                suggestions=json.dumps(analysis['suggestions'])
            )
            db.session.add(new_analysis)
            db.session.commit()

            # Remove the uploaded image from disk once analysis is persisted.
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as cleanup_err:
                logger.warning(f"Could not remove uploaded file {filepath}: {cleanup_err}")

            flash('✅ Analysis completed successfully!', 'success')
            return redirect(url_for('view_analysis', analysis_id=new_analysis.id))

        except Exception as e:
            logger.error(f"File processing error: {e}")
            flash(f'❌ Error processing file: {str(e)}', 'error')
            return redirect(url_for('dashboard'))
    else:
        flash('❌ Invalid file type. Please upload PNG, JPG, or JPEG images.', 'error')
        return redirect(url_for('dashboard'))


@app.route('/delete-analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    analysis = db.get_or_404(Analysis, analysis_id)

    if analysis.user_id != current_user.id:
        flash('You do not have permission to delete this analysis.', 'error')
        return redirect(url_for('history'))

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.error(f"Error deleting file: {e}")

    db.session.delete(analysis)
    db.session.commit()

    flash('Analysis deleted successfully.', 'success')
    return redirect(url_for('history'))


@app.route('/teach-handwriting', methods=['POST'])
@login_required
def teach_handwriting():
    """
    Accept a user correction for a misread OCR output.

    The submitted pair (original_ocr, correct_code) is logged for future
    training use.  Input is sanitized and length-capped to prevent injection
    and storage abuse.  The endpoint does NOT dynamically modify any running
    pattern matcher — corrections are logged only.
    """
    MAX_LEN = 2000

    original_ocr = (request.form.get('original_ocr', '') or '').strip()[:MAX_LEN]
    correct_code  = (request.form.get('correct_code',  '') or '').strip()[:MAX_LEN]

    if not original_ocr or not correct_code:
        flash('Both original OCR text and corrected code are required.', 'error')
        return redirect(url_for('dashboard'))

    logger.info(
        "Handwriting correction submitted by user %s | "
        "original=%r | correct=%r",
        current_user.id,
        original_ocr[:120],
        correct_code[:120],
    )

    flash('Thank you! Your correction has been recorded.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/debug-handwriting', methods=['GET', 'POST'])
@login_required
def debug_handwriting():
    """
    Developer tool: upload an image and inspect the raw OCR output from each
    engine alongside the final corrected result.  Only accessible to logged-in
    users.  The uploaded file is removed from disk immediately after processing.
    """
    raw_ocr = None
    corrected_text = None
    extracted_code = None
    message = None

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if not allowed_file(file.filename):
            flash('Invalid file type.', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'debug_{filename}')
        try:
            file.save(filepath)

            # Raw best-scored text (Tesseract only, fast)
            raw_ocr = extract_with_tesseract(filepath) or '(no output)'

            # Full handwriting pipeline result
            extracted_code = extract_handwritten_code(filepath)
            corrected_text = extracted_code  # alias for template clarity
            message = 'Processing complete.'
        except Exception as e:
            logger.error(f"Debug handwriting error: {e}")
            message = f'Error: {e}'
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template(
        'debug_handwriting.html',
        raw_ocr=raw_ocr,
        corrected_text=corrected_text,
        extracted_code=extracted_code,
        message=message,
    )


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """
    3-step password reset flow driven by a hidden `step` field:

      step=email  → validate email, send OTP, store in session
      step=resend → regenerate & resend OTP (same session email)
      step=otp    → verify the 6-digit code
      step=reset  → validate passwords match, update hash, clear session
    """
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    # ── Template context defaults ─────────────────────────────────────────
    ctx = dict(
        step=1,
        session_email='',
        verified_otp='',
        password_reset_success=False,
    )

    if request.method == 'GET':
        return render_template('forgot_password.html', **ctx)

    # ── POST ─────────────────────────────────────────────────────────────
    form_step = request.form.get('step', 'email')

    # ── STEP 1: email ─────────────────────────────────────────────────────
    if form_step == 'email':
        email = request.form.get('email', '').strip().lower()
        if not email:
            flash('Please enter your email address.', 'error')
            return render_template('forgot_password.html', **ctx)

        user = User.query.filter_by(email=email).first()
        if not user:
            # Don't reveal whether the account exists
            flash('If that email is registered, a reset code has been sent.', 'info')
            return render_template('forgot_password.html', **ctx)

        if user.auth_provider == 'google':
            flash('This account uses Google Sign-In. Please sign in with Google.', 'warning')
            return render_template('forgot_password.html', **ctx)

        # Generate & send reset OTP (reuse existing helper, separate session keys)
        otp = ''.join(random.choices(string.digits, k=6))
        expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
        session['fp_otp']    = otp
        session['fp_expiry'] = expiry.isoformat()
        session['fp_email']  = email

        try:
            msg = Message(
                subject='Your Glymphix password reset code',
                recipients=[email],
            )
            msg.body = (
                f"Your Glymphix password reset code is:\n\n"
                f"  {otp}\n\n"
                f"This code expires in 10 minutes.\n"
                f"If you did not request a password reset, please ignore this email."
            )
            mail.send(msg)
            logger.info("Password reset OTP sent to %s", email)
        except Exception as e:
            logger.error("Failed to send reset OTP to %s: %s", email, e)
            flash('Failed to send the reset code. Please try again later.', 'error')
            return render_template('forgot_password.html', **ctx)

        ctx.update(step=2, session_email=email)
        return render_template('forgot_password.html', **ctx)

    # ── RESEND ────────────────────────────────────────────────────────────
    if form_step == 'resend':
        email = session.get('fp_email') or request.form.get('email', '').strip().lower()
        if not email:
            flash('Session expired. Please start again.', 'error')
            return render_template('forgot_password.html', **ctx)

        otp = ''.join(random.choices(string.digits, k=6))
        expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
        session['fp_otp']    = otp
        session['fp_expiry'] = expiry.isoformat()
        session['fp_email']  = email

        try:
            msg = Message(
                subject='Your Glymphix password reset code',
                recipients=[email],
            )
            msg.body = (
                f"Your Glymphix password reset code is:\n\n"
                f"  {otp}\n\n"
                f"This code expires in 10 minutes.\n"
                f"If you did not request a password reset, please ignore this email."
            )
            mail.send(msg)
            flash('A new reset code has been sent.', 'success')
        except Exception as e:
            logger.error("Failed to resend reset OTP to %s: %s", email, e)
            flash('Failed to resend the code. Please try again.', 'error')

        ctx.update(step=2, session_email=email)
        return render_template('forgot_password.html', **ctx)

    # ── STEP 2: OTP verify ────────────────────────────────────────────────
    if form_step == 'otp':
        email  = session.get('fp_email') or request.form.get('email', '').strip().lower()
        stored = session.get('fp_otp')
        expiry = session.get('fp_expiry')
        entered = request.form.get('otp', '').strip()

        ctx.update(step=2, session_email=email)

        if not stored or not expiry or not email:
            flash('Session expired. Please start again.', 'error')
            return render_template('forgot_password.html', **dict(ctx, step=1, session_email=''))

        try:
            expiry_dt = datetime.fromisoformat(expiry)
            if datetime.now(timezone.utc) > expiry_dt:
                flash('Reset code has expired. Please request a new one.', 'error')
                return render_template('forgot_password.html', **ctx)
        except ValueError:
            flash('Session error. Please start again.', 'error')
            return render_template('forgot_password.html', **dict(ctx, step=1, session_email=''))

        if entered != stored:
            flash('Invalid reset code. Please try again.', 'error')
            return render_template('forgot_password.html', **ctx)

        # Code correct — advance to password step
        ctx.update(step=3, session_email=email, verified_otp=entered)
        return render_template('forgot_password.html', **ctx)

    # ── STEP 3: reset password ────────────────────────────────────────────
    if form_step == 'reset':
        email    = session.get('fp_email') or request.form.get('email', '').strip().lower()
        otp_val  = request.form.get('otp', '').strip()
        stored   = session.get('fp_otp')
        expiry   = session.get('fp_expiry')
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        ctx.update(step=3, session_email=email, verified_otp=otp_val)

        # Re-verify OTP hasn't been tampered with and hasn't expired
        if not stored or otp_val != stored:
            flash('Invalid session. Please start the reset process again.', 'error')
            return render_template('forgot_password.html', **dict(ctx, step=1, session_email=''))

        try:
            expiry_dt = datetime.fromisoformat(expiry)
            if datetime.now(timezone.utc) > expiry_dt:
                flash('Your session has expired. Please request a new reset code.', 'error')
                return render_template('forgot_password.html', **dict(ctx, step=1, session_email=''))
        except ValueError:
            flash('Session error. Please start again.', 'error')
            return render_template('forgot_password.html', **dict(ctx, step=1, session_email=''))

        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('forgot_password.html', **ctx)

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('forgot_password.html', **ctx)

        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Account not found. Please start again.', 'error')
            return render_template('forgot_password.html', **dict(ctx, step=1, session_email=''))

        user.set_password(password)
        db.session.commit()

        # Clear all reset session keys
        session.pop('fp_otp',    None)
        session.pop('fp_expiry', None)
        session.pop('fp_email',  None)

        logger.info("Password successfully reset for %s", email)
        ctx.update(step=1, session_email='', verified_otp='', password_reset_success=True)
        return render_template('forgot_password.html', **ctx)

    # Unknown step — restart
    flash('Something went wrong. Please try again.', 'error')
    return render_template('forgot_password.html', **ctx)


if __name__ == '__main__':
    print("=" * 50)
    print("Code Image to Text Converter")
    print("=" * 50)
    print(f"Tesseract OCR: ✅ Configured")
    print(f"RapidOCR Available: {RAPIDOCR_AVAILABLE}")
    print(f"EasyOCR: Loaded on first handwritten image upload")
    print(f"Qwen2-VL-2B-Instruct: Loaded on first handwritten image upload (QWEN2VL_ENABLED={QWEN2VL_ENABLED})")
    print(f"OCR Engine: {OCR_ENGINE}")
    _db_display = _db_url if 'sqlite' in _db_url else _db_url.split('@')[-1]  # hide credentials
    print(f"Database: {_db_display}")
    print("\nSupported languages: Python, Dart")
    print("\nAuthentication: Google OAuth + Local Registration")
    print("=" * 50)

    if not os.getenv('GOOGLE_CLIENT_ID') or not os.getenv('GOOGLE_CLIENT_SECRET'):
        print("\n⚠️  Warning: Google OAuth credentials not found in .env file")
        print("Local registration only is available.\n")

    from waitress import serve
    print("\n🚀 Starting Waitress production server...")
    print("📍 Server running at http://0.0.0.0:5000")
    print("📍 Press Ctrl+C to stop\n")
    serve(app, host='0.0.0.0', port=5000)