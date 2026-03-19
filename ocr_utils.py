import re

# ---------------------------------------------------------------------------
# Code tokens that are strong positive signals for OCR quality.
# Each entry is (pattern, points).  Points are cumulative — the same token
# type can score multiple times if it appears multiple times in the text.
# ---------------------------------------------------------------------------

# Keywords common to Python and Dart
_KEYWORD_SCORE = [
    (r'\bif\b',       4),
    (r'\belse\b',     4),
    (r'\belif\b',     4),
    (r'\bfor\b',      4),
    (r'\bwhile\b',    4),
    (r'\bdef\b',      5),
    (r'\bclass\b',    5),
    (r'\breturn\b',   4),
    (r'\bimport\b',   4),
    (r'\bfrom\b',     3),
    (r'\bin\b',       2),
    (r'\bnot\b',      2),
    (r'\band\b',      2),
    (r'\bor\b',       2),
    (r'\bTrue\b',     3),
    (r'\bFalse\b',    3),
    (r'\bNone\b',     3),
    (r'\bpass\b',     3),
    (r'\bbreak\b',    3),
    (r'\bcontinue\b', 3),
    (r'\bwith\b',     3),
    (r'\bas\b',       2),
    (r'\btry\b',      4),
    (r'\bexcept\b',   4),
    # Python builtins
    (r'\bprint\s*\(', 5),
    (r'\binput\s*\(', 5),
    (r'\brange\s*\(', 4),
    (r'\blen\s*\(',   4),
    (r'\bint\s*\(',   4),
    (r'\bstr\s*\(',   3),
    (r'\bfloat\s*\(', 3),
    (r'\bappend\s*\(', 4),
    # Python stdlib / common identifiers
    (r'\brandom\b',    3),
    (r'\brandint\s*\(', 4),
    (r'\bsecret\b',    2),
    (r'\bguess\b',     2),
    # Dart keywords
    (r'\bvoid\b',     4),
    (r'\bmain\b',     3),
    (r'\bvar\b',      3),
    (r'\bfinal\b',    3),
    (r'\bconst\b',    3),
]

# Syntax characters — each occurrence adds points
_SYNTAX_SCORE = [
    (r'==',           3),
    (r'!=',           3),
    (r'>=',           3),
    (r'<=',           3),
    (r'=>',           3),   # Dart fat arrow
    (r'\+=',          3),
    (r'-=',           3),
    (r'%',            2),
    (r'\(',           1),
    (r'\)',           1),
    (r'\[',           2),
    (r'\]',           2),
    (r'\{',           2),
    (r'\}',           2),
    (r':$',           3),   # line ending in colon (Python block opener)
    (r';$',           2),   # line ending in semicolon (Dart)
    (r'"[^"]*"',      2),   # double-quoted string
    (r"'[^']*'",      2),   # single-quoted string
    (r'f"[^"]*"',     3),   # f-string
    (r"f'[^']*'",     3),
    (r'#[^\n]+',      2),   # comment
    (r'///[^\n]+',    2),   # Dart doc comment
]

# Structural patterns that are strong signals of real code
_STRUCTURAL_SCORE = [
    (r'^\s{4,}',      2),   # indented line (4+ spaces = Python block body)
    (r'^\s{2,}',      1),   # indented line (2+ spaces = Dart block body)
    (r'[a-zA-Z_]\w*\s*=\s*\S', 3),  # variable assignment
    (r'def\s+\w+\s*\(', 6),          # function definition
    (r'class\s+\w+',   6),            # class definition
    (r'void\s+main\s*\(\)', 8),       # Dart entry point
    (r'if\s+\w.*:$',   5),            # Python if statement
    (r'for\s+\w.*:$',  5),            # Python for statement
    (r'print\s*\(.*\)', 4),           # print call
]

# Patterns that indicate OCR noise — deduct points
_NOISE_PENALTY = [
    (r'[^\x00-\x7F]',  -1),  # non-ASCII characters (OCR hallucinations)
    (r'\b[A-Z]{5,}\b', -2),  # long ALL-CAPS word (rarely real code)
    (r'(.)\1{4,}',     -3),  # character repeated 5+ times (noise streak)
    (r'\|{2,}',        -3),  # multiple pipe chars (ruled-line artefact)
    (r'_{3,}',         -2),  # long underscore run
    (r'^\s*[|lI!]{3,}', -4), # line of pipe/l/I chars (ruled-line misread)
]

# Hard minimum: if fewer than this many characters, score is 0
_MIN_LENGTH = 4

# Score is capped at this value to keep comparisons stable
_MAX_SCORE = 200


def ocr_quality_score(text: str) -> int:
    """
    Score OCR output on how much it resembles real code.

    Returns an integer where:
        < 0   — almost certainly noise / garbled output
        0–19  — very little recognisable structure (below usable threshold)
        20–49 — partial recognition, may be usable
        50+   — strong code signal, high confidence result

    The threshold used in app.py to accept a result as "usable" is >= 20.
    Scores are compared between engines to pick the best preprocessing
    variant and the best overall OCR engine for a given image.
    """
    if not text or len(text.strip()) < _MIN_LENGTH:
        return 0

    score = 0
    lines = text.splitlines()

    # ── Per-token keyword scoring ────────────────────────────────────────────
    for pattern, points in _KEYWORD_SCORE:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        score += len(matches) * points

    # ── Syntax character scoring ─────────────────────────────────────────────
    for pattern, points in _SYNTAX_SCORE:
        matches = re.findall(pattern, text, re.MULTILINE)
        score += len(matches) * points

    # ── Structural pattern scoring (per line) ────────────────────────────────
    for line in lines:
        for pattern, points in _STRUCTURAL_SCORE:
            if re.search(pattern, line, re.MULTILINE):
                score += points

    # ── Noise penalties ──────────────────────────────────────────────────────
    for pattern, penalty in _NOISE_PENALTY:
        matches = re.findall(pattern, text, re.MULTILINE)
        score += len(matches) * penalty  # penalty values are already negative

    # ── Pipe / ruled-line noise penalty ─────────────────────────────────────
    # Tesseract frequently produces lines like "| | a \ a > | ' { } : | } !"
    # on images it can't read.  These contain syntax chars ({, }, :, () that
    # score positively above, so we must counter that here.
    #
    # Penalty A: any line where more than 50 % of characters are non-word
    #   non-space characters (pipes, backslashes, angle brackets etc.) is
    #   almost certainly ruled-line noise, not code.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        non_word = len(re.findall(r'[^\w\s]', stripped))
        if len(stripped) > 4 and non_word / len(stripped) > 0.50:
            score -= 8

    # Penalty B: lines that consist primarily of pipe/backslash/angle-bracket
    #   characters with only short scattered letters — classic Tesseract noise.
    _PIPE_LINE = re.compile(r'^[\s|\\/<>!@#^*~`\'\"{}()\[\],.?;:®©°±×÷]{4,}$')
    for line in lines:
        if _PIPE_LINE.match(line.strip()):
            score -= 12

    # Penalty C: if more than 40 % of ALL non-whitespace characters in the
    #   entire text are non-alphanumeric, the text is predominantly noise.
    all_nonws = re.sub(r'\s', '', text)
    if all_nonws:
        non_alnum = len(re.findall(r'[^\w]', all_nonws))
        if non_alnum / len(all_nonws) > 0.40:
            score -= 20

    # ── Partial-credit for OCR-garbled keyword variants ──────────────────────
    # When an engine produces "agc >= 16:" instead of "age >= 18:" the strict
    # keyword patterns above score 0 for those tokens.  These patterns award
    # half-credit for garbled forms that the HandwritingFixer will later clean,
    # so partially-correct output isn't discarded before it reaches the fixers.
    _PARTIAL_CREDIT = [
        (r'\b(agc|ogc|agc|og[ce])\b',            2),   # age
        (r'\b(nimbtr|humber|lnvnber|lwvmbt|nvmber|nwmber|nwriber)\b', 2),  # number
        (r'\b(inpul|inpuk|iuput|linput|cinput)\b',              2),   # input
        (r'\b(prut|prmt|pr1nt|prin\+)\b',         2),   # print
        (r'\b(relurn|refurn|rfurn)\b',             2),   # return
        (r'\b(whlie|whi1e|whil3)\b',              2),   # while
        (r'\b(voidmain)\b',                        3),   # void main (fused)
        (r'\b(randem|randam|randan)\b',            2),   # random
        (r'\b(randit)\b',                          2),   # randint
        (r'\b(cf|1f)\b',                           1),   # if
        (r'>=\s*1[68]',                            3),   # >= 18 (common misread: 16)
        (r'\b(adolt|aduit|aduit)\b',              2),   # adult
        (r'\b(frvit|frvits)\b',                   2),   # fruit/fruits
    ]
    for pattern, points in _PARTIAL_CREDIT:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        score += len(matches) * points


    non_empty_lines = [l for l in lines if l.strip()]
    if len(non_empty_lines) >= 3:
        score += 5
    if len(non_empty_lines) >= 6:
        score += 5

    # ── Complete-line bonus: reward lines that look like full statements ─────
    # A line with a keyword + opening paren/brace/colon is much stronger
    # evidence than the same tokens scattered across many short fragments.
    complete_line = re.compile(
        r'^\s*(for\s*\(|if\s*\(|void\s+main|print\s*\(|def\s+\w|'
        r'[a-z_]\w*\s*=\s*\S|while\s*\(|class\s+\w|return\s+\S)',
        re.MULTILINE
    )
    score += len(complete_line.findall(text)) * 8

    # ── Short-line penalty: many 1-3 char lines = OCR fragmented the output ──
    short_lines = [l for l in lines if l.strip() and len(l.strip()) <= 3]
    score -= len(short_lines) * 3

    return max(-999, min(_MAX_SCORE, score))
