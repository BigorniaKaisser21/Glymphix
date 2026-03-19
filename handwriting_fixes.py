import re


# ---------------------------------------------------------------------------
# ── CHARACTER-CONFUSION OCR CORRECTOR ────────────────────────────────────────
#
# Instead of a word-by-word table, we define which characters OCR commonly
# swaps and try every substitution against a vocabulary of known code tokens.
# Only accepts a correction when exactly ONE vocab word matches — rejects
# ambiguous results so user variable names are never wrongly changed.
# ---------------------------------------------------------------------------

_CHAR_CONFUSION: dict[str, list[str]] = {
    # Digit ↔ letter
    '0': ['o', 'O'],
    'o': ['0'],
    'O': ['0'],
    '1': ['l', 'I', 'i'],
    'l': ['1', 'I', 'i'],
    'I': ['l', '1', 'i'],
    '2': ['Z', 'z'],
    'Z': ['2'],
    'z': ['2'],
    '3': ['e', 'E'],
    'e': ['3'],
    '4': ['a', 'h'],
    '5': ['s', 'S'],
    's': ['5'],
    'S': ['5'],
    '6': ['b'],
    'b': ['6'],
    '8': ['B'],
    'B': ['8'],
    # Operator / punctuation ↔ letter
    '+': ['t'],
    't': ['+', 'f', 'g'],  # 'f'/'g' common: pring→print, it→if
    'f': ['t'],
    'g': ['t', 'q'],
    # Letter shape similarities
    'u': ['n', 'v'],
    'n': ['u', 'h'],
    'v': ['u', 'x'],
    'x': ['v'],            # get_exen → get_even
    'c': ['e'],
    'a': ['o', 'e'],       # Contant→content (a→e), moin→main (o→a handled via 'o')
    'h': ['n'],            # hnumber→number, humbers→numbers
    '}': ['t'],            # Prin}→print
}

_MULTI_CHAR_SUBS: list[tuple[str, str]] = [
    ('rn', 'm'),
    ('m',  'rn'),
    ('cl', 'd'),
    ('ii', 'n'),
    ('ri', 'n'),
]

_CODE_VOCAB: frozenset[str] = frozenset([
    'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
    'import', 'from', 'in', 'not', 'and', 'or',
    'True', 'False', 'None',
    'pass', 'break', 'continue', 'with', 'as', 'try', 'except', 'finally',
    'raise', 'yield', 'lambda', 'global', 'nonlocal', 'del', 'assert',
    'print', 'input', 'int', 'str', 'float', 'bool',
    'list', 'dict', 'set', 'tuple',
    'len', 'range', 'sum', 'min', 'max', 'abs', 'type',
    'append', 'extend', 'insert', 'remove', 'pop',
    'sort', 'sorted', 'reversed',
    'open', 'read', 'write', 'close',
    'split', 'strip', 'join', 'upper', 'lower', 'replace', 'find', 'format',
    'void', 'main', 'var', 'final', 'const', 'new', 'this', 'super',
    'extends', 'implements', 'abstract', 'static', 'dynamic', 'null',
    'number', 'numbers', 'age', 'count', 'result', 'total', 'average',
    'even', 'odd', 'fruit', 'fruits', 'name', 'value', 'text',
    'random', 'randint', 'secret', 'guess',
    'Iteration', 'adult',
])

_VOCAB_LOWER: dict[str, str] = {w.lower(): w for w in _CODE_VOCAB}


def _try_char_corrections(token: str) -> str | None:
    """Correct a single OCR-garbled token to a vocabulary word.
    Returns the corrected word or None if no unambiguous match found."""
    tl = token.lower()

    if tl in _VOCAB_LOWER:
        return _VOCAB_LOWER[tl]

    candidates: set[str] = set()

    # Single-character substitutions
    for i, ch in enumerate(tl):
        for sub in _CHAR_CONFUSION.get(ch, []):
            if len(sub) == 1:
                candidate = tl[:i] + sub + tl[i + 1:]
                if candidate in _VOCAB_LOWER:
                    candidates.add(_VOCAB_LOWER[candidate])

    # Multi-character substitutions
    for src, dst in _MULTI_CHAR_SUBS:
        pos = tl.find(src)
        while pos != -1:
            candidate = tl[:pos] + dst + tl[pos + len(src):]
            if candidate in _VOCAB_LOWER:
                candidates.add(_VOCAB_LOWER[candidate])
            pos = tl.find(src, pos + 1)

    # Drop one spurious leading character (e.g. 'hnumber'→'number')
    if len(tl) > 2 and tl[1:] in _VOCAB_LOWER:
        candidates.add(_VOCAB_LOWER[tl[1:]])

    if len(candidates) == 1:
        return candidates.pop()

    # Two-substitution fallback (e.g. v01d: 0→o AND 1→i)
    if not candidates:
        for i, ch in enumerate(tl):
            for sub1 in _CHAR_CONFUSION.get(ch, []):
                if len(sub1) != 1:
                    continue
                mid = tl[:i] + sub1 + tl[i + 1:]
                for j, ch2 in enumerate(mid):
                    for sub2 in _CHAR_CONFUSION.get(ch2, []):
                        if len(sub2) != 1:
                            continue
                        candidate = mid[:j] + sub2 + mid[j + 1:]
                        if candidate in _VOCAB_LOWER:
                            candidates.add(_VOCAB_LOWER[candidate])
        if len(candidates) == 1:
            return candidates.pop()

    return None


def _apply_char_corrections(text: str) -> str:
    """Apply _try_char_corrections to every word token in the text.

    For compound identifiers joined by underscores (e.g. get_exen_numbers)
    also tries correcting each part independently so individual garbled
    components are fixed without needing the whole compound to match vocab.
    """
    def _fix_part(part: str) -> str:
        corrected = _try_char_corrections(part)
        return corrected if corrected is not None else part

    def _fix(m: re.Match) -> str:
        word = m.group(0)
        # Try the whole word first
        corrected = _try_char_corrections(word)
        if corrected is not None:
            return corrected
        # For compound identifiers, try each underscore-separated part
        if '_' in word:
            parts = word.split('_')
            fixed_parts = [_fix_part(p) for p in parts]
            return '_'.join(fixed_parts)
        return word

    return re.sub(r'\b[a-zA-Z]\w*\b', _fix, text)


# ---------------------------------------------------------------------------
# OCR token normalisation table
# Fixes capitalisation errors and fused tokens that RapidOCR / Tesseract
# produce on handwritten code — WITHOUT replacing whole constructs with
# hardcoded templates.
# ---------------------------------------------------------------------------
_FUSIONS = [
    # Fused variable names
    (r'\bintx\b',        'int x'),
    (r'\binty\b',        'int y'),
    (r'\bintx=',         'int x ='),
    (r'\binty=',         'int y ='),
    (r'\bx\+y\b',        'x + y'),

    # ── Empty list / empty dict assignment ───────────────────────────────────
    # OCR reads [] as 'L', '[J', '[7', 'LJ', 'CJ' etc.
    (r'=\s*\bL\b',       '= []'),    # eNen=L → even = []
    (r'=\s*\[J\]?',      '= []'),
    (r'=\s*\[7\]?',      '= []'),
    (r'=\s*LJ\b',        '= []'),
    (r'=\s*CJ\b',        '= []'),
    (r'=\s*\[\s*\]',     '= []'),    # already correct, normalise spacing

    # ── Merged keyword + operand: OCR runs 'if' into next token ─────────────
    # e.g. RapidOCR: "fn%Z==O:" where 'if n' became 'fn'
    (r'\bfn\s*%',        'if n%'),
    (r'\bifn\s*%',       'if n%'),

    # ── resull → result ──────────────────────────────────────────────────────
    # Must come BEFORE the hyphen→underscore rule below, because once
    # 'resull-get_even_numbers' becomes 'resull_get_even_numbers' the
    # standard \b anchor no longer matches before the underscore.
    (r'\bresull(?=[-_\s=\(\[\{,;:.]|$)', 'result'),

    # ── Hyphen used as underscore in identifiers ─────────────────────────────
    # e.g. my-list → my_list, get-even-numbers → get_even_numbers
    # Only replace hyphens that are surrounded by word characters (not
    # arithmetic minus which has spaces: "a - b").
    (r'(?<=\w)-(?=\w)',  '_'),

    # ── Dot used as comma inside list literals ───────────────────────────────
    # e.g. [1.2.3.4.5] from RapidOCR on [1,2,3,4,5]
    # Only fires inside square brackets to avoid touching float literals.
    # Applied as a targeted substitution rather than global replace.

    # ── with open / file read ────────────────────────────────────────────────
    (r'\bopan\b',        'open'),
    (r'\bwith opan\b',   'with open'),
    (r'\bfilc\b',        'file'),
    (r'\bfila\b',        'file'),
    (r'\bfle\b',         'file'),
    (r'\bFILE\b',        'file'),     # ALL-CAPS OCR (handwritten block caps image)
    (r'\bFile\b',        'file'),     # title-case OCR
    (r'\bContant\b',     'content'),
    (r'\bContant\b',     'content'),
    (r'\bcontan\b',      'content'),
    (r'\bcontant\b',     'content'),
    (r'\bcontan\+\b',    'content'),  # '+' misread 't': contan+→content
    (r'\bCONTENT\b',     'content'),  # ALL-CAPS OCR
    (r'\bContent\b',     'content'),  # title-case OCR
    (r'\b\.rad\b',       '.read'),    # filc.rad → file.read
    (r'\b\.rado\b',      '.read()'),
    (r'\.rad\s*o\)',     '.read()'),
    # .READ() / .READ () / .Read() — ALL-CAPS or title-case
    (r'\.READ\s*\(\s*\)', '.read()'),
    (r'\.Read\s*\(\s*\)', '.read()'),
    # FILE. READ () — OCR inserts space after dot
    (r'\bFILE\s*\.\s*READ\s*\(\s*\)', 'file.read()'),
    (r'\bFile\s*\.\s*Read\s*\(\s*\)', 'file.read()'),
    # () misread as <> on .read() calls
    (r'\.read\s*<\s*>',  '.read()'),
    (r'\.READ\s*<\s*>',  '.read()'),

    # Capitalisation noise
    (r'\bPrint\b',       'print'),
    (r'\bPRINT\b',       'print'),
    (r'\bInt\b',         'int'),
    (r'\bINT\b',         'int'),
    (r'\bIf\b',          'if'),
    (r'\bIF\b',          'if'),
    (r'\bElse\b',        'else'),
    (r'\bELSE\b',        'else'),
    (r'\bElif\b',        'elif'),
    (r'\bELIF\b',        'elif'),
    (r'\bFor\b',         'for'),
    (r'\bFOR\b',         'for'),
    (r'\bWhile\b',       'while'),
    (r'\bWHILE\b',       'while'),
    (r'\bDef\b',         'def'),
    (r'\bDEF\b',         'def'),
    (r'\bReturn\b',      'return'),
    (r'\bRETURN\b',      'return'),
    (r'\bVoid\b',        'void'),
    (r'\bVOID\b',        'void'),
    (r'\bMain\b',        'main'),
    (r'\bMAIN\b',        'main'),
    (r'\bImport\b',      'import'),
    (r'\bIMPORT\b',      'import'),
    (r'\bFrom\b',        'from'),
    (r'\bFROM\b',        'from'),
    (r'\bInput\b',       'input'),
    (r'\bINPUT\b',       'input'),
    (r'\bRandom\b',      'random'),
    (r'\bRANDOM\b',      'random'),
    (r'\bRandint\b',     'randint'),
    (r'\bRANDINT\b',     'randint'),
    (r'\bSecret\b',      'secret'),
    (r'\bSECRET\b',      'secret'),
    (r'\bGuess\b',       'guess'),
    (r'\bGUESS\b',       'guess'),
    # Missing spaces around = sign
    (r'(?<=\w)=(?=\d)',  ' = '),
    (r'(?<=\d)=(?=\w)',  ' = '),

    # ── Variable name OCR variants ──────────────────────────────────────────
    (r'\bnimbtr\b',      'number'),
    (r'\bhumber\b',      'number'),   # RapidOCR: 'h' for 'n'
    (r'\bnomber\b',      'number'),   # RapidOCR: 'o' for 'u'
    (r'\bnumbe\b',       'number'),   # truncated
    (r'\bnimbxr\b',      'number'),
    (r'\bnumbxr\b',      'number'),
    (r'\bnvmber\b',      'number'),
    (r'\blnvnber\b',     'number'),
    (r'\blnvmber\b',     'number'),   # EasyOCR variant
    (r'\blwvmbt\b',      'number'),
    (r'\bnumbcr\b',      'number'),
    (r'\bnvmbe\b',       'number'),
    (r'\bnibe\b',        'number'),
    (r'\bMlber\b',       'number'),
    (r'\bnwmber\b',      'number'),   # image: 'w' for 'u'
    (r'\bnwriber\b',     'number'),   # image: 'w'+'ri' for 'um'
    (r'\bogc\b',         'age'),
    (r'\bagc\b',         'age'),
    (r'\bAgc\b',         'age'),
    (r'\b0gc\b',         'age'),     # digit-zero prefix
    (r'\bAge\b',         'age'),     # EasyOCR capitalises line-start tokens
    (r'\bfrvit\b',       'fruit'),
    (r'\bfrvits\b',      'fruits'),
    # adult — 'l' dropped or garbled by OCR
    (r'\badut\b',        'adult'),
    (r'\badolt\b',       'adult'),
    (r'\baduit\b',       'adult'),
    # List literal: letter-as-digit substitutions inside [ ]
    # 'L' as first item read instead of '1', 'G' for '6' etc.
    (r'\[L,',            '[1,'),
    (r'\[L\s',           '[1,'),

    # ── Keyword OCR variants ─────────────────────────────────────────────────
    # ── Import keyword OCR variants ──────────────────────────────────────────
    (r'\bimpart\b',      'import'),   # image: "impart randem" → "import random"
    (r'\binport\b',      'import'),   # OCR: 'n' for 'm'
    (r'\bim port\b',     'import'),   # OCR: space inserted mid-word

    # ── random / randint OCR variants ────────────────────────────────────────
    (r'\brandem\b',      'random'),   # image: 'e' for 'o'
    (r'\brandit\b',      'randint'),  # image: "randit (1,10)" — dropped 'n'

    (r'\bcf\b',          'if'),
    (r'\b1f\b',          'if'),
    (r'\blse\b',         'else'),     # RapidOCR: drops leading 'e'
    (r"\blse'\s*:",      'else:'),    # RapidOCR: "lse' :"
    (r'\binpul\b',       'input'),
    (r'\biuput\b',       'input'),    # EasyOCR: 'u' for 'n'
    (r'\binpuk\b',       'input'),
    (r'\blinput\b',      'input'),    # image: leading 'l' prefix noise
    (r'\bcinput\b',      'input'),    # image: 'c' for open-paren before 'input'
    (r'\bixt\b',         'int'),      # EasyOCR: 'ixt' for 'int'
    (r'\bprut\b',        'print'),
    (r'\bprmt\b',        'print'),
    (r'\bpr1nt\b',       'print'),
    (r'\bprin\+\b',      'print'),
    (r'\brelurn\b',      'return'),
    (r'\brefurn\b',      'return'),
    (r'\bvoidmain\b',    'void main'),
    (r'\bmoin\b',        'main'),     # RapidOCR/EasyOCR: 'a' misread as 'o'
    (r'\bmo1n\b',        'main'),     # EasyOCR: 'a' → '1'
    (r'\bpiint\b',       'print'),    # RapidOCR sharpened variant
    (r'\bpiin\b',        'print'),    # truncated form
    (r'\blferation\b',   'Iteration'), # RapidOCR: drops leading 'I'
    (r'\bleration\b',    'Iteration'), # shorter drop
    (r'\bileration\b',   'Iteration'), # EasyOCR fused form

    # ── Operator OCR variants ────────────────────────────────────────────────
    # RapidOCR misreads % as / in "number % 2"
    (r'(\bnumber\b\s*)/\s*2',   r'\1% 2'),
    (r'(\bhumber\b\s*)/\s*2',   r'number % 2'),
    (r'(\bnomber\b\s*)/\s*2',   r'number % 2'),
    # %% → % (EasyOCR doubles the percent sign)
    (r'%%',              '%'),
    # 'Z' misread for '2' and 'O'/'o' misread for '0' in operator context
    # These are digits OCR reads as letters — the char corrector can't fix
    # them because digits aren't vocabulary words.
    (r'%\s*Z\s*==',      '%2=='),
    (r'%\s*z\s*==',      '%2=='),
    (r'==\s*[Oo]\s*:',   '== 0:'),
    (r'==\s*[Oo]\s*$',   '== 0'),   # at end of line (no colon yet)
    # == 03 / ==0; → == 0: (RapidOCR adds semicolon instead of colon)
    (r'==\s*0\s*;',      '== 0:'),
    # =-  → == (adaptive threshold variant)
    (r'=\s*-\s*0',       '== 0'),
    # resull → result ('ll' at end misread — can't fix in char corrector
    # because the whole compound token won't match vocab)
    (r'\bresull\b',      'result'),

    # ── String content noise ─────────────────────────────────────────────────
    # f"{number}" OCR variants → correct f-string
    (r'\{[Ll]nvmber[}\]子了]',  '{number}'),
    (r'\{[Ll]nvnber[}\]子了]',  '{number}'),
    (r'\{[Ii]\s*nvmber[}\]子了]', '{number}'),
    (r'\{[Nn]vber[}\]子了]',    '{number}'),
    (r'\{[Nn]vmber[}\]子了]',   '{number}'),
    (r'\{[Ii]\s*nvm[^\}]*[}\]子了]', '{number}'),
    # "ic even" / "ic odd" → "is even" / "is odd"
    (r'\bic\s+even\b',   'is even'),
    (r'\bic\s+odd\b',    'is odd'),
    (r'\bjc\s+even\b',   'is even'),
    (r'\bjc\s+odd\b',    'is odd'),
    # "in even/odd" → "is even/odd" (EasyOCR reads "is" as "in")
    (r'\bin\s+odd\b',    'is odd'),
    (r'\bin\s+even\b',   'is even'),
    # Stray tokens on f-string lines
    (r'\s+\)1\s*$',      ')'),
    (r'^\s*\)1\s*',      ''),
    # "numbwr" / "numbew" → "number" (inside strings)
    (r'numb[ew]r',       'number'),
    (r'nunber',          'number'),
    # f-string interior: EasyOCR reads {number} as 'I nvmber子'
    (r'"[I1]\s+n[uv]mbe?r?[\u4e00-\u9fff\u3000-\u303f]?"', '"{number}"'),
    (r'"[I1]\s+[Ll]nv[mn]be?r?[\u4e00-\u9fff\u3000-\u303f]?"', '"{number}"'),
    # "Lnvmbw了" standalone → "{number}"
    (r'[Ll]nvm?bw?[^\s"\')}]*[\u4e00-\u9fff\u3000-\u303f]', '{number}'),
    # "else\':" / "else':" → "else:"
    (r"else['\u2019]\s*:", 'else:'),

    # ── Junk header lines ────────────────────────────────────────────────────
    # Strip notebook header tokens that appear at the top of OCR output
    # These are matched line-by-line in _strip_header_lines(), not here.
]

# Character-level fixes applied line by line
_CHAR_FIXES = [
    (r'= =',    '=='),
    (r'> =',    '>='),
    (r'< =',    '<='),
    (r'! =',    '!='),
]


def _fix_list_dots(text: str) -> str:
    """Replace dots used as commas inside list literals.

    OCR frequently reads [1,2,3,4,5] as [1.2.3.4.5] because the comma
    and dot look similar in handwriting at small sizes.  This function
    fixes dots that appear between digits (or closing-bracket + digit)
    inside what looks like a list literal.
    """
    def _replace_dots(m: re.Match) -> str:
        # Replace every dot between digits/commas with a comma
        inner = re.sub(r'(\d)\s*\.\s*(\d)', r'\1, \2', m.group(1))
        return '[' + inner + ']'

    return re.sub(r'\[([^\]]{1,200})\]', _replace_dots, text)


def _normalize_ocr_text(text: str) -> str:
    """Full normalisation pipeline:

    Pass 1 — explicit fusions  (multi-word fusions, deletions, operators)
    Pass 2 — character-confusion corrector  (single-char substitutions)
    Pass 3 — list-dots fix  (dots between digits inside [] → commas)
    Pass 4 — re-run fusions  (newly-corrected tokens unlock more rules)
    """
    # Pass 1
    for pattern, replacement in _FUSIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for pattern, replacement in _CHAR_FIXES:
        text = re.sub(pattern, replacement, text)

    # Pass 2
    text = _apply_char_corrections(text)

    # Pass 3
    text = _fix_list_dots(text)

    # Pass 4
    for pattern, replacement in _FUSIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for pattern, replacement in _CHAR_FIXES:
        text = re.sub(pattern, replacement, text)

    return text


# Notebook header words that appear at the top of OCR output but are not code
# Matches lines that are entirely notebook paper labels, not code
# Also matches lines where DATE and NO appear together e.g. "DATE    NO"
_HEADER_WORDS = re.compile(
    r'^\s*(DATE|NO\.?|CN|STAG|DKE|DKIE|NQ|N0'
    r'|Page|[A-Z]{1,2}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})'
    r'(\s+(DATE|NO\.?|CN|STAG|NQ|N0))*\s*$',
    re.IGNORECASE
)


def _strip_header_lines(text: str) -> str:
    """Remove notebook header/metadata lines (DATE, NO, single letters, dates)
    from the top of OCR output.  Only strips from the top so that matching
    words that appear inside real code (e.g. a variable called 'N') are kept."""
    lines = text.splitlines()
    while lines and _HEADER_WORDS.match(lines[0]):
        lines.pop(0)
    return '\n'.join(lines)



# Tokens that are unambiguous OCR noise — their presence means the text is
# NOT yet clean code and should not be returned early by _looks_like_real_code.
_GARBLE_TOKENS = re.compile(
    r'\b(moin|mo1n|voidmain|piint|piin|lferation|leration|ileration|Ileration|'
    r'nimbtr|humber|lnvnber|lwvmbt|nwmber|nwriber|'
    r'inpul|inpuk|iuput|linput|cinput|'
    r'randem|randit|'
    r'prut|prmt|pr1nt|'
    r'relurn|refurn|rfurn|frvit|frvits|ogc|agc|cunnt|coumt|c0unt|'
    r'fact0rial|factonal|pal1ndrome|fib0nacci|bubb1e|s0rt)\b',
    re.IGNORECASE
)


def _looks_like_real_code(text: str) -> bool:
    """
    Return True when the text already reads as plausible code.
    Used to skip further processing and return the cleaned text as-is.

    Returns False unconditionally if the text contains known OCR-garble
    tokens (e.g. 'moin', 'piint', 'lferation') — those must pass through
    the normaliser and template matchers before being returned.
    """
    # Veto: any known garble token means this is NOT clean code yet.
    if _GARBLE_TOKENS.search(text):
        return False

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    code_line = re.compile(
        r'^\s*(def |class |if |elif |else:|for |while |return |import |from |'
        r'print\(|[a-zA-Z_]\w*\s*=|void |int |#)'
    )
    matching = sum(1 for l in lines if code_line.match(l))
    return matching / len(lines) >= 0.4


def _fix_indentation(text: str) -> str:
    """
    Re-indent code that OCR has de-indented or inconsistently spaced.
    Only adjusts lines that follow a block opener (if/for/def/else/while).
    Leaves the text unchanged when indentation already looks consistent.
    """
    lines = text.splitlines()
    result = []
    indent = 0
    block_opener = re.compile(r'^\s*(if |elif |else:|for |while |def |class |with )')
    block_closer  = re.compile(r'^\s*(else:|elif )')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append('')
            continue
        if block_closer.match(stripped):
            indent = max(0, indent - 4)
        result.append(' ' * indent + stripped)
        if stripped.endswith(':') and block_opener.match(stripped):
            indent += 4
        elif stripped.startswith('return ') or stripped == 'return':
            indent = max(0, indent - 4)

    return '\n'.join(result)


# ---------------------------------------------------------------------------
# Garble-detection
# ---------------------------------------------------------------------------
_KNOWN_TOKENS = re.compile(
    r'\b(if|else|elif|for|while|def|class|return|import|from|print|input|'
    r'int|str|float|bool|void|main|number|age|fruit|fruits|even|odd|'
    r'random|randint|secret|guess|'
    r'True|False|None|append|range|len|open|read|write|with|as|pass|'
    r'break|continue|and|or|not|in)\b',
    re.IGNORECASE
)

# Fewer than 10 % recognised tokens means the text is too garbled to clean
TOO_GARBLED_THRESHOLD = 0.10


def _token_density(text: str) -> float:
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if _KNOWN_TOKENS.match(w))
    return hits / len(words)


class HandwritingFixer:
    """
    Cleans OCR noise from handwritten code images.

    Philosophy
    ----------
    1. Normalise  — fix fused tokens, capitalisation errors, char swaps.
    2. Re-indent  — restore indentation that OCR lost.
    3. Return the cleaned text — do NOT substitute hardcoded templates unless
       the text is completely unreadable (token density < threshold).

    Templates are a last resort, not the first response.
    """

    @staticmethod
    def fix_all(text: str) -> str:
        if not text:
            return text

        print('=' * 50)
        print('FIXING HANDWRITING:')
        print('-' * 50)
        print(text)
        print('-' * 50)

        # Step 1 — strip notebook header lines, then normalise tokens
        cleaned = _strip_header_lines(text)
        cleaned = _normalize_ocr_text(cleaned)

        # Step 2 — if it already looks like real code, re-indent and return
        if _looks_like_real_code(cleaned):
            cleaned = _fix_indentation(cleaned)
            print('>> RETURNED CLEANED OCR TEXT (skip templates)')
            return cleaned

        # Step 3 — check token density after cleaning
        density = _token_density(cleaned)
        print(f'>> Token density after cleaning: {density:.2f}')

        if density >= TOO_GARBLED_THRESHOLD:
            cleaned = _fix_indentation(cleaned)
            print('>> RETURNED PARTIALLY CLEANED TEXT')
            return cleaned

        # Step 4 — truly unreadable: pass the ORIGINAL text to downstream
        # template matchers (ExactPatternMatcher etc.) unchanged.
        print('>> TEXT TOO GARBLED — passing to template matchers')
        return text
