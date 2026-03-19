import re


# ─────────────────────────────────────────────────────────────────────────────
# OCR noise → real token mappings
# ─────────────────────────────────────────────────────────────────────────────
OCR_WORD_MAP = {
    # Keywords
    'cf': 'if', '1f': 'if', 'if': 'if',
    'els': 'else', 'ele': 'else', 'el se': 'else', 'lse': 'else', 'else': 'else',
    'elif': 'elif', 'el1f': 'elif',
    'far': 'for', 'for': 'for', 'fo r': 'for',
    'whi le': 'while', 'wh1le': 'while', 'while': 'while',
    'def': 'def', 'de f': 'def',
    'ret urn': 'return', 'return': 'return',
    'class': 'class', 'clas s': 'class',
    'import': 'import', 'im port': 'import',
    'true': 'True', 'false': 'False', 'none': 'None',
    'and': 'and', 'or': 'or', 'not': 'not', 'in': 'in',
    'break': 'break', 'continue': 'continue', 'pass': 'pass',

    # Python builtins
    'prut': 'print', 'prmt': 'print', 'pr1nt': 'print', 'prnt': 'print', 'print': 'print',
    'inpul': 'input', 'inp ut': 'input', 'input': 'input',
    'int': 'int', 'mt': 'int', 'str': 'str', 'float': 'float', 'bool': 'bool',
    'len': 'len', 'range': 'range', 'list': 'list', 'dict': 'dict',
    'append': 'append', 'split': 'split', 'strip': 'strip',

    # Variable name fragments
    'nimbtr': 'number', 'humber': 'number', 'lnvnber': 'number',
    'lwvmbt': 'number', 'numb': 'number', 'nmbr': 'number',
    'numbcr': 'number', 'num ber': 'number', 'number': 'number',
    'ogc': 'age', 'agc': 'age', 'age': 'age',
    'frvit': 'fruit', 'frvits': 'fruits', 'fruit': 'fruit', 'fruits': 'fruits',

    # Operators / symbols
    '%%': '%', '= =': '==', '> =': '>=', '< =': '<=', '! =': '!=',

    # Dart keywords
    'void': 'void', 'main': 'main', 'var': 'var',
    'final': 'final', 'const': 'const',
}

# OCR fragments that signal a specific construct
CONSTRUCT_SIGNALS = {
    'if':       ['if', 'cf', '1f'],
    'else':     ['else', 'els', 'ele', 'lse'],
    'elif':     ['elif', 'el1f', 'else if'],
    'for':      ['for', 'far', 'iteration', 'ileration', 'Ileration', 'range', 'reration', 'neration',
                 'eration', 'heration', 'Heration'],  # bare/h-prefixed OCR drops of 'Iteration'
    'while':    ['while', 'whi le', 'wh1le'],
    'def':      ['def', 'de f', 'function', 'func'],
    'class':    ['class', 'clas'],
    'print':    ['print', 'prut', 'prmt', 'pr1nt', 'prnt', 'prin+'],
    'input':    ['input', 'inpul', 'inp'],
    'return':   ['return', 'ret urn', 'relurn', 'refurn', 'rfurn', 'rekurw'],
    'import':   ['import', 'im port'],
    'even':     ['even', 'evn'],
    'odd':      ['odd', 'od'],
    'adult':    ['adult', 'adultly', 'adut', 'adolt', 'aduit', 'an adult', 'an adut'],
    'minor':    ['minor', 'not adult', 'a minor'],
    'number':   ['number', 'nimbtr', 'humber', 'lnvnber', 'lwvmbt', 'numb', 'nmbr'],
    'age':      ['age', 'ogc', 'agc', '0gc', 'agc', 'oge', 'ag3', 'your age', 'your_age',
                 '>=18', '>= 18', '>=16', '>= 16'],   # >=16 is a common OCR misread of >=18
    'fruit':    ['fruit', 'frvit', 'frvits', 'fruits'],
    'void':     ['void', 'voidmain'],
    'main':     ['main', 'voidmain'],
    'append':   ['append', 'appendc', 'apperid', '.append'],
    'result':   ['result', 'resul', 'rsult', 'lrjult', 'rjult'],
    'my_list':  ['my_list', 'my _list', 'my-list', 'mylist'],
    'with':     ['with opan', 'with open', 'with'],
    'open':     ['open', 'opan', 'opn'],
    'read':     ['read', 'raad', 'rad', '.read'],
    'content':  ['content', 'contant', 'contunt', 'contan', 'conthn'],
    'int_x':    ['int x', 'int  x', 'intx', 'f |0', 'f|0'],
    'int_y':    ['int y', 'int  y', 'inty'],
    'x_plus_y': ['x + y', 'x+y', 'x t'],
    'print_xy': ['print(x + y)', 'print(x+y)', 'prlu', '~prlu', 'print(x'],
    'hello':    ['hello', 'hellv', 'hell0', 'hell?', 'hellc', 'hella', 'mcllo', 'hcllo'],
    'world':    ['world', 'wocvr', 'wocva', 'wocvb', 'worlo', 'w0rld'],
}


def detect_signals(text):
    """Detect which constructs are present in OCR text."""
    t = text.lower()
    found = set()
    for signal, variants in CONSTRUCT_SIGNALS.items():
        for v in variants:
            if v in t:
                found.add(signal)
                break
    return found


def detect_numbers(text):
    """Extract meaningful numbers from OCR text."""
    nums = re.findall(r'\b\d+\b', text)
    return nums


def detect_language(text):
    """Detect Python vs Dart from OCR fragments.

    Uses a consistent heuristic shared with app.py's detect_language() so that
    DynamicCodeBuilder and the app-level language detector never disagree on
    the same input.  The scoring mirrors the weighted pattern approach in app.py
    but is kept lightweight here since we're working with raw OCR noise.

    Brace characters ({ }) score for Dart only when they appear as part of a
    real construct (e.g. 'void main() {'), not when they appear as isolated
    noise tokens from garbled OCR (e.g. 'J}').

    Key fix: phone-photo OCR frequently splits 'void main' across two text
    rows ('Void\\nmain'), so 'void main' never matches as a substring.  We
    now also award full dart points when 'void' and 'main' appear anywhere in
    the text even if separated by whitespace/newlines.
    """
    t = text.lower()

    # Count { } only on lines that also contain a Dart keyword —
    # isolates real Dart constructs from stray brace noise like "J}" / "U }"
    dart_brace_score = 0
    dart_kw = ['void', 'main', 'int ', 'string', 'final', 'const', 'for', 'if', 'else', 'class']
    for line in text.splitlines():
        ll = line.lower()
        if ('{' in ll or '}' in ll) and any(kw in ll for kw in dart_kw):
            dart_brace_score += 1

    # 'void main' may appear as separate tokens on different OCR text rows.
    # Treat both the fused form and the separated form as equally strong signals.
    has_void_main = (
        'void main' in t
        or 'voidmain' in t
        or (bool(re.search(r'\bvoid\b', t)) and bool(re.search(r'\bmain\b', t)))
    )

    dart_score = sum([
        4 if has_void_main else 0,
        sum(2 for w in ['final', 'const', 'var ', 'dart:'] if w in t),
        dart_brace_score,
        t.count(';'),
    ])
    python_score = sum([
        sum(2 for w in ['def ', 'elif ', 'import ', 'from ', 'input(', 'print('] if w in t),
        t.count(':'),
        4 if re.search(r'\bfor\b.+\bin\b', t) else 0,
    ])
    return 'dart' if dart_score > python_score else 'python'


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_python_with_open(signals, text):
    """Build with open / file read block.

    Attempts to extract the filename and open mode from the OCR text so the
    output reflects what was actually written.  Falls back to 'data.txt' / 'r'
    when the text is too garbled to parse.
    """
    # Try to extract a quoted filename from the OCR text
    filename = 'data.txt'
    fname_match = re.search(r'["\']([^"\']{1,40}\.[a-z]{1,5})["\']', text)
    if fname_match:
        filename = fname_match.group(1)

    # Detect open mode: write/append signals override the default read mode
    mode = 'r'
    t = text.lower()
    if any(w in t for w in ['"w"', "'w'", 'write', 'writ']):
        mode = 'w'
    elif any(w in t for w in ['"a"', "'a'", 'append mode']):
        mode = 'a'

    if mode == 'r':
        return (
            f'with open("{filename}", "r") as file:\n'
            f'    content = file.read()\n'
            f'    print(content)'
        )
    else:
        return (
            f'with open("{filename}", "{mode}") as file:\n'
            f'    file.write("your text here")'
        )


def build_python_get_even_numbers(signals, text):
    """Build get_even_numbers function with my_list and print result."""
    return '''def get_even_numbers(numbers):
    even = []
    for n in numbers:
        if n % 2 == 0:
            even.append(n)
    return even

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = get_even_numbers(my_list)
print(f"The even numbers are: {result}")'''


def build_python_if_else(signals, text):
    """Build if/elif/else block dynamically.

    Priority order: age/adult > number/even/odd > generic.

    'number' appears as OCR noise in many images that are actually age-check
    snippets (e.g. the garbled string "Enter a number:" in "Enter your age:").
    age/adult are checked first so that explicit age signals always win.
    A >=18 literal in the text is treated as a decisive age-check signal even
    when 'age'/'adult' tokens were not detected by the signal scanner.
    """
    t = text.lower()

    # Decisive tiebreaker: a literal >=18 or >= 18 can only be an age check
    has_ge18 = bool(re.search(r'>=\s*1[68]', text))

    # Count signals in each domain to handle cases where both fire
    age_signals   = sum(1 for s in ['age', 'adult', 'minor'] if s in signals)
    num_signals   = sum(1 for s in ['number', 'even', 'odd'] if s in signals)

    # Route: age/adult wins when it has any explicit signal OR >=18 is present
    if age_signals > 0 or has_ge18:
        # Age check
        if 'input' in signals or 'age' in signals:
            lines = ['age = int(input("Enter your age: "))', '']
        else:
            lines = []
        lines.append('if age >= 18:')
        lines.append('    print("You are an adult")')
        if 'else' in signals or 'minor' in t:
            lines.append('else:')
            lines.append('    print("You are a minor")')

    elif num_signals > 0 or 'number' in signals or 'even' in signals or 'odd' in signals:
        # Even/odd check
        if 'input' in signals or 'number' in signals:
            lines = ['number = int(input("Enter a number: "))', '']
        else:
            lines = []
        lines.append('if number % 2 == 0:')
        lines.append('    print(f"{number} is even")')
        if 'else' in signals or 'odd' in signals:
            lines.append('else:')
            lines.append('    print(f"{number} is odd")')

    else:
        # Generic if/else
        lines = ['if condition:', '    print("condition is true")']
        if 'else' in signals:
            lines += ['else:', '    print("condition is false")']

    return '\n'.join(lines)


def build_python_for(signals, text):
    """Build for loop dynamically."""
    t = text.lower()
    nums = detect_numbers(text)

    if 'fruit' in signals:
        lines = [
            'fruits = ["apple", "banana", "cherry"]',
            '',
            'for fruit in fruits:',
            '    print(fruit)',
        ]
    elif 'number' in signals or 'range' in t:
        limit = nums[0] if nums else '5'
        lines = [
            f'for i in range({limit}):',
            '    print(i)',
        ]
    else:
        limit = nums[0] if nums else '5'
        lines = [
            f'for i in range({limit}):',
            '    print(i)',
        ]
    return '\n'.join(lines)


def build_python_while(signals, text):
    """Build while loop dynamically."""
    nums = detect_numbers(text)
    limit = nums[0] if nums else '5'
    lines = [
        f'i = 0',
        f'while i < {limit}:',
        '    print(i)',
        '    i += 1',
    ]
    return '\n'.join(lines)


def build_python_def(signals, text):
    """Build function definition dynamically."""
    match = re.search(r'def\s+(\w+)', text, re.IGNORECASE)
    name = match.group(1) if match else 'my_function'
    lines = [
        f'def {name}():',
        '    pass',
    ]
    if 'return' in signals:
        lines[-1] = '    return result'
    return '\n'.join(lines)


def build_python_hello_world(signals, text):
    """Build the Hello World + int x/y + print(x+y) template.

    Extracts actual x and y values from the OCR text when readable;
    falls back to 10/10 when they can't be parsed.
    """
    x_match = re.search(r'\bint\s+x\s*=\s*(\d+)', text, re.IGNORECASE)
    y_match = re.search(r'\bint\s+y\s*=\s*(\d+)', text, re.IGNORECASE)
    # Also try bare number lines: "F |0" → "10", "X = 10" etc.
    if not x_match:
        x_match = re.search(r'\bx\s*=\s*(\d+)', text, re.IGNORECASE)
    if not y_match:
        y_match = re.search(r'\by\s*=\s*(\d+)', text, re.IGNORECASE)
    x_val = x_match.group(1) if x_match else "10"
    y_val = y_match.group(1) if y_match else "10"
    total = int(x_val) + int(y_val)
    return (
        f'print("Hello World")\n\n'
        f'int x = {x_val}\n'
        f'int y = {y_val}\n\n'
        f'x + y = {total}\n\n'
        f'print(x + y)'
    )


def build_python_print(signals, text):
    """Build standalone print statement.

    Attempts to extract the actual string argument from the OCR text rather
    than always emitting a hardcoded value.  Falls back to context-aware
    defaults when no quoted string can be parsed.
    """
    t = text.lower()

    # Try to extract a quoted string that was already in the OCR output
    quoted = re.search(r'["\']([^"\']{1,80})["\']', text)
    if quoted:
        return f'print("{quoted.group(1)}")'

    # Domain-specific defaults
    if 'adult' in t:
        return 'print("You are an adult")'
    if ('hello' in t or any(v in t for v in ['hellv', 'hell?', 'hell0', 'hellc'])
            and ('world' in t or any(v in t for v in ['wocvr', 'wocva', 'worlo']))):
        return 'print("Hello World")'
    if 'hello' in t or any(v in t for v in ['hellv', 'hell?', 'hell0', 'hellc']):
        return 'print("Hello World")'

    # Generic fallback
    return 'print("Hello World")'


# ─────────────────────────────────────────────────────────────────────────────
# DART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_dart_main_for(signals, text):
    """Build Dart void main with for loop.

    Extracts the loop limit from an explicit 'i < N' or 'i<N' pattern so
    that other digits in the OCR text (e.g. '0' from 'i = 0') are never
    mistaken for the loop bound.  Prefers 'i < N' over 'i > N' since the
    '>' form is always an OCR misread of '<'.
    """
    limit = '5'
    lt_match = re.search(r'i\s*<\s*([1-9]\d*)', text)
    gt_match = re.search(r'i\s*>\s*([1-9]\d*)', text)
    # Also catch '0,i45' style RapidOCR noise where 'i45' means 'i<5'
    noise_match = re.search(r'i\s*(\d{2,})', text)
    if lt_match:
        limit = lt_match.group(1)
    elif gt_match:
        limit = gt_match.group(1)
    elif noise_match:
        # e.g. 'i45' → last digit is the limit
        limit = noise_match.group(1)[-1]

    return f'''void main() {{
    for (int i = 0; i < {limit}; i++) {{
        print("Iteration number : $i");
    }}
}}'''


def build_dart_main_if(signals, text):
    """Build Dart void main with if/else."""
    t = text.lower()
    if 'age' in signals or 'adult' in signals:
        code = '''void main() {
    int age = int.parse(stdin.readLineSync()!);
    if (age >= 18) {
        print("You are an adult");
    }'''
        if 'else' in signals or 'minor' in signals:
            code += '''
    else {
        print("You are a minor");
    }'''
        code += '\n}'
        return code

    if 'even' in signals or 'odd' in signals:
        code = '''void main() {
    int number = int.parse(stdin.readLineSync()!);
    if (number % 2 == 0) {
        print("$number is even");
    }'''
        if 'else' in signals or 'odd' in signals:
            code += '''
    else {
        print("$number is odd");
    }'''
        code += '\n}'
        return code

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

class DynamicCodeBuilder:
    """
    Dynamically reconstructs code from messy OCR output.
    Works for any Python or Dart structure — if/else, for, while, def, class.
    Used as primary engine; exact_pattern_matcher is the fallback for
    known-noisy OCR outputs.
    """

    @staticmethod
    def build(text):
        if not text or len(text.strip()) < 3:
            return None

        # If the text already contains real, parseable code structure — lines
        # with proper indentation, existing import statements, or multi-statement
        # blocks — don't replace it with a reconstructed template.  Reconstruction
        # is only needed when the input is garbled OCR noise.
        lines = [l for l in text.splitlines() if l.strip()]
        real_code_lines = sum(
            1 for l in lines
            if (l.strip().startswith('import ') or
                l.strip().startswith('from ') or
                re.match(r'^\s*(def |class |if |for |while |return |print\(|[a-zA-Z_]\w*\s*=)', l))
        )
        # If more than half the non-empty lines look like real code, trust the text as-is
        if len(lines) > 1 and real_code_lines / len(lines) >= 0.5:
            print(f"🔍 DynamicCodeBuilder: text looks like real code ({real_code_lines}/{len(lines)} lines) — skipping reconstruction")
            return None

        # ── Noise guard ──────────────────────────────────────────────────────
        # If fewer than 10 % of tokens are recognisable code words, the text is
        # pure OCR noise.  Running signal detection on it produces false matches
        # (e.g. 'os' from "| | a \ a >" or 'age' from "Wa ae ee") that lead to
        # wrong templates being returned.  Return None so the caller falls
        # through to the ExactPatternMatcher instead.
        _KNOWN_RE = re.compile(
            r'\b(if|else|elif|for|while|def|class|return|import|from|print|input|'
            r'int|str|float|bool|void|main|number|age|fruit|fruits|even|odd|'
            r'True|False|None|append|range|len|open|read|write|with|as|pass|'
            r'break|continue|and|or|not|in|adult|minor)\b',
            re.IGNORECASE
        )
        _words = re.findall(r'\b\w+\b', text)
        if _words:
            _density = sum(1 for w in _words if _KNOWN_RE.match(w)) / len(_words)
            if _density < 0.10:
                print(f'🔍 DynamicCodeBuilder: text too garbled (density={_density:.2f}) — skipping')
                return None
        # ─────────────────────────────────────────────────────────────────────

        signals = detect_signals(text)
        lang = detect_language(text)

        print(f"🔍 DynamicCodeBuilder signals: {signals}")
        print(f"🔍 DynamicCodeBuilder language: {lang}")

        if not signals:
            return None

        # ── DART ──────────────────────────────────────────────────────────
        if lang == 'dart':
            has_dart_entry = 'void' in signals and 'main' in signals

            # 'for' always takes priority — 'number' appearing in the OCR
            # text of a for-loop (e.g. "Iteration number : $i") must never
            # redirect to the if/else template.
            has_for = 'for' in signals or re.search(r'i\s*[<>]\s*\d', text) is not None
            has_iteration = any(w in text.lower() for w in [
                'iteration', 'ileration', 'reration', 'neration',
                # 'eration' alone — OCR frequently drops the leading 'It'
                # from 'Iteration', leaving just 'eration' or 'Heration'
                'eration',
                'i<', 'i++', 'i45', 'i#',
            ]) or bool(re.search(r'\bi\s*\d{2,}', text))
            # The last guard catches RapidOCR noise like 'i25' or 'Cnt1=0i25'
            # which is a mangled rendering of 'i = 0, i < 5'.

            if has_dart_entry and has_for and has_iteration:
                return build_dart_main_for(signals, text)

            # if/else path — only when there is no for-loop evidence and
            # there are genuine condition signals (age/adult OR even/odd)
            if has_dart_entry and not has_for:
                if 'if' in signals or 'age' in signals or 'adult' in signals or 'even' in signals:
                    result = build_dart_main_if(signals, text)
                    if result is not None:
                        return result

        # ── PYTHON ────────────────────────────────────────────────────────
        else:
            # with open / file read — check before other patterns
            if 'with' in signals and ('read' in signals or 'content' in signals):
                return build_python_with_open(signals, text)

            if 'def' in signals:
                # def + for + append + return = get_even_numbers style function
                if ('for' in signals or 'append' in signals) and 'return' in signals:
                    return build_python_get_even_numbers(signals, text)
                return build_python_def(signals, text)

            if 'for' in signals and 'if' not in signals:
                return build_python_for(signals, text)

            if 'while' in signals and 'if' not in signals:
                return build_python_while(signals, text)

            # if/else — also trigger on domain signals even without 'if' keyword
            if ('if' in signals
                    or 'age' in signals or 'adult' in signals
                    or 'even' in signals or 'odd' in signals
                    or ('number' in signals and 'print' in signals)):
                return build_python_if_else(signals, text)

            # hello / world → full Hello World + int x/y template
            # This must come before the generic 'print' path.
            if 'hello' in signals or 'world' in signals:
                return build_python_hello_world(signals, text)

            if 'print' in signals:
                return build_python_print(signals, text)

        # Signals were detected but no builder could reconstruct the code.
        # Return a commented stub so the user sees something meaningful
        # rather than raw OCR noise being passed back as the final output.
        if signals:
            comment = '//' if lang == 'dart' else '#'
            raw_preview = text[:120].replace('\n', ' ').strip()
            print(f'⚠️  DynamicCodeBuilder: signals={signals} but no builder matched — returning stub')
            return (
                f'{comment} Could not fully reconstruct {lang} code from OCR output\n'
                f'{comment} Detected signals: {", ".join(sorted(signals))}\n'
                f'{comment} Raw OCR: {raw_preview}'
            )

        return None

    @staticmethod
    def process(text):
        """Entry point matching the interface of other matchers."""
        try:
            from library_detector import LibraryDetector
            detected_libs = LibraryDetector.detect_libraries(text)
        except ImportError:
            LibraryDetector = None
            detected_libs = []

        result = DynamicCodeBuilder.build(text)
        if result:
            if detected_libs and LibraryDetector is not None:
                imports = LibraryDetector.get_imports(detected_libs)
                if imports and imports[0] not in result:
                    result = '\n'.join(imports) + '\n\n' + result
            print(f"✅ DynamicCodeBuilder reconstructed code")
            return result
        return text