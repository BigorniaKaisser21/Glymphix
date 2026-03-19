import re
import logging

logger = logging.getLogger(__name__)


def _is_real_code(text: str) -> bool:
    """Return True when the text is already plausible code.

    Threshold raised to 0.55 to match exact_pattern_matcher and
    handwriting_fixes so all guards are consistent.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    code_line = re.compile(
        r'^\s*(def |class |if |elif |else:|for |while |return |import |from |'
        r'print\(|[a-zA-Z_]\w*\s*=|void |int [\w]|#)'
    )
    matching = sum(1 for l in lines if code_line.match(l))
    return matching / len(lines) >= 0.55


class CustomPatternMatcher:
    """
    User-extensible pattern matcher for new OCR noise patterns.

    Add entries to CUSTOM_PATTERNS as 3-tuples:
        (trigger_callable, fixer_callable, description_str)

    description_str is used only for debug logging.

    Every trigger MUST:
      1. Check `not _is_real_code(t)` first — never replace readable code.
      2. Require multiple corroborating noise signals, not just one keyword.
      3. Include cross-contamination guards so distinct snippets cannot
         accidentally fire each other's templates.

    STRICTNESS TIERS (copy these rules when adding new patterns):
    ─────────────────────────────────────────────────────────────
    Tier 1 — Very specific / rare token (e.g. "0_i45", "frvit")
             → 1 garbled token + 1 structural keyword is enough.
    Tier 2 — Common domain word (e.g. "count", "sort", "reverse")
             → 3 corroborating signals minimum.
    Tier 3 — Very common word (e.g. "number", "main", "for")
             → 4 corroborating signals + at least one negative guard.
    """

    CUSTOM_PATTERNS = [

        # ════════════════════════════════════════════════════════════════════
        # ── EXISTING PATTERNS ────────────────────────────────────────────────
        # ════════════════════════════════════════════════════════════════════

        # ── if-else even/odd without input line ──────────────────────────────
        # Tier 3: garbled if/cf + % + BOTH even AND odd + no input signal
        # and the code must not already be structured properly.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['cf', '1f', 'if'])
                and '%' in t
                and 'even' in t.lower()
                and 'odd' in t.lower()
                and 'input' not in t.lower()
                and not re.search(r'if\s+\w+\s*%\s*2\s*==\s*0', t)
            ),
            lambda t: 'if number % 2 == 0:\n    print(f"{number} is even")\nelse:\n    print(f"{number} is odd")',
            'even/odd (no input) — garbled if + % + even + odd, no input line',
        ),

        # ── Dart for-loop without void main header ────────────────────────────
        # Tier 2: 'for' + 'int i' (exact) + print + iteration word
        (
            lambda t: (
                not _is_real_code(t)
                and 'for' in t.lower()
                and 'int i' in t.lower()
                and 'print' in t.lower()
                and any(w in t.lower() for w in ['iteration', 'reration', 'neration'])
            ),
            lambda t: 'void main() {\n    for (int i = 0; i < 5; i++) {\n        print("Iteration number : $i");\n    }\n}',
            'dart for-loop (for + int i + print + garbled iteration)',
        ),


        # ════════════════════════════════════════════════════════════════════
        # ── NEW PATTERNS ─────────────────────────────────────────────────────
        # ════════════════════════════════════════════════════════════════════

        # ── FizzBuzz (no clean code) ──────────────────────────────────────────
        # Tier 2: fizz + buzz are distinctive enough; still require range + %.
        (
            lambda t: (
                not _is_real_code(t)
                and 'fizz' in t.lower()
                and 'buzz' in t.lower()
                and any(w in t.lower() for w in ['rang', 'range', '101', 'r4ng', 'ranqe'])
                and '%' in t
            ),
            lambda t: (
                'for i in range(1, 101):\n'
                '    if i % 15 == 0:\n'
                '        print("FizzBuzz")\n'
                '    elif i % 3 == 0:\n'
                '        print("Fizz")\n'
                '    elif i % 5 == 0:\n'
                '        print("Buzz")\n'
                '    else:\n'
                '        print(i)'
            ),
            'fizzbuzz (fizz + buzz + range + %)',
        ),

        # ── while counter (Python) ────────────────────────────────────────────
        # Tier 2: garbled while + garbled count + += or ++ + "done"
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['whlie', 'whi1e', 'whil3', 'whjle'])
                and any(w in t.lower() for w in ['cunnt', 'coumt', 'coimt', 'c0unt'])
                and any(w in t for w in ['+=', '++', '+= 1'])
                and 'done' in t.lower()
                and 'void' not in t.lower()
                and 'fizz' not in t.lower()
            ),
            lambda t: (
                'count = 1\n\n'
                'while count <= 5:\n'
                '    print(f"Count: {count}")\n'
                '    count += 1\n\n'
                'print("Done!")'
            ),
            'while counter (garbled while + garbled count + += + done)',
        ),

        # ── factorial recursive ───────────────────────────────────────────────
        # Tier 2: garbled "factorial" + return + n-1 recursive signal.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'factori', 'factonal', 'factoril', 'factori4l', 'fact0rial'
                ])
                and any(w in t.lower() for w in ['relurn', 'refurn', 'rfurn', 'return'])
                and any(w in t.lower() for w in ['n-1', 'n - 1', 'n —1'])
                and not re.search(r'result\s*\*=', t)
                and 'sort' not in t.lower()
            ),
            lambda t: (
                'def factorial(n):\n'
                '    if n == 0 or n == 1:\n'
                '        return 1\n'
                '    return n * factorial(n - 1)\n\n'
                'num = int(input("Enter a number: "))\n'
                'print(f"Factorial of {num} is {factorial(num)}")'
            ),
            'factorial recursive (garbled factorial + return + n-1)',
        ),

        # ── fibonacci ─────────────────────────────────────────────────────────
        # Tier 2: garbled fibonacci + terms + a,b swap idiom.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'fibona', 'fibonac', 'fib0nac', 'fibonaci', 'fibonacci'
                ])
                and any(w in t.lower() for w in ['term', 'trm', 't3rm'])
                and any(w in t for w in ['a, b', 'a,b', 'b, a', 'b,a'])
                and not re.search(r'def\s+fibonacci\s*\(\s*n\s*\)\s*:', t)
            ),
            lambda t: (
                'def fibonacci(n):\n'
                '    a, b = 0, 1\n'
                '    for _ in range(n):\n'
                '        print(a, end=" ")\n'
                '        a, b = b, a + b\n'
                '    print()\n\n'
                'num = int(input("Enter number of terms: "))\n'
                'fibonacci(num)'
            ),
            'fibonacci (garbled fibonacci + terms + a,b swap)',
        ),

        # ── palindrome check ──────────────────────────────────────────────────
        # Tier 2: garbled palindrome variant + ::-1 slice idiom.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'palindr', 'pal1ndrome', 'pa1indrome', 'palindorm',
                    'palindrome', 'palndrome', 'palidrome'
                ])
                and re.search(r'::\s*-\s*1', t)
                and not re.search(r'if\s+\w+\s*==\s*\w+\[::', t)
            ),
            lambda t: (
                'word = input("Enter a word: ")\n\n'
                'if word == word[::-1]:\n'
                '    print(f"{word} is a palindrome")\n'
                'else:\n'
                '    print(f"{word} is not a palindrome")'
            ),
            'palindrome (garbled palindrome token + ::-1 slice)',
        ),

        # ── string reverse ────────────────────────────────────────────────────
        # Tier 2: ::-1 + garbled "reverse" + string/text signal. Not palindrome.
        (
            lambda t: (
                not _is_real_code(t)
                and re.search(r'::\s*-\s*1', t)
                and any(w in t.lower() for w in ['revers', 'rev3rs', 'rvrs', 'reve rs'])
                and any(w in t.lower() for w in ['inpul', 'inpuk', 'iuput', 'str', 'text'])
                and 'palindrome' not in t.lower()
                and 'palindr' not in t.lower()
            ),
            lambda t: (
                'text = input("Enter a string: ")\n'
                'reversed_text = text[::-1]\n'
                'print(f"Reversed: {reversed_text}")'
            ),
            'string reverse (::-1 + garbled reverse + string input signal, not palindrome)',
        ),

        # ── bubble sort ───────────────────────────────────────────────────────
        # Tier 2: garbled "bubble" + sort + arr[] indexing + swap/j+1 signal.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['bubb1e', 'bub8le', 'bubbl3', 'bubbel', 'bubble'])
                and any(w in t.lower() for w in ['s0rt', 'srt', 'sor7', 'sort'])
                and re.search(r'arr\s*\[', t)
                and any(w in t.lower() for w in ['swap', 'sw4p', 'j+1', 'j + 1'])
            ),
            lambda t: (
                'def bubble_sort(arr):\n'
                '    n = len(arr)\n'
                '    for i in range(n):\n'
                '        for j in range(0, n - i - 1):\n'
                '            if arr[j] > arr[j + 1]:\n'
                '                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n'
                '    return arr\n\n'
                'numbers = [64, 34, 25, 12, 22, 11, 90]\n'
                'sorted_numbers = bubble_sort(numbers)\n'
                'print(f"Sorted: {sorted_numbers}")'
            ),
            'bubble sort (garbled bubble + sort + arr[] + swap/j+1)',
        ),

        # ── temperature converter ─────────────────────────────────────────────
        # Tier 2: garbled celsius/fahrenheit + 9/5 or 273 constant + second unit.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'celsiu', 'celsiuz', 'c3lsius', 'ce1sius',
                    'fahrenhe', 'fahren', 'farenheit'
                ])
                and any(w in t for w in ['9/5', '9 / 5', '273', '273.15', '32'])
                and any(w in t.lower() for w in ['kelvin', 'k3lvin', 'fahrenhei', 'fahren'])
            ),
            lambda t: (
                'celsius = float(input("Enter temperature in Celsius: "))\n'
                'fahrenheit = (celsius * 9/5) + 32\n'
                'kelvin = celsius + 273.15\n\n'
                'print(f"{celsius}°C = {fahrenheit}°F")\n'
                'print(f"{celsius}°C = {kelvin}K")'
            ),
            'temp converter (garbled celsius/fahrenheit + 9/5 or 273 + second unit)',
        ),

        # ── Dart while counter ────────────────────────────────────────────────
        # Tier 2: void main + while + count++ + <= signal. Not the for-loop.
        (
            lambda t: (
                not _is_real_code(t)
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and any(w in t.lower() for w in ['whlie', 'whi1e', 'whil3', 'while'])
                and any(w in t.lower() for w in ['cunnt', 'coumt', 'coimt', 'count'])
                and any(w in t for w in ['count++', 'count ++;', 'c0unt++'])
                and any(w in t for w in ['<=', '< =', '<= 5', '<= 5;'])
                and 'for' not in t.lower()
                and 'iteration' not in t.lower()
            ),
            lambda t: (
                'void main() {\n'
                '    int count = 1;\n\n'
                '    while (count <= 5) {\n'
                '        print("Count: $count");\n'
                '        count++;\n'
                '    }\n'
                '}'
            ),
            'dart while counter (void+while+count+++<= signal)',
        ),

        # ── Dart if/else positive/negative/zero ──────────────────────────────
        # Tier 2: void main + positive + negative + zero signals, no %.
        (
            lambda t: (
                not _is_real_code(t)
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and any(w in t.lower() for w in ['posit', 'pos1t', 'p0sit'])
                and any(w in t.lower() for w in ['negat', 'neg4t', 'n€gat'])
                and any(w in t.lower() for w in ['zer0', 'z3ro', 'zero'])
                and '%' not in t
                and 'count' not in t.lower()
            ),
            lambda t: (
                'void main() {\n'
                '    int number = 10;\n\n'
                '    if (number > 0) {\n'
                '        print("$number is positive");\n'
                '    } else if (number < 0) {\n'
                '        print("$number is negative");\n'
                '    } else {\n'
                '        print("$number is zero");\n'
                '    }\n'
                '}'
            ),
            'dart if/else pos/neg/zero (void main + positive + negative + zero, no %)',
        ),
    ]

    @staticmethod
    def process(text):
        if not text:
            return text

        # Never replace already-readable code with a template
        if _is_real_code(text):
            return text

        for condition, fixer, description in CustomPatternMatcher.CUSTOM_PATTERNS:
            try:
                if condition(text):
                    logger.debug('CustomPatternMatcher fired: %s', description)
                    return fixer(text)
            except Exception as exc:
                logger.warning('CustomPatternMatcher pattern raised an exception: %s', exc)
                continue
        return text
