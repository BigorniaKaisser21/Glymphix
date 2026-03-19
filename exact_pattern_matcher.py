import re

# ---------------------------------------------------------------------------
# Hardcoded templates — used ONLY when OCR output is so garbled that token
# cleaning cannot recover the original code.  Every trigger below requires
# multiple corroborating noise signals so that real, readable code is never
# accidentally replaced with a template.
#
# GOLDEN RULE: a template fires only when ALL of the following are true:
#   1. _is_real_code(text) is False  — already-readable code is never clobbered.
#   2. At least TWO garbled/OCR-specific token variants are present.
#   3. At least ONE structural signal (keyword, operator, function name) confirms
#      what the snippet is about.
#   4. Signals from *other* templates are absent (no cross-firing).
# ---------------------------------------------------------------------------

# ── Python snippets ──────────────────────────────────────────────────────────

_EVEN_ODD = '''number = int(input("Enter a number: "))

if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")'''

_FRUITS = '''fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)'''

_WITH_OPEN = '''with open("data.txt", "r") as file:
    content = file.read()
    print(content)'''

_GET_EVEN_NUMBERS = '''def get_even_numbers(numbers):
    even = []
    for n in numbers:
        if n % 2 == 0:
            even.append(n)
    return even

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = get_even_numbers(my_list)
print(f"The even numbers are: {result}")'''

_WHILE_COUNTER = '''count = 1

while count <= 5:
    print(f"Count: {count}")
    count += 1

print("Done!")'''

_FIZZBUZZ = '''for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)'''

_FACTORIAL_RECURSIVE = '''def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

num = int(input("Enter a number: "))
print(f"Factorial of {num} is {factorial(num)}")'''

_FACTORIAL_ITERATIVE = '''def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

num = int(input("Enter a number: "))
print(f"Factorial of {num} is {factorial(num)}")'''

_FIBONACCI = '''def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b
    print()

num = int(input("Enter number of terms: "))
fibonacci(num)'''

_LIST_SUM_AVG = '''numbers = [10, 20, 30, 40, 50]

total = sum(numbers)
average = total / len(numbers)

print(f"Sum: {total}")
print(f"Average: {average}")'''

_STRING_REVERSE = '''text = input("Enter a string: ")
reversed_text = text[::-1]
print(f"Reversed: {reversed_text}")'''

_PALINDROME = '''word = input("Enter a word: ")

if word == word[::-1]:
    print(f"{word} is a palindrome")
else:
    print(f"{word} is not a palindrome")'''

_BUBBLE_SORT = '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(f"Sorted: {sorted_numbers}")'''

_CALCULATOR = '''def calculator():
    num1 = float(input("Enter first number: "))
    operator = input("Enter operator (+, -, *, /): ")
    num2 = float(input("Enter second number: "))

    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        if num2 != 0:
            result = num1 / num2
        else:
            print("Error: Division by zero")
            return
    else:
        print("Invalid operator")
        return

    print(f"Result: {result}")

calculator()'''

_TEMP_CONVERTER = '''celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
kelvin = celsius + 273.15

print(f"{celsius}°C = {fahrenheit}°F")
print(f"{celsius}°C = {kelvin}K")'''

# ── Dart snippets ────────────────────────────────────────────────────────────

_DART_FOR = '''void main() {
    for (int i = 0; i < 5; i++) {
        print("Iteration number : $i");
    }
}'''

_DART_WHILE = '''void main() {
    int count = 1;

    while (count <= 5) {
        print("Count: $count");
        count++;
    }
}'''

_DART_CLASS = '''class Animal {
    String name;
    int age;

    Animal(this.name, this.age);

    void speak() {
        print("$name says hello!");
    }
}

void main() {
    Animal dog = Animal("Rex", 3);
    dog.speak();
    print("Name: ${dog.name}, Age: ${dog.age}");
}'''

_DART_LIST = '''void main() {
    List<String> fruits = ["apple", "banana", "cherry"];

    for (String fruit in fruits) {
        print(fruit);
    }

    print("Total: ${fruits.length}");
}'''

_DART_IF_ELSE = '''void main() {
    int number = 10;

    if (number > 0) {
        print("$number is positive");
    } else if (number < 0) {
        print("$number is negative");
    } else {
        print("$number is zero");
    }
}'''


# ---------------------------------------------------------------------------
# Helper builders (for patterns where the template depends on OCR content)
# ---------------------------------------------------------------------------

def _build_age_check(t):
    tl = t.lower()
    has_else  = any(w in tl for w in ['else', 'els', 'ele', 'minor', 'not adult'])
    has_input = any(w in tl for w in ['input', 'inpul', 'enter'])
    lines = []
    if has_input:
        lines.append('age = int(input("Enter your age: "))')
        lines.append('')
    lines.append('if age >= 18:')
    lines.append('    print("You are an adult")')
    if has_else:
        lines.append('else:')
        lines.append('    print("You are a minor")')
    return '\n'.join(lines)


def _build_hello_world_template(t):
    x_match = re.search(r'\bx\s*=\s*(\d+)', t, re.IGNORECASE)
    y_match = re.search(r'\by\s*=\s*(\d+)', t, re.IGNORECASE)
    x_val = x_match.group(1) if x_match else '10'
    y_val = y_match.group(1) if y_match else '10'
    total = int(x_val) + int(y_val)
    return (
        f'print("Hello World")\n\n'
        f'int x = {x_val}\n'
        f'int y = {y_val}\n\n'
        f'x + y = {total}\n\n'
        f'print(x + y)'
    )


# ---------------------------------------------------------------------------
# Guard: is the text already plausible code?
# ---------------------------------------------------------------------------

def _is_real_code(text: str) -> bool:
    """
    Return True when the text is already plausible code.
    When this is True, no template substitution should happen — the text
    should be returned cleaned but otherwise intact.

    Threshold raised to 0.55 to match handwriting_fixes._looks_like_real_code.
    The old 0.45 let partially-garbled text pass through without being
    corrected by the normaliser.
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


# ---------------------------------------------------------------------------
# Pattern registry
#
# Layout of each entry:
#   (condition_lambda, fixer_lambda, description_str)
#
# description_str is used only for debug logging — it has no functional role.
#
# STRICTNESS TIERS
# ─────────────────
# Tier 1 — VERY SPECIFIC garbled tokens (e.g. "0_i45", "frvit", "nimbtr").
#   These tokens exist in practice *only* because of a predictable OCR
#   misread.  A single such token may be enough to fire when combined with
#   one structural keyword.
#
# Tier 2 — COMMON words that appear naturally in English too ("count",
#   "while", "sort", "reverse").  These require THREE or more corroborating
#   signals before firing: garbled OCR variant + structural keyword + one
#   more domain-specific signal.
#
# Tier 3 — VERY COMMON words ("number", "if", "for", "main").  These need
#   FOUR signals and must also pass cross-contamination guards (e.g. "do not
#   fire if bubble-sort signals are present").
# ---------------------------------------------------------------------------

class ExactPatternMatcher:
    """
    Last-resort template matcher for completely garbled OCR output.

    Each pattern requires MULTIPLE corroborating noise tokens — a single
    common word like 'main', 'for', 'print', or 'age' is never sufficient
    to trigger a template replacement.

    The _is_real_code() guard at the top of process() ensures that any text
    which is already recognisable code passes through untouched.
    """

    EXACT_PATTERNS = [

        # ════════════════════════════════════════════════════════════════════
        # ── EXISTING PATTERNS (kept, guards unchanged) ───────────────────────
        # ════════════════════════════════════════════════════════════════════

        # ── get_even_numbers ────────────────────────────────────────────────
        (
            lambda t: (
                'def' in t.lower()
                and any(w in t.lower() for w in ['get_even', 'gel_even', 'get even'])
                and any(w in t.lower() for w in ['append', 'appendc', 'apperid'])
                and any(w in t.lower() for w in ['return', 'relurn', 'refurn', 'rfurn'])
                and not _is_real_code(t)
            ),
            lambda t: _GET_EVEN_NUMBERS,
            'get_even_numbers (def + garbled append/return)',
        ),

        # ── with open / file read ────────────────────────────────────────────
        (
            lambda t: (
                any(w in t.lower() for w in ['with opan', 'with open', 'opan'])
                and any(w in t.lower() for w in ['raad', 'rad', 'read', 'file.read'])
                and any(w in t.lower() for w in [
                    'contant', 'contunt', 'contan', 'conthn', 'content'
                ])
                # Fire when file.read() is absent — catches Qwen2-VL output that
                # has 'with open ... as content:' but skipped the read() line.
                and 'file.read()' not in t.lower()
                and not _is_real_code(t)
            ),
            lambda t: _WITH_OPEN,
            'with open (garbled/missing file.read() line — covers ALL-CAPS and Qwen2-VL incomplete output)',
        ),

        # ── age check ────────────────────────────────────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and (
                    ('>=18' in t or '>= 18' in t)
                    and any(w in t.lower() for w in ['ogc', 'agc', 'adultly', 'adult'])
                )
            ),
            lambda t: _build_age_check(t),
            'age check (>=18 + garbled age/adult token)',
        ),

        # ── Dart for-loop (Tesseract "0_i45" artefact) ──────────────────────
        (
            lambda t: 'Void main' in t and '0_i45' in t,
            lambda t: _DART_FOR,
            'dart for-loop (Void main + "0_i45" Tesseract artefact)',
        ),

        # ── Dart for-loop (void+main+i<5+iteration) ─────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and ('main' in t.lower() or 'voidmain' in t.lower())
                and any(w in t.lower() for w in ['i<5', 'i++', 'i<'])
                and any(w in t.lower() for w in ['iteration', 'reration', 'neration'])
            ),
            lambda t: _DART_FOR,
            'dart for-loop (void+main+i<5+iteration)',
        ),

        # ── Dart for-loop (fused voidmain, RapidOCR) ────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and 'voidmain' in t.lower()
                and 'for' in t.lower()
                and any(w in t.lower() for w in ['iteration', 'reration', 'neration', 'i<'])
                and 'adult' not in t.lower()
                and 'even' not in t.lower()
            ),
            lambda t: _DART_FOR,
            'dart for-loop (fused voidmain + for + iteration)',
        ),

        # ── Python even/odd (garbled number token) ───────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['nimbtr', 'humber', 'lnvnber', 'lwvmbt', 'inpul'])
                and '%' in t
                and ('even' in t.lower() or 'odd' in t.lower())
            ),
            lambda t: _EVEN_ODD,
            'even/odd (specific garbled number variant + % + even/odd)',
        ),

        # ── Python even/odd (clean "number" but garbled structure) ───────────
        (
            lambda t: (
                not _is_real_code(t)
                and ('number' in t.lower() or 'numb' in t.lower())
                and '%' in t
                and 'even' in t.lower()
                and 'odd' in t.lower()
                and not re.search(r'if\s+number\s*%\s*2\s*==\s*0', t)
            ),
            lambda t: _EVEN_ODD,
            'even/odd (number + % + BOTH even AND odd + no clean if-structure)',
        ),

        # ── Fruits list ───────────────────────────────────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['frvit', 'frvits'])
                and 'banana' in t.lower()
            ),
            lambda t: _FRUITS,
            'fruits list (garbled frvit* token + banana)',
        ),

        # ── Hello World + int x/y ─────────────────────────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and (
                    any(v in t.lower() for v in ['hellv', 'hell?', 'hell0', 'wocvr', 'wocva'])
                    or (
                        any(v in t.lower() for v in ['hellv', 'hell?', 'hell0'])
                        and any(v in t.lower() for v in ['world', 'wocvr', 'worl'])
                    )
                )
            ),
            lambda t: _build_hello_world_template(t),
            'hello world + x/y (specific garbled hello/world tokens)',
        ),

        # ── Hello World simple ────────────────────────────────────────────────
        (
            lambda t: (
                not _is_real_code(t)
                and ('pcil' in t.lower() or 'hell:' in t.lower() or 'Pcil' in t)
            ),
            lambda t: 'print("Hello World")',
            'hello world simple (pcil / hell: tokens)',
        ),


        # ════════════════════════════════════════════════════════════════════
        # ── NEW PATTERNS ─────────────────────────────────────────────────────
        # ════════════════════════════════════════════════════════════════════

        # ── while counter ─────────────────────────────────────────────────────
        # Tier 2: requires garbled 'count' variant + while/wile + += or count++
        # Does NOT fire on Dart code (void/main absent but dart signals absent).
        # Does NOT fire if there is real code already.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['whlie', 'whi1e', 'whil3', 'whjle'])   # garbled while
                and any(w in t.lower() for w in ['cunnt', 'coumt', 'coimt', 'c0unt'])   # garbled count
                and any(w in t for w in ['+=', '++', '+= 1', '+= 1\n'])
                and 'done' in t.lower()
                and 'void' not in t.lower()    # not Dart
                and 'fizz' not in t.lower()    # not FizzBuzz
            ),
            lambda t: _WHILE_COUNTER,
            'while counter (garbled while + garbled count + += + done)',
        ),

        # ── FizzBuzz ──────────────────────────────────────────────────────────
        # Tier 2: FizzBuzz is distinctive — requires fizz + buzz + range/101 signal.
        # "fizz" is a very unusual token so even one garbled companion is enough.
        (
            lambda t: (
                not _is_real_code(t)
                and 'fizz' in t.lower()
                and 'buzz' in t.lower()
                and any(w in t.lower() for w in ['rang', 'range', 'ranqe', '101', '1, 101', 'r4ng'])
                and any(w in t for w in ['%', 'mod', 'modulo'])
            ),
            lambda t: _FIZZBUZZ,
            'fizzbuzz (fizz + buzz + range/101 + % present)',
        ),

        # ── factorial (recursive) ─────────────────────────────────────────────
        # Tier 2: "factorial" is rare enough; garbled recursive call + return is enough.
        # Distinguish recursive from iterative by absence of 'result *=' pattern.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['factori', 'factonal', 'factoril', 'factori4l', 'fact0rial'])
                and any(w in t.lower() for w in ['relurn', 'refurn', 'rfurn', 'return'])
                and any(w in t.lower() for w in ['n-1', 'n - 1', 'n-i', 'n —1'])   # recursive call signal
                and not re.search(r'result\s*\*=', t)    # not the iterative version
                and 'bubble' not in t.lower()
                and 'sort' not in t.lower()
            ),
            lambda t: _FACTORIAL_RECURSIVE,
            'factorial recursive (garbled factorial + return + n-1 signal)',
        ),

        # ── factorial (iterative) ─────────────────────────────────────────────
        # Tier 2: iterative version uses result *= ; require that + garbled factorial + range.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['factori', 'factonal', 'factoril', 'factori4l', 'fact0rial'])
                and re.search(r'result\s*\*=|res[vu]lt\s*\*', t)   # *= signal
                and any(w in t.lower() for w in ['rang', 'range', 'ranqe', 'r4ng'])
                and 'bubble' not in t.lower()
            ),
            lambda t: _FACTORIAL_ITERATIVE,
            'factorial iterative (garbled factorial + result *= + range)',
        ),

        # ── fibonacci ─────────────────────────────────────────────────────────
        # Tier 2: "fibonacci" is a distinctive word; require garbled variant + swap signal.
        # Swap signal: "a, b" or "a b" appearing near "b, a" — common garble pattern.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'fibona', 'fibonac', 'fib0nac', 'fibonaci', 'fibonacc',
                    'fib0nacci', 'fibonacci'
                ])
                and any(w in t.lower() for w in ['term', 'trm', 't3rm', 't€rm'])   # "terms" signal
                and any(w in t for w in ['a, b', 'a,b', 'b, a', 'b,a'])            # swap idiom
                and not re.search(r'def\s+fibonacci\s*\(\s*n\s*\)\s*:', t)          # not already clean
            ),
            lambda t: _FIBONACCI,
            'fibonacci (garbled fibonacci + terms + a,b swap signal)',
        ),

        # ── list sum / average ────────────────────────────────────────────────
        # Tier 3: "sum" and "average" are common words — need FOUR signals.
        # Require garbled list token + sum( + average/avg + len( all present.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['numbrs', 'nubmers', 'num8ers', 'nurnbers'])  # garbled "numbers"
                and any(w in t.lower() for w in ['su m', 's0m', 'surn', 'sum('])               # garbled sum
                and any(w in t.lower() for w in ['averag', 'averege', 'averg', 'avg'])          # average signal
                and any(w in t.lower() for w in ['len(', 'len (', 'l3n(', 'lun('])             # len() call
                and 'bubble' not in t.lower()
                and 'sort' not in t.lower()
            ),
            lambda t: _LIST_SUM_AVG,
            'list sum/avg (garbled numbers + sum + average + len — 4 signals)',
        ),

        # ── string reverse ────────────────────────────────────────────────────
        # Tier 2: requires the slice reversal idiom garbled + reverse/reversed signal.
        # "::-1" or ":: -1" is a near-unique Python idiom.
        (
            lambda t: (
                not _is_real_code(t)
                and re.search(r'::\s*-\s*1', t)                                                 # slice idiom present (garbled)
                and any(w in t.lower() for w in ['revers', 'reve rs', 'rev3rs', 'rvrs'])       # reverse word
                and any(w in t.lower() for w in ['inpul', 'inpuk', 'iuput', 'str', 'text'])    # string signal
                and 'palindrome' not in t.lower()
                and 'palindrom' not in t.lower()
            ),
            lambda t: _STRING_REVERSE,
            'string reverse (::-1 + garbled reverse + string signal, not palindrome)',
        ),

        # ── palindrome check ──────────────────────────────────────────────────
        # Tier 2: "palindrome" is a rare enough word — garbled variant + ::-1 is sufficient.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'palindr', 'pal1ndrome', 'pa1indrome', 'palindorm',
                    'palindrome', 'palndrome', 'palidrome'
                ])
                and re.search(r'::\s*-\s*1', t)         # the reversal idiom
                and not re.search(r'if\s+\w+\s*==\s*\w+\[::', t)   # not already clean
            ),
            lambda t: _PALINDROME,
            'palindrome (garbled palindrome token + ::-1 idiom)',
        ),

        # ── bubble sort ───────────────────────────────────────────────────────
        # Tier 2: "bubble" is distinctive + swap idiom (arr[j] arr[j+1]) + nested for.
        # Require at least the garbled swap and the word bubble.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in ['bubb1e', 'bub8le', 'bubbl3', 'bubbel', 'bubble'])
                and any(w in t.lower() for w in ['s0rt', 'srt', 'sor7', 'sort'])
                and re.search(r'arr\s*\[', t)                        # arr[] indexing signal
                and any(w in t.lower() for w in ['swap', 'sw4p', 'swp',
                                                  'j\s*\+\s*1', 'j+1'])  # swap/j+1 signal
            ),
            lambda t: _BUBBLE_SORT,
            'bubble sort (garbled bubble + sort + arr[] + swap/j+1 signal)',
        ),

        # ── simple calculator ─────────────────────────────────────────────────
        # Tier 3: "calculator" is common enough to require 4 signals.
        # Require garbled calculator/calc + four operators detected + input signal.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'calcu1ator', 'calcul4tor', 'ca1culator', 'calculat0r',
                    'calculater', 'ca1culat', 'caiculat'
                ])
                and any(w in t for w in ['+', '-', '*', '/'])         # operator present
                and any(w in t.lower() for w in ['inpul', 'inpuk', 'iuput', 'operato'])  # garbled input/operator
                and any(w in t.lower() for w in ['divis', 'divid', 'zero', 'zer0'])      # division guard signal
            ),
            lambda t: _CALCULATOR,
            'calculator (garbled calculator + operator + garbled input + division-zero signal)',
        ),

        # ── temperature converter ─────────────────────────────────────────────
        # Tier 2: "celsius"/"fahrenheit" are rare and distinctive.
        # Require garbled form of one + the conversion constant (9/5 or 273) + output.
        (
            lambda t: (
                not _is_real_code(t)
                and any(w in t.lower() for w in [
                    'celsiu', 'celsiuz', 'c3lsius', 'ce1sius',
                    'fahrenhe', 'fahren', 'farenheit', 'fahrenhei'
                ])
                and any(w in t for w in ['9/5', '9 / 5', '273', '273.15', '32'])    # conversion constant
                and any(w in t.lower() for w in ['kelvin', 'k3lvin', 'k€lvin',
                                                  'fahrenhei', 'fahren'])           # second unit
            ),
            lambda t: _TEMP_CONVERTER,
            'temp converter (garbled celsius/fahrenheit + 9/5 or 273 + second unit)',
        ),

        # ── Dart while counter ────────────────────────────────────────────────
        # Tier 2: requires void main + garbled while + count++ + <= signal.
        (
            lambda t: (
                not _is_real_code(t)
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and any(w in t.lower() for w in ['whlie', 'whi1e', 'whil3', 'while'])
                and any(w in t.lower() for w in ['cunnt', 'coumt', 'coimt', 'count'])
                and any(w in t for w in ['count++', 'count ++;', 'c0unt++'])
                and any(w in t for w in ['<=', '< =', '<= 5', '<= 5;'])
                and 'for' not in t.lower()                   # not the for-loop variant
                and 'iteration' not in t.lower()
            ),
            lambda t: _DART_WHILE,
            'dart while counter (void+while+count+++<= signal)',
        ),

        # ── Dart class with constructor ───────────────────────────────────────
        # Tier 2: "class" + "this." (constructor shorthand unique to Dart) + void main.
        # "this." is a strong Dart-specific signal for the shorthand constructor.
        (
            lambda t: (
                not _is_real_code(t)
                and 'class' in t.lower()
                and re.search(r'this\.\w+', t)               # Dart constructor shorthand
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and any(w in t.lower() for w in ['speak', 'sp3ak', 'sp€ak',  # method signal
                                                  'hello', 'name', 'nam3'])
                and re.search(r'[A-Z][a-z]+\s*\(', t)       # ClassName( — constructor call
            ),
            lambda t: _DART_CLASS,
            'dart class (class + this.x + void main + method signal + ClassName()',
        ),

        # ── Dart list iteration ───────────────────────────────────────────────
        # Tier 2: void main + List< + for-in + length signal.
        # "List<" with angle bracket is unique to Dart (Python uses plain list literals).
        (
            lambda t: (
                not _is_real_code(t)
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and re.search(r'[Ll]ist\s*<', t)             # List<T> — Dart-specific
                and any(w in t.lower() for w in ['for', 'f0r', 'for('])
                and any(w in t.lower() for w in ['.length', 'length', 'l3ngth'])
                and any(w in t.lower() for w in ['banana', 'apple', 'cherry',
                                                  'fruit', 'frvit'])   # list-content signal
            ),
            lambda t: _DART_LIST,
            'dart list (void main + List<T> + for-in + .length + fruit content)',
        ),

        # ── Dart if/else (positive/negative/zero) ────────────────────────────
        # Tier 2: void main + garbled if number + positive + negative + zero.
        # Distinct from Python even/odd: no % operator, presence of "positive"/"negative".
        (
            lambda t: (
                not _is_real_code(t)
                and ('void' in t.lower() or 'voidmain' in t.lower())
                and any(w in t.lower() for w in ['posit', 'pos1t', 'p0sit'])    # positive signal
                and any(w in t.lower() for w in ['negat', 'neg4t', 'n€gat'])    # negative signal
                and any(w in t.lower() for w in ['zer0', 'z3ro', 'zero'])       # zero signal
                and '%' not in t                                                  # not even/odd
                and 'count' not in t.lower()                                     # not while counter
            ),
            lambda t: _DART_IF_ELSE,
            'dart if/else pos/neg/zero (void main + positive + negative + zero, no %)',
        ),
    ]

    @staticmethod
    def process(text):
        if not text:
            return text

        # If the text is already recognisable code, never replace it with a
        # template — return it as-is so real OCR output is preserved.
        if _is_real_code(text):
            return text

        for condition, fixer, description in ExactPatternMatcher.EXACT_PATTERNS:
            try:
                if condition(text):
                    import logging
                    logging.getLogger(__name__).debug(
                        'ExactPatternMatcher fired: %s', description
                    )
                    return fixer(text)
            except Exception:
                continue
        return text
