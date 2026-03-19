import re

# ─────────────────────────────────────────────────────────────────────────────
# OCR noise variants for library/module names
# ─────────────────────────────────────────────────────────────────────────────
LIBRARY_OCR_VARIANTS = {
    # ── Python Standard ───────────────────────────────────────────────────
    'math':       ['math', 'mat h', 'math'],
    'random':     ['random', 'rand om', 'randam', 'randan', 'randem'],
    'datetime':   ['datetime', 'date time', 'datetme'],
    'os':         ['os', 'o s'],
    'sys':        ['sys', 'sy s'],
    'json':       ['json', 'js on'],
    'time':       ['time', 'tim e'],
    'string':     ['string', 'str ing'],
    'collections':['collections', 'collect ions'],
    'itertools':  ['itertools', 'iter tools', 'itertool'],
    'functools':  ['functools', 'func tools', 'functool'],
    're':         ['import re', 'import r e'],
    'pathlib':    ['pathlib', 'path lib', 'path'],
    'typing':     ['typing', 'typ ing'],
    'copy':       ['copy', 'cop y'],
    'io':         ['import io'],  # bare 'io' is too short — matches OCR noise like voidmain

    # ── Python Data Science ───────────────────────────────────────────────
    'numpy':      ['numpy', 'num py', 'nump y', 'import np', 'import numpy'],
    'pandas':     ['pandas', 'pand as', 'import pd', 'import pandas'],
    'matplotlib': ['matplotlib', 'mat plotlib', 'import plt', 'import matplotlib'],
    'scipy':      ['scipy', 'sci py'],
    'sklearn':    ['sklearn', 'scikit', 'scikit-learn', 'sk learn'],
    'tensorflow': ['tensorflow', 'import tf', 'tensor flow'],
    'torch':      ['torch', 'pytorch', 'py torch'],
    'keras':      ['keras', 'ker as'],
    'seaborn':    ['seaborn', 'sea born', 'import sns'],
    'cv2':        ['cv2', 'opencv', 'open cv'],
    'PIL':        ['pil', 'pillow', 'image', 'PIL'],

    # ── Python Web ────────────────────────────────────────────────────────
    'requests':   ['requests', 'request s'],
    'flask':      ['flask', 'flas k'],
    'django':     ['django', 'djan go'],
    'fastapi':    ['fastapi', 'fast api'],
    'sqlalchemy': ['sqlalchemy', 'sql alchemy'],
    'aiohttp':    ['aiohttp', 'aio http'],
    'bs4':        ['bs4', 'beautifulsoup', 'beautiful soup'],
    'selenium':   ['selenium', 'selen ium'],
    'urllib':     ['urllib', 'url lib'],
    'http':       ['http', 'http.client'],
    'socket':     ['socket', 'sock et'],

    # ── Dart core ─────────────────────────────────────────────────────────
    'dart:core':      ['dart:core', 'dart core'],
    'dart:math':      ['dart:math', 'dart math', 'dart : math'],
    'dart:io':        ['dart:io', 'dart io', 'dart : io'],
    'dart:convert':   ['dart:convert', 'dart convert'],
    'dart:async':     ['dart:async', 'dart async'],
    'dart:collection':['dart:collection', 'dart collection'],
    'dart:typed_data':['dart:typed_data', 'typed data'],

    # ── Flutter ───────────────────────────────────────────────────────────
    'flutter':             ['flutter', 'flutt er'],
    'material':            ['material', 'mater ial', 'material.dart'],
    'cupertino':           ['cupertino', 'cupert ino'],
    'StatelessWidget':     ['statelesswidget', 'stateless widget', 'stateless'],
    'StatefulWidget':      ['statefulwidget', 'stateful widget', 'stateful'],
    'build':               ['build(', 'build (', 'buildcontext'],
    'scaffold':            ['scaffold', 'scaffo ld'],
    'appbar':              ['appbar', 'app bar'],
    'text':                ['text(', 'text ('],
    'container':           ['container', 'contain er'],
    'column':              ['column', 'col umn'],
    'row':                 ['row(', 'row ('],
    'listview':            ['listview', 'list view'],
    'navigator':           ['navigator', 'navig ator'],
    'setState':            ['setstate', 'set state'],
    'initState':           ['initstate', 'init state'],
    'provider':            ['provider', 'provid er'],
    'riverpod':            ['riverpod', 'river pod'],
    'getx':                ['getx', 'get x'],
}

# ─────────────────────────────────────────────────────────────────────────────
# Library → canonical import statement
# ─────────────────────────────────────────────────────────────────────────────
LIBRARY_IMPORTS = {
    # Python Standard
    'math':        'import math',
    'random':      'import random',
    'datetime':    'from datetime import datetime',
    'os':          'import os',
    'sys':         'import sys',
    'json':        'import json',
    'time':        'import time',
    'string':      'import string',
    'collections': 'from collections import defaultdict, Counter',
    'itertools':   'import itertools',
    'functools':   'import functools',
    're':          'import re',
    'pathlib':     'from pathlib import Path',
    'typing':      'from typing import List, Dict, Optional',
    'copy':        'import copy',
    'io':          'import io',

    # Python Data Science
    'numpy':       'import numpy as np',
    'pandas':      'import pandas as pd',
    'matplotlib':  'import matplotlib.pyplot as plt',
    'scipy':       'import scipy',
    'sklearn':     'from sklearn.model_selection import train_test_split',
    'tensorflow':  'import tensorflow as tf',
    'torch':       'import torch',
    'keras':       'from tensorflow import keras',
    'seaborn':     'import seaborn as sns',
    'cv2':         'import cv2',
    'PIL':         'from PIL import Image',

    # Python Web
    'requests':    'import requests',
    'flask':       'from flask import Flask, render_template, request',
    'django':      'from django.db import models',
    'fastapi':     'from fastapi import FastAPI',
    'sqlalchemy':  'from sqlalchemy import create_engine',
    'aiohttp':     'import aiohttp',
    'bs4':         'from bs4 import BeautifulSoup',
    'selenium':    'from selenium import webdriver',
    'urllib':      'from urllib.request import urlopen',
    'http':        'import http.client',
    'socket':      'import socket',

    # Dart
    'dart:core':       "import 'dart:core';",
    'dart:math':       "import 'dart:math';",
    'dart:io':         "import 'dart:io';",
    'dart:convert':    "import 'dart:convert';",
    'dart:async':      "import 'dart:async';",
    'dart:collection': "import 'dart:collection';",
    'dart:typed_data': "import 'dart:typed_data';",

    # Flutter
    'flutter':             "import 'package:flutter/material.dart';",
    'material':            "import 'package:flutter/material.dart';",
    'cupertino':           "import 'package:flutter/cupertino.dart';",
    'provider':            "import 'package:provider/provider.dart';",
    'riverpod':            "import 'package:flutter_riverpod/flutter_riverpod.dart';",
    'getx':                "import 'package:get/get.dart';",
}

# ─────────────────────────────────────────────────────────────────────────────
# Library → common usage template
# ─────────────────────────────────────────────────────────────────────────────
LIBRARY_TEMPLATES = {
    'math': '''import math

result = math.sqrt(16)
print(result)

print(math.pi)
print(math.factorial(5))''',

    'random': '''import random

num = random.randint(1, 100)
print(num)

items = ["apple", "banana", "cherry"]
print(random.choice(items))''',

    'datetime': '''from datetime import datetime

now = datetime.now()
print(now)
print(now.strftime("%Y-%m-%d %H:%M:%S"))''',

    'os': '''import os

print(os.getcwd())
files = os.listdir(".")
for f in files:
    print(f)''',

    'json': '''import json

data = {"name": "Alice", "age": 25}
json_str = json.dumps(data)
print(json_str)

parsed = json.loads(json_str)
print(parsed["name"])''',

    'numpy': '''import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr.mean())
print(arr.sum())''',

    'pandas': '''import pandas as pd

df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
print(df)
print(df.describe())''',

    'matplotlib': '''import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.title("My Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()''',

    'requests': '''import requests

response = requests.get("https://api.example.com/data")
print(response.status_code)
print(response.json())''',

    'flask': '''from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)''',

    'torch': '''import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
print(tensor)
print(tensor.mean())''',

    'tensorflow': '''import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.summary()''',

    'cv2': '''import cv2

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()''',

    'dart:math': '''import 'dart:math';

void main() {
    double result = sqrt(16);
    print(result);
    print(pi);
    print(pow(2, 10));
}''',

    'dart:io': '''import 'dart:io';

void main() {
    stdout.write("Enter your name: ");
    String? name = stdin.readLineSync();
    print("Hello, $name!");
}''',

    'dart:convert': '''import 'dart:convert';

void main() {
    String jsonStr = \'{"name": "Alice", "age": 25}\';
    Map<String, dynamic> data = jsonDecode(jsonStr);
    print(data["name"]);
    print(jsonEncode(data));
}''',

    'flutter': '''import 'package:flutter/material.dart';

void main() {
    runApp(MyApp());
}

class MyApp extends StatelessWidget {
    @override
    Widget build(BuildContext context) {
        return MaterialApp(
            home: Scaffold(
                appBar: AppBar(title: Text("My App")),
                body: Center(child: Text("Hello World")),
            ),
        );
    }
}''',

    'StatelessWidget': '''import 'package:flutter/material.dart';

class MyWidget extends StatelessWidget {
    @override
    Widget build(BuildContext context) {
        return Scaffold(
            appBar: AppBar(title: Text("My Widget")),
            body: Center(child: Text("Hello World")),
        );
    }
}''',

    'pathlib': '''from pathlib import Path

p = Path(".")
for file in p.iterdir():
    print(file)

new_file = Path("example.txt")
new_file.write_text("Hello World")
print(new_file.read_text())''',

    'itertools': '''import itertools

nums = [1, 2, 3]
for combo in itertools.combinations(nums, 2):
    print(combo)

for perm in itertools.permutations(nums, 2):
    print(perm)''',

    'functools': '''import functools

@functools.lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))''',

    'scipy': '''import scipy
from scipy import stats
import numpy as np

data = [2, 4, 4, 4, 5, 5, 7, 9]
print(stats.mean(data))
print(stats.stdev(data))''',

    'seaborn': '''import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()''',

    'fastapi': '''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}''',

    'bs4': '''from bs4 import BeautifulSoup
import requests

response = requests.get("https://example.com")
soup = BeautifulSoup(response.text, "html.parser")

title = soup.find("title")
print(title.text)

links = soup.find_all("a")
for link in links:
    print(link.get("href"))''',

    'selenium': '''from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://example.com")

element = driver.find_element(By.TAG_NAME, "h1")
print(element.text)

driver.quit()''',

    'StatefulWidget': '''import 'package:flutter/material.dart';

class MyWidget extends StatefulWidget {
    @override
    _MyWidgetState createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
    int _count = 0;

    @override
    Widget build(BuildContext context) {
        return Scaffold(
            appBar: AppBar(title: Text("Counter")),
            body: Center(child: Text("Count: $_count")),
            floatingActionButton: FloatingActionButton(
                onPressed: () => setState(() => _count++),
                child: Icon(Icons.add),
            ),
        );
    }
}''',
}


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LibraryDetector:
    """
    Detects Python and Dart/Flutter library usage from noisy OCR text.
    Returns corrected import statements and/or code templates.
    """

    @staticmethod
    def detect_libraries(text):
        """Return list of detected library names from OCR text."""
        t = text.lower()
        detected = []
        for lib_name, variants in LIBRARY_OCR_VARIANTS.items():
            for variant in variants:
                v = variant.lower()
                # Skip very short variants (<=3 chars) unless they appear as a
                # whole word — prevents 'io', 'os', 're' etc. matching mid-word
                # in OCR noise like 'voidmain', 'iteration', 'number'
                if len(v) <= 3:
                    import re as _re
                    if not _re.search(r'(?<![a-z])' + _re.escape(v) + r'(?![a-z])', t):
                        continue
                if v in t:
                    if lib_name not in detected:
                        detected.append(lib_name)
                    break
        return detected

    @staticmethod
    def get_imports(libraries):
        """Return correct import statements for detected libraries."""
        imports = []
        seen = set()
        for lib in libraries:
            stmt = LIBRARY_IMPORTS.get(lib)
            if stmt and stmt not in seen:
                imports.append(stmt)
                seen.add(stmt)
        return imports

    @staticmethod
    def get_template(libraries):
        """Return the best matching code template."""
        # Priority order — more specific wins
        priority = [
            'StatefulWidget', 'StatelessWidget', 'flutter', 'material', 'cupertino',
            'riverpod', 'provider', 'getx',
            'tensorflow', 'torch', 'sklearn', 'matplotlib', 'seaborn',
            'pandas', 'numpy', 'scipy', 'cv2', 'PIL',
            'flask', 'django', 'fastapi', 'requests', 'bs4', 'selenium',
            'sqlalchemy', 'aiohttp',
            'dart:io', 'dart:math', 'dart:convert', 'dart:async',
            'dart:collection', 'dart:typed_data',
            'pathlib', 'json', 'datetime', 'random', 'math', 'os',
            'sys', 'time', 're', 'collections', 'itertools', 'functools',
            'typing', 'copy', 'string', 'socket', 'urllib',
        ]
        for lib in priority:
            if lib in libraries and lib in LIBRARY_TEMPLATES:
                return LIBRARY_TEMPLATES[lib]
        return None

    @staticmethod
    def fix_import_line(text):
        """Fix a garbled import line to its correct form."""
        t = text.lower()
        # Detect 'import' keyword (various OCR forms)
        has_import = any(w in t for w in ['import', 'im port', 'impart', 'inport'])
        has_from = 'from' in t or 'fron' in t or 'fro m' in t

        if not has_import and not has_from:
            return None

        detected = LibraryDetector.detect_libraries(text)
        if not detected:
            return None

        imports = LibraryDetector.get_imports(detected)
        return '\n'.join(imports) if imports else None

    @staticmethod
    def process(text):
        """
        Main entry point. Returns reconstructed code if libraries detected,
        otherwise returns original text unchanged.

        A full template is only returned when the OCR text does NOT already
        contain real code structure (if/for/def/class/print…).  This prevents
        the detector from discarding actual user code just because a library
        name appears somewhere in it.

        Noise guard: if fewer than 15 % of tokens in the text are recognisable
        code words, the text is too garbled to reliably detect library names
        and the original text is returned unchanged.  This prevents Tesseract
        noise like "| | a \\ a > | ' { } : | } !" from matching 'os' via the
        whole-word regex and incorrectly prepending 'import os'.
        """
        if not text:
            return text

        # ── Noise guard ──────────────────────────────────────────────────────
        import re as _re
        _KNOWN = _re.compile(
            r'\b(if|else|elif|for|while|def|class|return|import|from|print|input|'
            r'int|str|float|bool|void|main|number|age|fruit|fruits|even|odd|'
            r'random|randint|secret|guess|'
            r'True|False|None|append|range|len|open|read|write|with|as|pass|'
            r'break|continue|and|or|not|in|age|adult|minor)\b',
            _re.IGNORECASE
        )
        words = _re.findall(r'\b\w+\b', text)
        if words:
            density = sum(1 for w in words if _KNOWN.match(w)) / len(words)
            if density < 0.15:
                print(f"📚 LibraryDetector: text too garbled (density={density:.2f}) — skipping")
                return text
        # ─────────────────────────────────────────────────────────────────────

        detected = LibraryDetector.detect_libraries(text)
        if not detected:
            return text

        print(f"📚 LibraryDetector found: {detected}")

        # Check for real code signals — if present, only prepend missing imports
        # rather than replacing the whole text with a boilerplate template.
        CODE_SIGNALS = [
            'def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ',
            'return ', 'print(', 'input(', 'void main', '{', '};',
        ]
        has_real_code = any(sig in text for sig in CODE_SIGNALS)

        if not has_real_code:
            template = LibraryDetector.get_template(detected)
            if template:
                print(f"✅ LibraryDetector returning template for: {detected[0]}")
                return template

        # Text contains real code — only prepend any import lines that are absent
        imports = LibraryDetector.get_imports(detected)
        if imports:
            missing = [imp for imp in imports if imp not in text]
            if missing:
                print(f"✅ LibraryDetector prepending missing imports for: {detected}")
                return '\n'.join(missing) + '\n\n' + text

        return text
