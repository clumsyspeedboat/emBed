# tests/conftest.py
import sys, os, importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT, 'src')

# Put repo root (and src dir) first on path
for p in (SRC_DIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# If a non-local 'src' was already imported, drop it
if 'src' in sys.modules:
    mfile = getattr(sys.modules['src'], '__file__', '') or ''
    if not os.path.abspath(mfile).startswith(SRC_DIR):
        del sys.modules['src']

# Import local src to lock it in
importlib.import_module('src')
