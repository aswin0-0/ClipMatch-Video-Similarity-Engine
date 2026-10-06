"""
verify_imports.py — Quick sanity check that all ClipMatch modules import cleanly.
Run: python verify_imports.py
"""
import sys

checks = [
    "config",
    "database",
    "services.frame_extractor",
    "services.ncc",
    "services.matcher",
    "routers.match",
    "main",
]

failures = []
for mod in checks:
    try:
        __import__(mod)
        print(f"  [OK]   {mod}")
    except Exception as e:
        print(f"  [FAIL] {mod} -> {e}")
        failures.append(mod)

print()
if failures:
    print(f"FAILED: {len(failures)} module(s) had import errors.")
    sys.exit(1)
else:
    print("All modules imported successfully. ClipMatch is ready.")
