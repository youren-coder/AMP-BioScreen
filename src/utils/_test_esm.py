import importlib, sys, os

# Verify esm import
for mod_name in ["torch", "esm"]:
    try:
        importlib.import_module(mod_name)
        print(f"{mod_name}: OK")
    except ImportError as e:
        print(f"{mod_name}: NOT FOUND - {e}")

print("Environment ready")
