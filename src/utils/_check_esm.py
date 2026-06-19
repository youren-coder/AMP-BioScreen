import esm
methods = [m for m in dir(esm.pretrained) if "load" in m.lower()]
print("ESM pretrained load methods:", methods[:10])
