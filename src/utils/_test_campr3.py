import requests, re
import urllib3
urllib3.disable_warnings()

# Final check: PGAT-ABPp moonseter repo contents
r = requests.get("https://api.github.com/repos/moonseter/PGAT-ABPp/contents", timeout=15)
if r.status_code == 200:
    files = [f['name'] for f in r.json()]
    print(f"PGAT-ABPp moonseter files ({len(files)}):")
    for f in files[:20]:
        print(f"  {f}")
    py_files = [f for f in files if f.endswith('.py')]
    print(f"Python files: {py_files}")

# Check AMPpred-MFA
r2 = requests.get("https://api.github.com/repos/Jiangle525/AMPpred-MFA/contents", timeout=15)
if r2.status_code == 200:
    files2 = [f['name'] for f in r2.json()]
    print(f"\nAMPpred-MFA files ({len(files2)}):")
    for f in files2[:20]:
        print(f"  {f}")
else:
    print(f"\nAMPpred-MFA: {r2.status_code}")
    # Try alternative
    r3 = requests.get("https://api.github.com/search/repositories?q=AMPpred-MFA&per_page=3", timeout=15)
    if r3.status_code == 200:
        items = r3.json().get('items', [])
        for item in items:
            print(f"  Search found: {item['full_name']} - {item['html_url']}")

print("\n=== FINAL SUMMARY ===")
print("CAMPR3: Webserver POST works but prediction engine broken (empty response)")
print("iAMPpred: Server unreachable")
print("AMPScanner v2: Cloned, requires TF1.x + password, not deployable")
print("AMPpred-MFA: GitHub repo exists but inaccessible / empty")
print("PGAT-ABPp: 73MB repo, 6 stars, moonseter fork - needs pip install")
print("LMPred: Model loads but needs ProtTrans embeddings extraction")
print()
print("Actionable: Try pip install PGAT-ABPp or use LMPred with ProtTrans")
