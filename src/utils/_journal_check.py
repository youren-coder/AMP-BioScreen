import requests, re
import urllib3
urllib3.disable_warnings()

journals = {
    "Computational and Structural Biotechnology Journal (CSBJ)": [
        "https://www.sciencedirect.com/journal/computational-and-structural-biotechnology-journal",
        "https://www.elsevier.com/journals/computational-and-structural-biotechnology-journal/2001-0370/open-access-options",
    ],
    "Journal of Chemical Information and Modeling (JCIM)": [
        "https://pubs.acs.org/journal/jcisd8",
        "https://pubs.acs.org/page/jcisd8/submission/authors.html",
    ],
    "BMC Bioinformatics": [
        "https://bmcbioinformatics.biomedcentral.com/",
        "https://bmcbioinformatics.biomedcentral.com/submission-guidelines/fees-and-funding",
    ],
}

for name, urls in journals.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    for url in urls[:2]:
        try:
            r = requests.get(url, timeout=20, verify=False, 
                           headers={'User-Agent': 'Mozilla/5.0'})
            print(f"  URL: {url}")
            print(f"  Status: {r.status_code}, Size: {len(r.text)}")
            
            # Extract key info
            text = r.text
            
            # APC / fees
            for kw in ['APC', 'article processing charge', 'publication fee', 'open access fee', 'page charge', 'fee']:
                idx = text.lower().find(kw)
                if idx >= 0:
                    snippet = re.sub(r'<[^>]+>', ' ', text[idx:idx+300])
                    snippet = re.sub(r'\s+', ' ', snippet).strip()
                    if len(snippet) > 30:
                        print(f"  [{kw}]: {snippet[:250]}")
                        break
            
            # Review time
            for kw in ['review', 'decision', 'turnaround', 'submission to first', 'peer review']:
                idx = text.lower().find(kw)
                if idx >= 0 and kw != 'review' or (kw == 'review' and 'time' in text[idx:idx+50].lower()):
                    snippet = re.sub(r'<[^>]+>', ' ', text[idx:idx+200])
                    snippet = re.sub(r'\s+', ' ', snippet).strip()
                    if len(snippet) > 20 and ('day' in snippet.lower() or 'week' in snippet.lower() or 'month' in snippet.lower()):
                        print(f"  [{kw}]: {snippet[:200]}")
                        break
            
            # Impact factor
            for kw in ['impact factor', 'IF ', 'CiteScore']:
                idx = text.lower().find(kw)
                if idx >= 0:
                    snippet = re.sub(r'<[^>]+>', ' ', text[idx:idx+150])
                    snippet = re.sub(r'\s+', ' ', snippet).strip()
                    print(f"  [{kw}]: {snippet[:150]}")
                    break
                    
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}")
