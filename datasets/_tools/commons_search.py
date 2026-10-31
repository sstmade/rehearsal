"""Search Wikimedia Commons (bitmap files) and print license/author/size per hit as JSON lines.
Usage: python -I commons_search.py OUT.jsonl "query 1" "query 2" ...   (uses curl for HTTP)"""
import json, subprocess, sys, time, urllib.parse, re

API = "https://commons.wikimedia.org/w/api.php"
UA = "RehearsalDataScout/0.1 (hackathon test-set curation)"


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(5):
        raw = subprocess.run(["curl", "-s", "-A", UA, url], capture_output=True, check=True).stdout
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            time.sleep(3 * (k + 1))
    return {}


def strip(h):
    return re.sub(r"<[^>]+>", "", h or "").strip()


out = open(sys.argv[1], "a")
for q in sys.argv[2:]:
    time.sleep(1)
    r = get({"action": "query", "format": "json", "generator": "search", "gsrsearch": f"{q} filetype:bitmap",
             "gsrnamespace": 6, "gsrlimit": 40, "prop": "imageinfo",
             "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 320})
    for p in (r.get("query", {}).get("pages", {}) or {}).values():
        ii = p["imageinfo"][0]; m = ii.get("extmetadata", {})
        rec = {"q": q, "title": p["title"], "w": ii["width"], "h": ii["height"], "mime": ii["mime"],
               "url": ii["url"], "thumb": ii.get("thumburl"), "page": ii["descriptionurl"],
               "license": strip(m.get("LicenseShortName", {}).get("value")),
               "license_url": m.get("LicenseUrl", {}).get("value"),
               "artist": strip(m.get("Artist", {}).get("value")),
               "desc": strip(m.get("ImageDescription", {}).get("value"))[:200]}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(rec["license"], rec["w"], rec["h"], rec["title"])
