import os
from bs4 import BeautifulSoup
from collections import deque

def spider(base_folder="data/rhf", max_pages=100):
    visited = set()
    html_files = []

    # try direct index.html first 
    start_file = os.path.join(base_folder, "index.html")
    if not os.path.exists(start_file):
        for root, _, files in os.walk(base_folder):
            if "index.html" in files:
                start_file = os.path.join(root, "index.html")
                break
        else:
            raise FileNotFoundError("index.html not found in subfolders")

    print(f"=== Starting crawl from {start_file} ===")

    queue = deque([start_file])

    # Pre-binding for speed
    normpath = os.path.normpath
    join = os.path.join
    exists = os.path.exists
    BeautifulSoup_local = BeautifulSoup

    # Predefine allowed HTML extensions
    html_ext = (".html", ".htm")

    while queue and len(visited) < max_pages:

        file_path = queue.popleft()

        if file_path in visited:
            continue

        visited.add(file_path)
        html_files.append(file_path)

        # READ WITH MINIMAL ENCODING ATTEMPTS 
        soup = None
        for enc in ("utf-8", "latin1"):
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    # Using lxml parser 
                    soup = BeautifulSoup_local(f, "lxml")
                break
            except Exception:
                continue

        if soup is None:
            print(f"Could not read {file_path}")
            continue

        # Prebind to avoid repeated attribute lookups
        current_dir = os.path.dirname(file_path)
        append_queue = queue.append

        #  FAST HYPERLINK EXTRACTION 
        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Skip non-HTML links early
            if not href.endswith(html_ext):
                continue

            target_path = normpath(join(current_dir, href))

            # Avoid duplicate os.path.exists calls
            if target_path not in visited and exists(target_path):
                append_queue(target_path)

    print(f"\n=== Crawl complete! Total files found: {len(html_files)} ===")
    return html_files
