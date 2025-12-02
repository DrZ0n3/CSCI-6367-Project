import os
from bs4 import BeautifulSoup, SoupStrainer
from collections import deque
from functools import lru_cache

def spider(base_folder="data/rhf", max_pages=9000):
    visited = set()
    html_files = []

    # Fast: directly check main index, skip os.walk if not needed
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

    # Only parse <a href=...> tags, skip rest of HTML tree
    link_only = SoupStrainer("a")

    # Cache filesystem checks 
    @lru_cache(maxsize=10000)
    def fast_exists(path):
        return os.path.exists(path)

    while queue and len(visited) < max_pages:
        file_path = queue.popleft()
        if file_path in visited:
            continue

        visited.add(file_path)
        html_files.append(file_path)

        # Try reading with best-guess encoding first
        content = None
        for enc in ("utf-8", "latin1"):
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                pass

        if content is None:
            print(f"Could not read {file_path}")
            continue

        current_dir = os.path.dirname(file_path)

        # fast link parsing
        soup = BeautifulSoup(content, "lxml", parse_only=link_only)

        for link in soup:
    # Skip non-tag elements (Doctype, comments, strings, etc.)
            if not hasattr(link, "get"):
                continue

            href = link.get("href")
            if not href or not href.endswith((".html", ".htm")):
                continue

            target_path = os.path.normpath(os.path.join(current_dir, href))

            if fast_exists(target_path) and target_path not in visited:
                queue.append(target_path)


    print(f"\n=== Crawl complete! Total files found: {len(html_files)} ===")
    return html_files
