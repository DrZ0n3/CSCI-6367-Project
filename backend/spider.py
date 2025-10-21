import os
from bs4 import BeautifulSoup
from collections import deque

def spider(base_folder="data/rhf", max_pages=100):
    visited = set()
    html_files = []

    # Find starting point
    start_file = os.path.join(base_folder, "index.html")
    if not os.path.exists(start_file):
        for root, files in os.walk(base_folder):
            if "index.html" in files:
                start_file = os.path.join(root, "index.html")
                break
        else:
            raise FileNotFoundError("index.html not found in subfolders")

    print(f"=== Starting crawl from {start_file} ===")

    # Use a queue 
    queue = deque([start_file])

    while queue and len(visited) < max_pages:
        file_path = queue.popleft()
        if file_path in visited:
            continue
        visited.add(file_path)
        html_files.append(file_path)
        print(f"🕷️ Visiting: {file_path}")

        # Try reading with multiple encodings
        for enc in ("utf-8", "windows-1252", "iso-8859-1", "macroman"):
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    soup = BeautifulSoup(f, "lxml")
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Could not read {file_path} with any encoding.")
            continue

        # Get directory of current file and normalize paths
        current_dir = os.path.dirname(file_path)
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith((".html", ".htm")):
                target_path = os.path.normpath(os.path.join(current_dir, href))
                if os.path.exists(target_path) and target_path not in visited:
                    print(f"  ↳ Found link to: {target_path}")
                    queue.append(target_path)

    print(f"\n=== Crawl complete! Total files found: {len(html_files)} ===")
    return html_files
