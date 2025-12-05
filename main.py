from backend.spider import spider
from backend.indexer import build_index
from gui.app import search_engine_gui

def main():
    html_files = spider(max_pages=9000)
    inverted_index, doc_metadata, docs, doc_vectors, vocab = build_index(html_files)
    search_engine_gui(inverted_index, doc_metadata, docs, doc_vectors, vocab)

if __name__ == "__main__":
    main()
