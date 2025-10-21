from backend.spider import spider
from backend.indexer import build_index
from gui.app import search_engine_gui

def main():

    html_files = spider(max_pages=20)
    inverted_index, doc_metadata, docs, doc_vectors, vocab = build_index(html_files)
    search_engine_gui( inverted_index, doc_metadata, docs, doc_vectors, vocab)
    
    print("\nFinal list of HTML files:\n")
    for file in html_files:
        print(file)

if __name__ == "__main__":
    main()