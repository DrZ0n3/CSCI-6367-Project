import tkinter as tk
import webbrowser

def insert_url(text_widget, url, doc_id=None):
    """Insert a clickable URL into a Tkinter Text widget."""
    if not url:
        return
    
    # Make unique tag (optional if multiple URLs)
    tag = f"url_{doc_id or url.replace('https://', '').replace('/', '_')}"
    
    # Record where the insertion starts
    start_idx = text_widget.index(tk.INSERT)
    
    # Insert the URL text
    text_widget.insert(tk.END, f"URL: {url}\n")
    
    # Record where it ends
    end_idx = text_widget.index(tk.INSERT)
    
    # Tag just the URL portion (skip the "URL: " prefix)
    text_widget.tag_add(tag, f"{start_idx}+5c", end_idx.strip())
    
    # Style for the hyperlink
    text_widget.tag_config(tag, foreground="blue", underline=True)
    
    # Bind events for click and hover
    text_widget.tag_bind(tag, "<Button-1>", lambda e, link=url: webbrowser.open_new(link))
    text_widget.tag_bind(tag, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
    text_widget.tag_bind(tag, "<Leave>", lambda e: text_widget.config(cursor=""))

# Example usage
root = tk.Tk()
root.title("Insert Clickable URL")

results_text = tk.Text(root, wrap="word", height=6, width=60)
results_text.pack(padx=10, pady=10)

insert_url(results_text, "https://example.com", doc_id=1)
insert_url(results_text, "https://python.org", doc_id=2)

root.mainloop()