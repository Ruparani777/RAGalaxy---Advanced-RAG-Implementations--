$script = @'
# pdf_probe.py
# Run this from your activated env: python pdf_probe.py
import sys, os, hashlib

path = "Data/bank_loan_recoverydata.pdf"
print("File path:", path)

# 1) pypdf
print("\n--- pypdf probe ---")
try:
    from pypdf import PdfReader
    r = PdfReader(path)
    pages = len(r.pages)
    print("pypdf pages:", pages)
    if pages > 0:
        text = r.pages[0].extract_text() or ""
        print("pypdf first page snippet:", text[:500])
except Exception as e:
    print("pypdf error:", repr(e))

# 2) pdfplumber
print("\n--- pdfplumber probe ---")
try:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        print("pdfplumber pages:", pages)
        if pages > 0:
            text = pdf.pages[0].extract_text() or ""
            print("pdfplumber first page snippet:", text[:500])
except Exception as e:
    print("pdfplumber error:", repr(e))

# 3) PyMuPDF (fitz)
print("\n--- PyMuPDF (fitz) probe ---")
try:
    import fitz  # pymupdf
    doc = fitz.open(path)
    pages = doc.page_count
    print("PyMuPDF pages:", pages)
    if pages > 0:
        text = doc.load_page(0).get_text() or ""
        print("PyMuPDF first page snippet:", text[:500])
except Exception as e:
    print("PyMuPDF error:", repr(e))

# 4) Basic file info
print("\n--- Basic file info ---")
try:
    size = os.path.getsize(path)
    print("Size (bytes):", size)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    print("SHA256:", h.hexdigest())
except Exception as e:
    print("File info error:", repr(e))

print("\nDone.")
'@

py