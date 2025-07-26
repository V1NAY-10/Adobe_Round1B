# utils/extractor.py

import fitz  # PyMuPDF

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    headings = []
    seen_fonts = {}

    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    font_size = round(span["size"], 1)
                    font = span["font"]

                    if not text or len(text) < 4 or text.replace(" ", "").isdigit():
                        continue

                    seen_fonts[font_size] = seen_fonts.get(font_size, 0) + 1

                    headings.append({
                        "text": text,
                        "size": font_size,
                        "font": font,
                        "page": page_number
                    })

    sorted_sizes = sorted(seen_fonts.items(), key=lambda x: -x[1])
    top_sizes = [s[0] for s in sorted_sizes[:3]]
    size_to_level = {}
    if len(top_sizes) >= 3:
        size_to_level = {top_sizes[0]: "H1", top_sizes[1]: "H2", top_sizes[2]: "H3"}
    elif len(top_sizes) == 2:
        size_to_level = {top_sizes[0]: "H1", top_sizes[1]: "H2"}
    elif len(top_sizes) == 1:
        size_to_level = {top_sizes[0]: "H1"}

    structured = {
        "title": doc.metadata.get("title") or pdf_path,
        "outline": []
    }

    for h in headings:
        level = size_to_level.get(h["size"])
        if level:
            structured["outline"].append({
                "level": level,
                "text": h["text"],
                "page": h["page"]
            })

    return structured
