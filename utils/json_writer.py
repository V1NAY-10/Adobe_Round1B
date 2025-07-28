# utils/json_writer.py

import json
import datetime

# This function likely exists in your file already, keep it.
def build_output_json(inp, top_idx, section_info, refined_texts):
    """Builds the final JSON object for output."""
    out_json = {
        "metadata": {
            "input_documents": [d["filename"] for d in inp["documents"]],
            "persona": inp["persona"]["role"],
            "job_to_be_done": inp["job_to_be_done"]["task"],
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": [] 
    }
    for i, idx in enumerate(top_idx, 1):
        sec = section_info[idx]
        out_json["extracted_sections"].append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": i,
            "page_number": sec["page_number"]
        })
        # Add subsection analysis if available
        if refined_texts[idx]:
             out_json["subsection_analysis"].append({
                "document": sec["document"],
                "refined_text": refined_texts[idx],
                "page_number": sec["page_number"]
            })
    return out_json


def write_json(obj, path):
    """
    Writes a Python dictionary to a JSON file, explicitly using UTF-8 encoding
    to prevent common Windows errors.
    """
    try:
        # Use the open() context manager with explicit encoding='utf-8'
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"ERROR: Failed to write JSON file at {path}. Reason: {e}")