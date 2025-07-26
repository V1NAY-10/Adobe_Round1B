import json, os
from datetime import datetime

def build_output_json(input_json, top_idx, section_info, refined_texts):
    """Compose final Round-1B output JSON structure."""
    metadata = {
        "input_documents": [d["filename"] for d in input_json["documents"]],
        "persona":         input_json["persona"]["role"],
        "job_to_be_done":  input_json["job_to_be_done"]["task"],
        "processing_timestamp": datetime.utcnow().isoformat(timespec="seconds")
    }

    extracted_sections   = []
    subsection_analysis  = []

    for rank, idx in enumerate(top_idx, start=1):
        info = section_info[idx]
        extracted_sections.append({
            "document":       info["document"],
            "section_title":  info["section_title"],
            "importance_rank": rank,
            "page_number":    info["page_number"]
        })
        subsection_analysis.append({
            "document":      info["document"],
            "refined_text":  refined_texts[idx],
            "page_number":   info["page_number"]
        })

    return {
        "metadata":           metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
