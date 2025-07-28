import fitz, re, numpy as np
from collections import Counter

def normalize(t): return re.sub(r'\s+', ' ', t.strip().lower())

def classify(txt, fs, avg):
    h1 = r'^[0-9]+\.\s+'; h2 = r'^[0-9]+\.[0-9]+\s+'; h3 = r'^[0-9]+\.[0-9]+\.[0-9]+\s+'
    if re.match(h1, txt) or fs/avg>1.25: return "H1"
    if re.match(h2, txt) or fs/avg>1.15: return "H2"
    if re.match(h3, txt) or fs/avg>1.05: return "H3"
    return "O"

def has_content_after(blocks,i):
    if i+1>=len(blocks): return False
    p,y,fs = blocks[i]["page"], blocks[i]["y_pos"], blocks[i]["font_size"]
    for nb in blocks[i+1:i+21]:
        if nb["page"]!=p: break
        if nb["y_pos"]<=y: continue
        if nb["word_count"]>=5 or nb["font_size"]<=fs: return True
    return False

def score(txt,fs,avg,bold,caps,colon,wc):
    s=0
    if fs/avg>1.1: s+=.3
    if bold: s+=.2
    if caps and wc<=6: s+=.1
    if colon: s+=.1
    if wc<=10: s+=.15
    if re.match(r'^[0-9]+\.\s+',txt): s+=.4
    elif re.match(r'^[0-9]+\.[0-9]+\s+',txt): s+=.3
    elif re.match(r'^[0-9]+\.[0-9]+\.[0-9]+\s+',txt): s+=.2
    return min(s,1.0)

def extract_title(blocks):
    fp=[b for b in blocks if b["page"]==1]
    if not fp: return ""
    max_fs=max(b["font_size"] for b in fp)
    cand=[b for b in fp if b["font_size"]>=.98*max_fs]
    best=min(cand,key=lambda x:x["y_pos"])
    return best["text"].strip()

def extract_outline(pdf):
    doc=fitz.open(pdf)
    hf=[]
    for page in doc:
        hy,fy=page.rect.height*.10,page.rect.height*.90
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type",0)!=0 or 'lines' not in blk: continue
            y=blk['bbox'][1]
            if y<hy or y>fy:
                txt=" ".join(s["text"] for l in blk["lines"] for s in l["spans"]).strip()
                if txt: hf.append(txt)
    hf_rm={t for t,c in Counter(hf).items() if c>=2}

    blocks=[]
    for p, page in enumerate(doc):
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type",0)!=0: continue
            for ln in blk.get("lines",[]):
                if not ln.get("spans"): continue
                sp=ln["spans"][0]
                txt=" ".join(s["text"].strip() for s in ln["spans"]).strip()
                if len(txt)<3 or txt in hf_rm: continue
                blocks.append({
                    "text":txt,"page":p+1,"font_size":sp["size"],
                    "bold":"bold" in sp["font"].lower() or "black" in sp["font"].lower(),
                    "is_caps":txt.isupper(),"ends_colon":txt.endswith(":"),
                    "word_count":len(txt.split()),"y_pos":sp["bbox"][1]
                })

    if not blocks: return {"title":"","outline":[]}
    avg=np.median([b["font_size"] for b in blocks if b['word_count']>5])
    title=extract_title(blocks); title_n=normalize(title)

    outline,seen=[],set()
    for i,b in enumerate(blocks):
        n=normalize(b["text"])
        if n in seen or n==title_n: continue
        if score(b["text"],b["font_size"],avg,b["bold"],b["is_caps"],
                 b["ends_colon"],b["word_count"])>0.55 and has_content_after(blocks,i):
            lv=classify(b["text"],b["font_size"],avg)
            if lv!="O":
                outline.append({"level":lv,"text":b["text"],"page":b["page"]})
                seen.add(n)

    # hierarchy fix
    for i in range(1,len(outline)):
        c=int(outline[i]['level'][1]); p=int(outline[i-1]['level'][1])
        if c-p>1: outline[i]['level']=f"H{p+1}"

    return {"title":title,"outline":outline}
