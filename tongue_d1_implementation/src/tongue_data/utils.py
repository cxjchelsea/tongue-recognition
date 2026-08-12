from pathlib import Path
import hashlib
from PIL import Image
import pandas as pd

IMAGE_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def file_md5(path: Path, chunk_size=1024*1024):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def image_size(path: Path):
    with Image.open(path) as im:
        return int(im.width), int(im.height)

def is_image(path: Path):
    return path.suffix.lower() in IMAGE_EXT

def infer_split(path: Path):
    parts = [p.lower() for p in path.parts]
    for s in ("train","val","valid","validation","test"):
        if s in parts:
            return "val" if s in ("valid","validation") else s
    return None

def read_table_auto(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python")

def normalize_na(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in {"na","n/a","nan","none","null",""}:
        return None
    return v
