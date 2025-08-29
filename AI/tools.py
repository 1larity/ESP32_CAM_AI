"""Utility tools for the Qt MDI app.
 - Image ingestion from a source directory into face/pet stores
 - Simple near-duplicate culling via average-hash (aHash)
"""
from __future__ import annotations
import os
import time
from typing import Tuple
import numpy as np
import cv2


def ingest_images(src_dir: str, dest_dir: str, size: Tuple[int,int], gray: bool=False) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    count=0
    for root,_,files in os.walk(src_dir):
        for fn in files:
            if not fn.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            try:
                img=cv2.imread(os.path.join(root,fn))
                if img is None: continue
                if gray:
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img=cv2.resize(img, size)
                now=time.time(); ms=int((now-int(now))*1000); ds=time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
                out=os.path.join(dest_dir, f'{ds}_{ms:03d}.jpg')
                cv2.imwrite(out, img)
                count+=1
            except Exception:
                pass
    return count


def cull_similar_in_dir(target: str, hash_size: int = 8, hamming_thresh: int = 4) -> int:
    """Remove near-duplicate images in a folder using aHash similarity.
    Returns number of removed files.
    """
    if not os.path.isdir(target):
        return 0
    files=[os.path.join(target,f) for f in os.listdir(target) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort(key=lambda p: os.path.getmtime(p))
    hashes=[]; removed=0
    def ahash(path):
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            img=cv2.resize(img,(hash_size,hash_size))
            avg=img.mean(); bits=(img>avg).astype(np.uint8)
            return bits
        except Exception:
            return None
    for fp in files:
        h=ahash(fp)
        if h is None: continue
        dup=False
        for hh in hashes:
            dist = int((h^hh).sum())
            if dist <= hamming_thresh:
                try:
                    os.remove(fp); removed+=1
                except Exception:
                    pass
                dup=True
                break
        if not dup:
            hashes.append(h)
    return removed


def find_similar_in_dir(target: str, hash_size: int = 8, hamming_thresh: int = 4):
    """Analyze a directory for near-duplicates by aHash.
    Returns (files:list[str], remove_indices:set[int]) where remove_indices
    contains indices in files suggested for deletion.
    """
    if not os.path.isdir(target):
        return [], set()
    files=[os.path.join(target,f) for f in os.listdir(target) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort(key=lambda p: os.path.getmtime(p))
    hashes=[]; remove=set()
    def ahash(path):
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            img=cv2.resize(img,(hash_size,hash_size))
            avg=img.mean(); bits=(img>avg).astype(np.uint8)
            return bits
        except Exception:
            return None
    for idx, fp in enumerate(files):
        h=ahash(fp)
        if h is None: continue
        for hh in hashes:
            dist = int((h^hh).sum())
            if dist <= hamming_thresh:
                remove.add(idx)
                break
        else:
            hashes.append(h)
    return files, remove
