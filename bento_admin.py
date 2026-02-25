import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# 枠ごとの判定ルール
# =========================================================
COMP_RULES = {
    "左上": {"baseline_fill": 0.75, "allow_shortage": 0.12},
    "右上": {"baseline_fill": 0.85, "allow_shortage": 0.10},
    "左下": {"baseline_fill": 0.92, "allow_shortage": 0.06},
    "右下": {"baseline_fill": 0.75, "allow_shortage": 0.12},
}

# =========================================================
# フォント
# =========================================================
def _find_font_path(preferred: str) -> str:
    candidates = [
        preferred,
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

# =========================================================
# 容器検出
# =========================================================
def get_bento_mask_and_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    edges = cv2.Canny(blur, 30, 120)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask, (x,y,w,h)

# =========================================================
# 食材検出
# =========================================================
def get_food_mask(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    _, color = cv2.threshold(s, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    rice = cv2.inRange(hsv, (0,0,160),(180,90,255))

    combined = cv2.bitwise_or(color, rice)
    combined = cv2.bitwise_and(combined, combined, mask=mask)
    return combined

# =========================================================
# 区画（固定）
# =========================================================
TEMPLATE = [
    ("左上", 0.04,0.06,0.34,0.48),
    ("右上", 0.35,0.06,0.96,0.48),
    ("左下", 0.04,0.52,0.70,0.95),
    ("右下", 0.72,0.52,0.96,0.95),
]

def build_comps(mask, bbox):
    x,y,w,h = bbox
    comps=[]
    for name,l,t,r,b in TEMPLATE:
        x0=int(x+l*w); y0=int(y+t*h)
        x1=int(x+r*w); y1=int(y+b*h)
        m=np.zeros_like(mask)
        cv2.rectangle(m,(x0,y0),(x1,y1),255,-1)
        m=cv2.bitwise_and(m,mask)
        comps.append((name,(x0,y0,x1,y1),m))
    return comps

# =========================================================
# 描画＆判定
# =========================================================
def draw_results(img, comps, food_mask, font_path):
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base=Image.fromarray(img_rgb).convert("RGBA")
    overlay=Image.new("RGBA", base.size, (0,0,0,0))
    draw=ImageDraw.Draw(overlay)

    font=_load_font(font_path,60)

    results=[]

    for name,(x0,y0,x1,y1),m in comps:
        area=np.count_nonzero(m)
        if area==0: continue

        food=np.count_nonzero(cv2.bitwise_and(food_mask,m))
        fill=food/area

        rule=COMP_RULES[name]
        base_fill=rule["baseline_fill"]
        allow=rule["allow_shortage"]

        shortage=max(0,(base_fill-fill)/base_fill)
        judge="OK" if shortage<=allow else "NG"

        sukaska=int(shortage*100)

        cx=(x0+x1)//2
        cy=(y0+y1)//2

        color=(255,0,0,255) if judge=="NG" else (255,255,255,255)

        draw.text((cx,cy), f"{sukaska}%", fill=color, font=font, anchor="mm")

        results.append({
            "name":name,
            "shortage":shortage,
            "judge":judge
        })

    out=Image.alpha_composite(base,overlay).convert("RGB")
    return np.array(out),results

# =========================================================
# UI
# =========================================================
st.title("スカスカ弁当 判定")

uploads=st.file_uploader("画像", type=["jpg","png"], accept_multiple_files=True)

font_path=_find_font_path("")

if uploads:
    rows=[]
    previews={}

    for up in uploads:
        img_bytes=np.frombuffer(up.getvalue(),np.uint8)
        img=cv2.imdecode(img_bytes,cv2.IMREAD_COLOR)

        mask,bbox=get_bento_mask_and_bbox(img)
        if mask is None: continue

        comps=build_comps(mask,bbox)
        food=get_food_mask(img,mask)

        render,details=draw_results(img,comps,food,font_path)

        overall="NG" if any(d["judge"]=="NG" for d in details) else "OK"

        area_str=", ".join([f'{d["name"]}:{int(d["shortage"]*100)}%({d["judge"]})' for d in details])

        rows.append({
            "ファイル名":up.name,
            "判定":overall,
            "詳細":area_str
        })

        previews[up.name]=render

    df=pd.DataFrame(rows)
    st.dataframe(df)

    for name,img in previews.items():
        st.image(img, caption=name)
