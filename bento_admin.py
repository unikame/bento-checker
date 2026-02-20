import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

# =========================
# 画像処理
# =========================

def get_bento_mask(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        edges = cv2.Canny(blur, 30, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask


def get_food_mask(img_bgr, bento_mask, v_min, s_max):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined


# =========================
# 区切り（テンプレ方式）
# =========================
def detect_compartments_template(img_bgr, bento_mask):
    ys, xs = np.where(bento_mask > 0)
    if len(xs) == 0:
        return [bento_mask]

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    W = max(1, x1 - x0)
    H = max(1, y1 - y0)

    rects = [
        ("TL", 0.03, 0.05, 0.35, 0.47),
        ("TR", 0.36, 0.05, 0.97, 0.47),
        ("BL", 0.03, 0.50, 0.72, 0.95),
        ("BR", 0.74, 0.50, 0.97, 0.95),
    ]

    masks = []
    for _, l, t, r, b in rects:
        rx0 = int(x0 + l * W)
        ry0 = int(y0 + t * H)
        rx1 = int(x0 + r * W)
        ry1 = int(y0 + b * H)

        m = np.zeros_like(bento_mask)
        cv2.rectangle(m, (rx0, ry0), (rx1, ry1), 255, -1)
        m = cv2.bitwise_and(m, bento_mask)
        masks.append(m)

    return masks


# =========================
# 描画
# =========================
def draw_results(img_bgr, comp_masks, food_mask):
    output = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_txts = []
    colors = [(0,255,0), (255,255,255), (255,0,0), (255,255,0)]

    for i, m in enumerate(comp_masks):
        area_px = np.count_nonzero(m)
        if area_px == 0:
            continue

        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        ratio = max(0, (area_px - food_px) / area_px * 100)

        ys, xs = np.where(m > 0)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        color = colors[i % len(colors)]
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 12)

        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)

        txt = f"{int(ratio)}%"
        cv2.putText(output, txt, (cx-70, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,0,0), 18, cv2.LINE_AA)
        cv2.putText(output, txt, (cx-70, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,255,255), 5, cv2.LINE_AA)

        res_txts.append(f"{int(ratio)}%")

    return output, res_txts


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

st.markdown("<h1>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 設定")
    ok_limit = st.slider("OK上限 (%)", 0, 50, 20)
    v_min = st.slider("ご飯の白さ", 0, 255, 170)
    s_max = st.slider("彩度上限", 0, 255, 70)

uploads = st.file_uploader("画像アップロード", type=["jpg", "png"], accept_multiple_files=True)

if uploads:
    results = []
    previews = {}

    for up in uploads:
        file_bytes = np.frombuffer(up.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        b_mask = get_bento_mask(img)
        if b_mask is None:
            continue

        f_mask = get_food_mask(img, b_mask, v_min, s_max)

        # ★ここが今回の本質
        comps = detect_compartments_template(img, b_mask)

        render, area_details = draw_results(img, comps, f_mask)

        total_ratio = (np.count_nonzero(b_mask) - np.count_nonzero(f_mask)) / np.count_nonzero(b_mask) * 100

        results.append({
            "ファイル名": up.name,
            "全体空白率": f"{total_ratio:.1f}%",
            "エリア別": ", ".join(area_details),
            "判定": "OK" if total_ratio < ok_limit else "NG"
        })

        previews[up.name] = render

    df = pd.DataFrame(results)

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.dataframe(df, use_container_width=True)

    with c2:
        if len(df) > 0:
            fname = df.iloc[0]["ファイル名"]
            st.image(previews[fname], use_container_width=True)
