import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List, Dict

# =========================================================
# 1) 容器検出（マスク + 外接矩形）
# =========================================================

def get_bento_mask_and_bbox(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    容器（トレー）の最大輪郭を狙ってマスク化し、外接矩形(bbox)も返す
    bbox = (x, y, w, h)
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # 小さすぎる輪郭しかない場合は失敗扱い
    if area < (H * W * 0.12):
        return None, None

    x, y, w, h = cv2.boundingRect(c)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    # 縁の誤検出を減らすため少し内側に
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)

    return mask, (x, y, w, h)

# =========================================================
# 2) 食材マスク
# =========================================================

def get_food_mask(img_bgr: np.ndarray, bento_mask: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 彩度で色物（おかず）を拾う
    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 明度で白物（ご飯）を拾う（容器の白も拾いやすいので閾値で調整）
    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    # 少し安定化（白ノイズの穴埋め＆平滑化）
    kernel = np.ones((5, 5), np.uint8)
    rice_mask = cv2.morphologyEx(rice_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    rice_mask = cv2.medianBlur(rice_mask, 5)

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined

# =========================================================
# 3) 4分割（この容器専用：固定テンプレ）
# =========================================================

# rect = (name, left, top, right, bottom)  ※ 0-1比率
TEMPLATE_FIXED_4: List[Tuple[str, float, float, float, float]] = [
    ("左上", 0.04, 0.06, 0.34, 0.48),
    ("右上", 0.35, 0.06, 0.96, 0.48),
    ("左下", 0.04, 0.52, 0.70, 0.95),
    ("右下", 0.72, 0.52, 0.96, 0.95),
]

def build_compartments_fixed_4(
    bento_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> List[Tuple[str, Tuple[int, int, int, int], np.ndarray]]:
    """
    固定テンプレ(比率) + bboxから、区画の矩形とマスクを作る
    return: [(name, (x0,y0,x1,y1), mask), ...]
    """
    x, y, w, h = bbox

    comps = []
    for name, l, t, r, b in TEMPLATE_FIXED_4:
        x0 = int(x + l * w)
        y0 = int(y + t * h)
        x1 = int(x + r * w)
        y1 = int(y + b * h)

        m = np.zeros_like(bento_mask)
        cv2.rectangle(m, (x0, y0), (x1, y1), 255, thickness=-1)

        # 容器外に出ないようクリップ
        m = cv2.bitwise_and(m, bento_mask)

        comps.append((name, (x0, y0, x1, y1), m))

    return comps

# =========================================================
# 4) 描画＆計算（文字サイズ自動調整）
# =========================================================

def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray):
    output = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_txts = []

    colors = [
        (0, 255, 0),     # 緑
        (255, 255, 255), # 白
        (255, 0, 0),     # 赤
        (255, 255, 0),   # 黄
    ]

    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        area_px = np.count_nonzero(m)
        if area_px == 0:
            continue

        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        ratio = max(0.0, (area_px - food_px) / area_px * 100.0)
        ratio_int = int(round(ratio))

        # 枠線
        color = colors[i % len(colors)]
        cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness=10)

        # --- 文字サイズを枠サイズで自動調整 ---
        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)
        box_size = min(box_w, box_h)

        font_scale = box_size / 200.0
        font_scale = max(0.8, min(font_scale, 2.5))

        thickness_main = max(2, int(font_scale * 3))
        thickness_border = max(6, int(font_scale * 8))

        # 中央寄せ
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        txt = f"{ratio_int}%"

        (text_w, text_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_main)
        tx = cx - text_w // 2
        ty = cy + text_h // 2

        # 縁取り → 本体
        cv2.putText(output, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_border, cv2.LINE_AA)
        cv2.putText(output, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_main, cv2.LINE_AA)

        res_txts.append((name, ratio))

    return output, res_txts

# =========================================================
# 5) Streamlit UI
# =========================================================

st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")
st.markdown("<h1 style='margin:0;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)
st.caption("※この版は「赤い4分割容器」専用です（テンプレ固定）")

with st.sidebar:
    st.header("⚙️ 判定調整")
    ok_limit = st.slider("OK上限（全体）(%)", 0, 50, 20)

    st.divider()
    st.write("▼ 認識が悪い場合のみ調整")
    v_min = st.slider("ご飯の白さ (明度)", 0, 255, 170)
    s_max = st.slider("彩度上限", 0, 255, 70)

uploads = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploads:
    results = []
    previews = {}

    for up in uploads:
        try:
            file_bytes = np.frombuffer(up.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            b_mask, bbox = get_bento_mask_and_bbox(img)
            if b_mask is None or bbox is None:
                st.warning(f"容器検出に失敗: {up.name}")
                continue

            f_mask = get_food_mask(img, b_mask, v_min, s_max)

            # ★ 4分割固定テンプレで区画生成
            comps = build_compartments_fixed_4(b_mask, bbox)

            render, area_details = draw_results(img, comps, f_mask)

            # 全体空白率
            total_ratio = (np.count_nonzero(b_mask) - np.count_nonzero(f_mask)) / max(1, np.count_nonzero(b_mask)) * 100.0

            # エリア別表示（固定順）
            area_str = ", ".join([f"{name}:{int(round(r))}%" for name, r in area_details])

            results.append({
                "ファイル名": up.name,
                "全体空白率": f"{total_ratio:.1f}%",
                "エリア別": area_str,
                "判定": "OK" if total_ratio < ok_limit else "NG"
            })
            previews[up.name] = render

        except Exception as e:
            st.error(f"解析失敗 {up.name}: {e}")

    if results:
        df = pd.DataFrame(results)
        c_list, c_prev = st.columns([1, 1.2])

        with c_list:
            st.subheader("📋 判定一覧")
            table_event = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

        with c_prev:
            rows = table_event.get("selection", {}).get("rows", [])
            idx = rows[0] if rows else 0
            if idx < len(df):
                fname = df.iloc[idx]["ファイル名"]
                st.subheader(f"🔍 解析結果: {fname}")
                st.image(previews[fname], use_container_width=True)

