import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List

# =========================================================
# 枠ごとの判定ルール
# baseline_fill：基準充填率
# allow_shortage：許容不足率（基準から何%減までOKか）
# =========================================================
COMP_RULES = {
    "左上": {"baseline_fill": 0.75, "allow_shortage": 0.12},
    "右上": {"baseline_fill": 0.85, "allow_shortage": 0.10},
    "左下": {"baseline_fill": 0.92, "allow_shortage": 0.06},  # ご飯は厳しめ
    "右下": {"baseline_fill": 0.75, "allow_shortage": 0.12},
}

# =========================================================
# 1) 容器検出（マスク + 外接矩形）
# =========================================================
def get_bento_mask_and_bbox(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
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
    if area < (H * W * 0.12):
        return None, None

    x, y, w, h = cv2.boundingRect(c)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    # フチ誤検出を減らす
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)

    return mask, (x, y, w, h)

# =========================================================
# 2) 食材マスク（簡易：彩度Otsu + 白抽出）
# =========================================================
def get_food_mask_base(img_bgr: np.ndarray, bento_mask: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # おかず（色）
    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ご飯（白）
    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))
    kernel = np.ones((5, 5), np.uint8)
    rice_mask = cv2.morphologyEx(rice_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    rice_mask = cv2.medianBlur(rice_mask, 5)

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined

# =========================================================
# 2.5) ご飯エリア専用：穴埋め強化
# =========================================================
def get_rice_mask_strong(
    img_bgr: np.ndarray,
    mask_roi: np.ndarray,
    rice_v_min: int,
    rice_s_max: int,
    close_kernel: int,
    close_iter: int,
    dilate_iter: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    rice = cv2.inRange(hsv, (0, 0, rice_v_min), (180, rice_s_max, 255))

    k = max(3, int(close_kernel))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)

    rice = cv2.morphologyEx(rice, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(close_iter)))
    rice = cv2.medianBlur(rice, 5)

    if dilate_iter > 0:
        rice = cv2.dilate(rice, np.ones((3, 3), np.uint8), iterations=int(dilate_iter))

    rice = cv2.bitwise_and(rice, rice, mask=mask_roi)
    return rice

# =========================================================
# 3) 4分割テンプレ（この容器用）
# =========================================================
TEMPLATE_FIXED_4: List[Tuple[str, float, float, float, float]] = [
    ("左上", 0.04, 0.06, 0.34, 0.48),
    ("右上", 0.35, 0.06, 0.96, 0.48),
    ("左下", 0.04, 0.52, 0.70, 0.95),  # ご飯枠
    ("右下", 0.72, 0.52, 0.96, 0.95),
]

def build_compartments_fixed_4(
    bento_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.02,
):
    x, y, w, h = bbox
    comps = []

    for name, l, t, r, b in TEMPLATE_FIXED_4:
        x0 = int(x + l * w)
        y0 = int(y + t * h)
        x1 = int(x + r * w)
        y1 = int(y + b * h)

        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        x0m, y0m, x1m, y1m = x0 + mx, y0 + my, x1 - mx, y1 - my

        if x1m <= x0m + 5 or y1m <= y0m + 5:
            x0m, y0m, x1m, y1m = x0, y0, x1, y1

        m = np.zeros_like(bento_mask)
        cv2.rectangle(m, (x0m, y0m), (x1m, y1m), 255, thickness=-1)
        m = cv2.bitwise_and(m, bento_mask)

        comps.append((name, (x0m, y0m, x1m, y1m), m))

    return comps

# =========================================================
# 4) 表示（cv2で確実に描く）
#    - 枠別：スカスカ% + OK/NG
# =========================================================
def _draw_label_cv2(img_rgb, x, y, text, is_ng: bool):
    # 背景ボックス + 文字（フォント依存なし）
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thick = 3

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 10
    x0, y0 = x - tw // 2 - pad, y - th // 2 - pad
    x1, y1 = x + tw // 2 + pad, y + th // 2 + pad

    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img_rgb.shape[1]-1, x1); y1 = min(img_rgb.shape[0]-1, y1)

    cv2.rectangle(img_rgb, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    color = (255, 0, 0) if is_ng else (255, 255, 255)  # RGB
    cv2.putText(img_rgb, text, (x - tw // 2, y + th // 2),
                font, scale, color, thick, cv2.LINE_AA)

def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 枠線（見やすく）
    outline_colors = [
        (0, 255, 0),
        (255, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    results = []

    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        # 枠線
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), outline_colors[i % len(outline_colors)], thickness=6)

        area_px = np.count_nonzero(m)
        if area_px == 0:
            continue

        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        fill = food_px / area_px

        rule = COMP_RULES.get(name, {"baseline_fill": 0.85, "allow_shortage": 0.10})
        baseline = rule["baseline_fill"]
        allow = rule["allow_shortage"]

        shortage = max(0.0, (baseline - fill) / baseline)
        sukaska_pct = int(round(shortage * 100))
        judge = "OK" if shortage <= allow else "NG"

        # 中央にラベル
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        _draw_label_cv2(img_rgb, cx, cy, f"{sukaska_pct}% {judge}", is_ng=(judge == "NG"))

        results.append({
            "name": name,
            "fill": fill,
            "shortage": shortage,
            "judge": judge,
        })

    return img_rgb, results

# =========================================================
# 5) Streamlit UI
# =========================================================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")
st.markdown("<h1 style='margin:0;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)
st.caption("枠ごとにスカスカ%とOK/NGを出し、1枠でもNGなら全体NG")

with st.sidebar:
    st.header("⚙️ 判定調整（食材マスク）")
    v_min = st.slider("白っぽい判定 (明度) v_min", 0, 255, 170)
    s_max = st.slider("白っぽい判定 (彩度) s_max", 0, 255, 70)

    st.divider()
    st.write("▼ 仕切り/フチ除外（効きます）")
    margin_ratio = st.slider("区画の内側マージン比率", 0.0, 0.06, 0.02, 0.005)

    st.divider()
    st.write("▼ ご飯エリア専用（穴埋め強化）")
    rice_v_min = st.slider("ご飯 v_min", 0, 255, 160)
    rice_s_max = st.slider("ご飯 s_max", 0, 255, 90)
    close_kernel = st.slider("ご飯 穴埋めカーネル", 3, 21, 13, 2)
    close_iter = st.slider("ご飯 穴埋め回数", 1, 6, 3)
    dilate_iter = st.slider("ご飯 つなぎ膨張", 0, 4, 1)

uploads = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

RICE_COMP_NAME = "左下"

if uploads:
    results = []
    previews = {}

    for up in uploads:
        try:
            file_bytes = np.frombuffer(up.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            b_mask, bbox = get_bento_mask_and_bbox(img)
            if b_mask is None or bbox is None:
                st.warning(f"容器検出に失敗: {up.name}")
                continue

            comps = build_compartments_fixed_4(b_mask, bbox, margin_ratio=margin_ratio)

            # 共通ベース
            food_mask = get_food_mask_base(img, b_mask, v_min, s_max)

            # ご飯枠だけ強化
            rice_m = None
            for (nm, rect, m) in comps:
                if nm == RICE_COMP_NAME:
                    rice_m = m
                    break

            if rice_m is not None:
                rice_strong = get_rice_mask_strong(
                    img_bgr=img,
                    mask_roi=rice_m,
                    rice_v_min=rice_v_min,
                    rice_s_max=rice_s_max,
                    close_kernel=close_kernel,
                    close_iter=close_iter,
                    dilate_iter=dilate_iter,
                )
                food_mask = cv2.bitwise_or(food_mask, rice_strong)

            # 描画＆判定
            render_rgb, area_details = draw_results(img, comps, food_mask)

            # 全体判定（1枠でもNGならNG）
            overall = "NG" if any(d["judge"] == "NG" for d in area_details) else "OK"

            area_str = ", ".join([
                f'{d["name"]}:{int(round(d["shortage"]*100))}%({d["judge"]})'
                for d in area_details
            ])

            results.append({
                "ファイル名": up.name,
                "全体判定": overall,
                "エリア別": area_str,
            })
            previews[up.name] = render_rgb

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
