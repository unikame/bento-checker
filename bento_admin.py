import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# フォント探索（ここが今回の肝）
# =========================================================

def _find_font_path(preferred: str) -> str:
    """
    preferred が存在すればそれを使う。
    無ければOS別によくあるフォント候補から見つける。
    それも無ければ空文字を返す（→ load_default に落ちる）。
    """
    candidates = []

    if preferred:
        candidates.append(preferred)

    # Linux/Streamlit Cloudでありがち
    candidates += [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    # Windowsでありがち
    candidates += [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/yu Gothic.ttf",
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            pass

    return ""


def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """
    TrueTypeが読めるなら必ずそれを使う。
    読めない場合のみ default フォントにフォールバック。
    """
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    return ImageFont.load_default()

# =========================================================
# 1) 容器検出（マスク + 外接矩形）
# =========================================================

def get_bento_mask_and_bbox(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
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
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)

    return mask, (x, y, w, h)

# =========================================================
# 2) 食材マスク
# =========================================================

def get_food_mask(img_bgr: np.ndarray, bento_mask: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    kernel = np.ones((5, 5), np.uint8)
    rice_mask = cv2.morphologyEx(rice_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    rice_mask = cv2.medianBlur(rice_mask, 5)

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined

# =========================================================
# 3) 4分割（固定テンプレ）
# =========================================================

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
    x, y, w, h = bbox
    comps = []

    for name, l, t, r, b in TEMPLATE_FIXED_4:
        x0 = int(x + l * w)
        y0 = int(y + t * h)
        x1 = int(x + r * w)
        y1 = int(y + b * h)

        m = np.zeros_like(bento_mask)
        cv2.rectangle(m, (x0, y0), (x1, y1), 255, thickness=-1)
        m = cv2.bitwise_and(m, bento_mask)

        comps.append((name, (x0, y0, x1, y1), m))

    return comps

# =========================================================
# 4) 描画＆計算（PILでフォント描画）
# =========================================================

def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray, font_path: str):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_txts = []

    colors = [
        (0, 255, 0),
        (255, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    # 枠線
    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), colors[i % len(colors)], thickness=10)

    # テキスト（PIL）
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        area_px = np.count_nonzero(m)
        if area_px == 0:
            continue

        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        ratio = max(0.0, (area_px - food_px) / area_px * 100.0)
        ratio_int = int(round(ratio))
        txt = f"{ratio_int}%"

        # フォントサイズ（もっと大きく）
        box_size = min(max(1, x1 - x0), max(1, y1 - y0))
        font_size = int(box_size * 0.55)          # ← 0.36 → 0.55 に増量
        font_size = max(36, min(font_size, 220))  # 最小値も上げる

        font = _load_font(font_path, font_size)

        font_size = 80
        font = _load_font(font_path, font_size)

        try:
            tb = draw.textbbox((0, 0), txt, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        except Exception:
            text_w, text_h = draw.textsize(txt, font=font)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        tx = cx - text_w // 2
        ty = cy - text_h // 2

        outline = max(4, int(font_size * 0.10))
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), txt, font=font, fill=(0, 0, 0))

        draw.text((tx, ty), txt, font=font, fill=(255, 255, 255))
        res_txts.append((name, ratio))

    return np.array(pil_img), res_txts

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

    st.divider()
    st.write("▼ 表示フォント（PIL描画）")

    default_font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    user_font_path = st.text_input("フォントファイルパス", value=default_font_path)
    font_path = _find_font_path(user_font_path)

    if font_path:
        st.success(f"使用フォント: {font_path}")
    else:
        st.warning("TrueTypeフォントが見つかりません。文字が小さく見える可能性があります。")

    st.caption("例) Windows: C:/Windows/Fonts/meiryo.ttc")

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
            comps = build_compartments_fixed_4(b_mask, bbox)

            render, area_details = draw_results(img, comps, f_mask, font_path)

            total_ratio = (np.count_nonzero(b_mask) - np.count_nonzero(f_mask)) / max(1, np.count_nonzero(b_mask)) * 100.0
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

