import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List, Dict

# ② import追加（フォント変更はPILで描画）
from PIL import Image, ImageDraw, ImageFont

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
# 4) 描画＆計算（PILでフォント描画：文字サイズ大きめ）
# =========================================================

def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """
    フォントロード（失敗時はデフォルトフォントにフォールバック）
    """
    try:
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return ImageFont.load_default()

def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray, font_path: str):
    """
    - 枠線はOpenCVで描画
    - 数字はPILで描画（好きなフォントに変更可能）
    - フォントサイズは枠サイズに応じて自動（やや大きめ）
    """
    # OpenCV→RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    res_txts = []

    colors = [
        (0, 255, 0),     # 緑
        (255, 255, 255), # 白
        (255, 0, 0),     # 赤
        (255, 255, 0),   # 黄
    ]

    # まず枠線はOpenCVで描く（img_rgbに描画）
    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        color = colors[i % len(colors)]
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), color, thickness=10)

    # PILに変換してテキスト描画
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

        # --- フォントサイズ自動調整（枠サイズ基準） ---
        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)
        box_size = min(box_w, box_h)

        # ✅ 文字を大きくする設定（0.36推奨）
        font_size = int(box_size * 0.36)
        font_size = max(26, min(font_size, 180))
        font = _load_font(font_path, font_size)

        # テキストサイズ取得（Pillowの新旧に対応）
        try:
            tb = draw.textbbox((0, 0), txt, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        except Exception:
            text_w, text_h = draw.textsize(txt, font=font)

        # 中央配置
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        tx = cx - text_w // 2
        ty = cy - text_h // 2

        # ✅ 縁取り（太さも少し強く）
        outline = max(3, int(font_size * 0.10))
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), txt, font=font, fill=(0, 0, 0))

        # 本体（白）
        draw.text((tx, ty), txt, font=font, fill=(255, 255, 255))

        res_txts.append((name, ratio))

    output = np.array(pil_img)  # PIL→numpy(RGB)
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

    st.divider()
    st.write("▼ 表示フォント（PIL描画）")

    # Streamlit Cloud/Linux想定: Noto Sans CJK が入っていることが多い
    # Windowsローカルなら "C:/Windows/Fonts/meiryo.ttc" などに変更してください
    default_font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    font_path = st.text_input("フォントファイルパス", value=default_font_path)
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

            # ★ 4分割固定テンプレで区画生成
            comps = build_compartments_fixed_4(b_mask, bbox)

            # ★ draw_results（PILでフォント描画）
            render, area_details = draw_results(img, comps, f_mask, font_path)

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
