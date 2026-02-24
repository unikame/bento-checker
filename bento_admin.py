import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List

from PIL import Image, ImageDraw, ImageFont

# =========================================================
# フォント探索 & ロード
# =========================================================

def _find_font_path(preferred: str) -> str:
    """
    preferred が存在すればそれを使う。
    無ければOS別によくあるフォント候補から見つける。
    """
    candidates = []
    if preferred:
        candidates.append(preferred)

    # Linux / Streamlit Cloud
    candidates += [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    # Windows
    candidates += [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
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
    TrueTypeが読めるならそれを使う。ダメならデフォルト。
    ※ デフォルトフォントはサイズ変更が効きづらいので、font_path はなるべく通す
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

def get_bento_mask_and_bbox(
    img_bgr: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
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
# 4) 描画＆計算（スタイリッシュ版）
#   - 白線なし
#   - 角丸なし（四角）
#   - 数字と%の上下ズレ解消
#   - %だけ小さく
# =========================================================

def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray, font_path: str):
    """
    - 枠線: OpenCV
    - ラベル: PIL (RGBA overlay)
    - 背景: 四角、白線なし
    - 文字: 数字と%を別フォントで中央揃え（mm） → ズレに強い
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    colors = [
        (0, 255, 0),      # 緑
        (255, 255, 255),  # 白
        (255, 0, 0),      # 赤
        (255, 255, 0),    # 黄
    ]

    # 枠線（太さ調整）
    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), colors[i % len(colors)], thickness=8)

    base = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    res_txts = []

    # フォントサイズ固定（統一）
    font_size = 72
    font_main = _load_font(font_path, font_size)
    font_pct = _load_font(font_path, int(font_size * 0.55))  # %だけ小さく

    # ラベル背景（四角）
    bg_fill = (0, 0, 0, 130)

    # 余白（上が詰まって見えるのでpad_yを厚めに）
    pad_x = int(font_size * 0.35)
    pad_y = int(font_size * 0.30)

    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        area_px = np.count_nonzero(m)
        if area_px == 0:
            continue

        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        ratio = max(0.0, (area_px - food_px) / area_px * 100.0)
        ratio_int = int(round(ratio))

        num_txt = str(ratio_int)
        pct_txt = "%"

        # サイズ取得（bbox）
        tb_num = draw.textbbox((0, 0), num_txt, font=font_main)
        num_w = tb_num[2] - tb_num[0]
        num_h = tb_num[3] - tb_num[1]

        tb_pct = draw.textbbox((0, 0), pct_txt, font=font_pct)
        pct_w = tb_pct[2] - tb_pct[0]
        pct_h = tb_pct[3] - tb_pct[1]

        total_w = num_w + pct_w
        total_h = max(num_h, pct_h)

        # 枠の中心
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        # 背景（四角）：全体を中央に配置
        label_x0 = cx - total_w // 2 - pad_x
        label_y0 = cy - total_h // 2 - pad_y
        label_x1 = cx + total_w // 2 + pad_x
        label_y1 = cy + total_h // 2 + pad_y
        draw.rectangle((label_x0, label_y0, label_x1, label_y1), fill=bg_fill)

        # 文字は中央アンカーで配置（mm）
        num_cx = cx - total_w // 2 + num_w // 2
        pct_cx = cx - total_w // 2 + num_w + pct_w // 2

        draw.text(
            (num_cx, cy),
            num_txt,
            font=font_main,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )

        draw.text(
            (pct_cx, cy),
            pct_txt,
            font=font_pct,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )

        res_txts.append((name, ratio))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(out), res_txts

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
        st.warning("TrueTypeフォントが見つかりません。デフォルトフォントで表示します（サイズが効きにくい場合があります）")

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

            # 4分割固定テンプレで区画生成
            comps = build_compartments_fixed_4(b_mask, bbox)

            # 描画
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
