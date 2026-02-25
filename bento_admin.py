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
    candidates = []
    if preferred:
        candidates.append(preferred)

    candidates += [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

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

    # 縁の誤検出を減らすため少し内側に（ここも効きます）
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)

    return mask, (x, y, w, h)

# =========================================================
# 2) 食材マスク（共通ベース）
#    - おかず: 彩度Otsu
#    - ご飯: 白抽出（ただし、ご飯は後で"専用補正"する）
# =========================================================

def get_food_mask_base(img_bgr: np.ndarray, bento_mask: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 彩度で色物（おかず）を拾う
    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 明度で白物（ご飯）を拾う
    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    # 少し安定化
    kernel = np.ones((5, 5), np.uint8)
    rice_mask = cv2.morphologyEx(rice_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    rice_mask = cv2.medianBlur(rice_mask, 5)

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined

# =========================================================
# 2.5) ご飯エリア専用マスク（精度上げの本丸）
#    - 米粒の隙間(赤い底)を空白として数えない方向へ
#    - 強めの穴埋め + 少し膨張
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

    # 粒間の穴を埋める
    rice = cv2.morphologyEx(rice, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(close_iter)))

    # ちょい平滑化
    rice = cv2.medianBlur(rice, 5)

    # 面としてつなげる（空白の盛りを抑える）
    if dilate_iter > 0:
        rice = cv2.dilate(rice, np.ones((3, 3), np.uint8), iterations=int(dilate_iter))

    rice = cv2.bitwise_and(rice, rice, mask=mask_roi)
    return rice

# =========================================================
# 3) 4分割（固定テンプレ）
#    - margin_ratio を追加: 区画を内側に縮めて、仕切り/フチを計算対象から外す
# =========================================================

TEMPLATE_FIXED_4: List[Tuple[str, float, float, float, float]] = [
    ("左上", 0.04, 0.06, 0.34, 0.48),
    ("右上", 0.35, 0.06, 0.96, 0.48),
    ("左下", 0.04, 0.52, 0.70, 0.95),  # ここがご飯枠想定
    ("右下", 0.72, 0.52, 0.96, 0.95),
]

def build_compartments_fixed_4(
    bento_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.02,   # 追加: 区画を内側に縮める比率（2%推奨）
) -> List[Tuple[str, Tuple[int, int, int, int], np.ndarray]]:
    x, y, w, h = bbox

    comps = []
    for name, l, t, r, b in TEMPLATE_FIXED_4:
        x0 = int(x + l * w)
        y0 = int(y + t * h)
        x1 = int(x + r * w)
        y1 = int(y + b * h)

        # 内側マージン（仕切り/フチを除外）
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        x0m = x0 + mx
        y0m = y0 + my
        x1m = x1 - mx
        y1m = y1 - my

        # 変な逆転防止
        if x1m <= x0m + 5 or y1m <= y0m + 5:
            x0m, y0m, x1m, y1m = x0, y0, x1, y1

        m = np.zeros_like(bento_mask)
        cv2.rectangle(m, (x0m, y0m), (x1m, y1m), 255, thickness=-1)
        m = cv2.bitwise_and(m, bento_mask)

        comps.append((name, (x0m, y0m, x1m, y1m), m))

    return comps

# =========================================================
# 4) 描画＆計算
#    - comps で渡された各区画マスクに対して空白率を算出
# =========================================================

def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray, font_path: str):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    colors = [
        (0, 255, 0),
        (255, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), colors[i % len(colors)], thickness=8)

    base = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    res_txts = []

    font_size = 72
    font_main = _load_font(font_path, font_size)
    font_pct = _load_font(font_path, int(font_size * 0.55))

    bg_fill = (0, 0, 0, 130)
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

        tb_num = draw.textbbox((0, 0), num_txt, font=font_main)
        num_w = tb_num[2] - tb_num[0]
        num_h = tb_num[3] - tb_num[1]

        tb_pct = draw.textbbox((0, 0), pct_txt, font=font_pct)
        pct_w = tb_pct[2] - tb_pct[0]
        pct_h = tb_pct[3] - tb_pct[1]

        total_w = num_w + pct_w
        total_h = max(num_h, pct_h)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        label_x0 = cx - total_w // 2 - pad_x
        label_y0 = cy - total_h // 2 - pad_y
        label_x1 = cx + total_w // 2 + pad_x
        label_y1 = cy + total_h // 2 + pad_y
        draw.rectangle((label_x0, label_y0, label_x1, label_y1), fill=bg_fill)

        num_cx = cx - total_w // 2 + num_w // 2
        pct_cx = cx - total_w // 2 + num_w + pct_w // 2

        draw.text((num_cx, cy), num_txt, font=font_main, fill=(255, 255, 255, 255), anchor="mm")
        draw.text((pct_cx, cy), pct_txt, font=font_pct, fill=(255, 255, 255, 255), anchor="mm")

        res_txts.append((name, ratio))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(out), res_txts

# =========================================================
# 5) Streamlit UI
# =========================================================

st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")
st.markdown("<h1 style='margin:0;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)
st.caption("※この版は「赤い4分割容器」専用です（テンプレ固定 + ご飯エリア強化 + 仕切り除外）")

with st.sidebar:
    st.header("⚙️ 判定調整")
    ok_limit = st.slider("OK上限（全体）(%)", 0, 50, 20)

    st.divider()
    st.write("▼ 共通マスク（おかず + 白物）")
    v_min = st.slider("白っぽい判定 (明度) v_min", 0, 255, 170)
    s_max = st.slider("白っぽい判定 (彩度) s_max", 0, 255, 70)

    st.divider()
    st.write("▼ 仕切り/フチ除外（超効きます）")
    margin_ratio = st.slider("区画の内側マージン比率", 0.0, 0.06, 0.02, 0.005)

    st.divider()
    st.write("▼ ご飯エリア専用（精度の本丸）")
    rice_v_min = st.slider("ご飯 v_min（少し下げると抜けが減る）", 0, 255, 160)
    rice_s_max = st.slider("ご飯 s_max（高いほど白を拾いやすい）", 0, 255, 90)
    close_kernel = st.slider("ご飯 穴埋めカーネル（大きいほど粒間を埋める）", 3, 21, 13, 2)
    close_iter = st.slider("ご飯 穴埋め回数", 1, 6, 3)
    dilate_iter = st.slider("ご飯 つなぎ膨張（少しでOK）", 0, 4, 1)

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

# 左下を「ご飯枠」とみなす（テンプレ固定のため、ここを変えれば別枠にも対応可）
RICE_COMP_NAME = "左下"

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

            # 区画生成（仕切り/フチ除外）
            comps = build_compartments_fixed_4(b_mask, bbox, margin_ratio=margin_ratio)

            # まず共通ベース
            food_mask = get_food_mask_base(img, b_mask, v_min, s_max)

            # ご飯枠だけ強化マスクに差し替え（food_mask を上書き合成）
            rice_comp = None
            for (nm, rect, m) in comps:
                if nm == RICE_COMP_NAME:
                    rice_comp = (nm, rect, m)
                    break

            if rice_comp is not None:
                _, _, rice_m = rice_comp
                rice_strong = get_rice_mask_strong(
                    img_bgr=img,
                    mask_roi=rice_m,
                    rice_v_min=rice_v_min,
                    rice_s_max=rice_s_max,
                    close_kernel=close_kernel,
                    close_iter=close_iter,
                    dilate_iter=dilate_iter,
                )

                # rice枠の部分は "強化ご飯マスク" を優先
                food_mask = food_mask.copy()
                food_mask = cv2.bitwise_or(food_mask, rice_strong)

            # 描画（区画ごとの%は food_mask を使って算出）
            render, area_details = draw_results(img, comps, food_mask, font_path)

            # 全体空白率（※ b_mask ではなく、区画マスク合計で出すと仕切り除外が一貫する）
            comp_union = np.zeros_like(b_mask)
            for _, _, m in comps:
                comp_union = cv2.bitwise_or(comp_union, m)

            total_area = np.count_nonzero(comp_union)
            total_food = np.count_nonzero(cv2.bitwise_and(food_mask, comp_union))
            total_ratio = (total_area - total_food) / max(1, total_area) * 100.0

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
