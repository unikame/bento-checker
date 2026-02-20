import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List

# =========================
# 容器検出（マスク + 外接矩形）
# =========================

def get_bento_mask_and_bbox(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
    """
    容器（トレー）の最大輪郭を狙ってマスク化し、外接矩形(bbox)も返す
    bbox = (x, y, w, h)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # エッジで容器の輪郭を拾う（Otsuより安定するケース多い）
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    # 最大輪郭を容器候補
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # 画像に対して小さすぎる輪郭しかない場合は失敗扱い
    if area < (h * w * 0.15):
        return None, None

    x, y, bw, bh = cv2.boundingRect(c)

    # マスク作成（輪郭の塗りつぶし）
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    # 縁・外側の誤検出を減らす（少し内側へ）
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)

    return mask, (x, y, bw, bh)


# =========================
# 食材マスク
# =========================
def get_food_mask(img_bgr: np.ndarray, bento_mask: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 彩度（おかず等）
    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 明度（ご飯の白さ）
    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined


# =========================
# 区切り（テンプレ方式：bbox基準で作る）
# =========================
def detect_compartments_template_by_bbox(
    bento_mask: np.ndarray,
    bbox: Tuple[int,int,int,int]
) -> List[Tuple[str, Tuple[int,int,int,int], np.ndarray]]:
    """
    bboxを基準に比率テンプレで4区画を作る
    return: [(name, (x0,y0,x1,y1), mask), ...]
    """
    x, y, w, h = bbox

    # 添付の弁当っぽい4枠テンプレ（必要ならここだけ調整）
    rects = [
        ("左上", 0.03, 0.06, 0.35, 0.50),
        ("右上", 0.36, 0.06, 0.97, 0.50),
        ("左下", 0.03, 0.52, 0.72, 0.95),
        ("右下", 0.74, 0.52, 0.97, 0.95),
    ]

    out = []
    for name, l, t, r, b in rects:
        x0 = int(x + l * w)
        y0 = int(y + t * h)
        x1 = int(x + r * w)
        y1 = int(y + b * h)

        m = np.zeros_like(bento_mask)
        cv2.rectangle(m, (x0, y0), (x1, y1), 255, thickness=-1)
        m = cv2.bitwise_and(m, bento_mask)  # 容器外に出ないようクリップ

        out.append((name, (x0, y0, x1, y1), m))

    return out


# =========================
# 描画＆計算（表示位置は「矩形中心」で固定）
# =========================
def draw_results(img_bgr: np.ndarray, comps, food_mask: np.ndarray):
    output = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_txts = []
    colors = [
        (0, 255, 0),     # 緑
        (255, 255, 255), # 白
        (255, 0, 0),     # 赤
        (255, 255, 0),   # 黄
        (255, 0, 255),
        (0, 165, 255),
    ]

    for i, (name, (x0, y0, x1, y1), m) in enumerate(comps):
        area_px = np.count_nonzero(m)
        if area_px == 0:
            continue

        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        ratio = max(0.0, (area_px - food_px) / area_px * 100.0)
        ratio_int = int(round(ratio))

        color = colors[i % len(colors)]
        cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness=12)

        # 表示位置は矩形中心（マスク形状に引っ張られない）
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        txt = f"{ratio_int}%"

        # 縁取り（黒）→ 本体（白）
        cv2.putText(output, txt, (cx - 70, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 18, cv2.LINE_AA)
        cv2.putText(output, txt, (cx - 70, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)

        res_txts.append(f"{name}:{ratio_int}%")

    return output, res_txts


# =========================
# Streamlit メイン
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

st.markdown("<h1 style='margin:0;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 判定調整")
    ok_limit = st.slider("OK上限 (%)", 0, 50, 20)
    st.divider()
    st.write("▼ 認識が悪い場合のみ調整")
    v_min = st.slider("ご飯の白さ (明度)", 0, 255, 170)
    s_max = st.slider("彩度上限", 0, 255, 70)

uploads = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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

            # ★ここが修正ポイント：bbox基準で区画生成
            comps = detect_compartments_template_by_bbox(b_mask, bbox)

            render, area_details = draw_results(img, comps, f_mask)

            total_ratio = (np.count_nonzero(b_mask) - np.count_nonzero(f_mask)) / max(1, np.count_nonzero(b_mask)) * 100

            results.append({
                "ファイル名": up.name,
                "全体空白率": f"{total_ratio:.1f}%",
                "エリア別": ", ".join(area_details),
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
