import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List

# =========================
# 画像処理（高精度・安定版）
# =========================

def get_bento_mask(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """容器の正確な外郭を抽出"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # ノイズ除去
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # 二値化（背景と容器を分離）
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 輪郭抽出
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Otsuでダメな場合はCannyで試行
        edges = cv2.Canny(blur, 30, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None

    # 最大の面積を持つ輪郭（容器）を選択
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    
    # マスクを少し内側に絞る（縁の誤検出防止）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask

def get_food_mask(img_bgr, bento_mask, v_min, s_max) -> np.ndarray:
    """食材部分を抽出"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 彩度（おかずの色）
    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 明度（ご飯の白さ）
    rice_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))
    
    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    return combined

def detect_compartments(img_bgr, bento_mask):
    """仕切りによるエリア分割"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 境界線の抽出（適応的二値化）
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    edges = cv2.bitwise_and(edges, edges, mask=bento_mask)
    
    # 距離変換で領域の芯を抽出
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    _, seeds = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
    
    seeds = np.uint8(seeds)
    cnts, _ = cv2.findContours(seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    masks = []
    min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.01
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        m = np.zeros_like(gray)
        cv2.drawContours(m, [c], -1, 255, -1)
        # 領域を拡大して境界まで埋める
        m = cv2.dilate(m, np.ones((21, 21), np.uint8), iterations=5)
        m = cv2.bitwise_and(m, bento_mask)
        masks.append(m)
        
    return masks if len(masks) > 0 else [bento_mask]

def draw_results(img_bgr, comp_masks, food_mask):
    """画像への枠線と数値の描画"""
    output = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_txts = []
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]
    
    for i, m in enumerate(comp_masks):
        area_px = np.count_nonzero(m)
        if area_px == 0: continue
        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        ratio = max(0, (area_px - food_px) / area_px * 100)
        
        # 枠線
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(output, cnts, -1, color, 12)
        
        # 数値
        M = cv2.moments(m)
        if M['m00'] > 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            txt = f"{int(ratio)}%"
            # 視認性向上のための太い縁取り
            cv2.putText(output, txt, (cx-70, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,0,0), 18, cv2.LINE_AA)
            cv2.putText(output, txt, (cx-70, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,255,255), 5, cv2.LINE_AA)
        
        res_txts.append(f"{int(ratio)}%")
    return output, res_txts

# =========================
# Streamlit メイン
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

# CSS: 画面デザイン調整
st.markdown("<style>div[data-testid='stSelectbox'], div[data-testid='stTextInput'] { display: none !important; }</style>", unsafe_allow_html=True)

# ヘッダー表示
h_c1, h_c2 = st.columns([1, 4])
with h_c1:
    try: st.image("header1_pc.png", width=180)
    except: st.write("### GLUG")
with h_c2:
    st.markdown("<h1 style='margin:0;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

# サイドバー設定
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
            # 画像読み込み
            file_bytes = np.frombuffer(up.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None: continue
            
            # 解析
            b_mask = get_bento_mask(img)
            if b_mask is None: continue
            
            f_mask = get_food_mask(img, b_mask, v_min, s_max)
            comps = detect_compartments(img, b_mask)
            render, area_details = draw_results(img, comps, f_mask)
            
            # 全体空白率
            total_ratio = (np.count_nonzero(b_mask) - np.count_nonzero(f_mask)) / np.count_nonzero(b_mask) * 100
            
            results.append({
                "ファイル名": up.name,
                "全体空白率": f"{total_ratio:.1f}%",
                "エリア別詳細": ", ".join(area_details),
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
            # 選択イベントの取得
            table_event = st.dataframe(df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        
        with c_prev:
            rows = table_event.get("selection", {}).get("rows", [])
            idx = rows[0] if rows else 0
            if idx < len(df):
                fname = df.iloc[idx]["ファイル名"]
                st.subheader(f"🔍 解析結果: {fname}")
                st.image(previews[fname], use_container_width=True)
