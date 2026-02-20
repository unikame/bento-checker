import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List

# =========================
# 画像処理（仕切り検出・エラー修正版）
# =========================

def largest_contour_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    """容器の外枠を検出してマスクを作成"""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100) 
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask

def food_mask_from_hsv(img_bgr, bento_mask, rice_v_min, rice_s_max) -> np.ndarray:
    """HSV空間を用いて食材部分のマスクを作成"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, color_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rice_mask = cv2.inRange(hsv, (0, 0, rice_v_min), (180, rice_s_max, 255))
    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    k = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k, iterations=1)
    return combined

def split_bento_compartments(img_bgr: np.ndarray, bento_mask: np.ndarray) -> List[np.ndarray]:
    """仕切りを検出し、領域を分割する"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    gray_adj = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(gray_adj, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    edges = cv2.Canny(gray_adj, 50, 150)
    boundary_map = cv2.bitwise_or(thresh, edges)
    boundary_map = cv2.bitwise_and(boundary_map, boundary_map, mask=bento_mask)
    
    dist_transform = cv2.distanceTransform(255 - boundary_map, cv2.DIST_L2, 5)
    _, markers_seeds = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    
    markers_seeds = np.uint8(markers_seeds)
    cnts, _ = cv2.findContours(markers_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    compartment_masks = []
    min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.005 
    
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        m = np.zeros_like(gray)
        cv2.drawContours(m, [c], -1, 255, -1)
        m = cv2.dilate(m, np.ones((21, 21), np.uint8), iterations=4)
        m = cv2.bitwise_and(m, bento_mask)
        compartment_masks.append(m)
        
    if len(compartment_masks) <= 1:
        return [bento_mask]
        
    return compartment_masks

def calc_detailed_ratios(img_bgr, comp_masks, food_mask):
    """各エリアの空白率を計算し描画（フォントエラー修正版）"""
    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = []
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 100, 255)]
    
    # フォントの種類を標準的なものに変更
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, m in enumerate(comp_masks):
        total_px = np.count_nonzero(m)
        if total_px == 0: continue
        
        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        empty_ratio = max(0, (total_px - food_px) / total_px * 100)
        
        cnt_list, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(annotated_rgb, cnt_list, -1, color, 12)
        
        M = cv2.moments(m)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            txt = f"{int(empty_ratio)}%"
            # 縁取り（黒）: thicknessを大きくして太く見せる
            cv2.putText(annotated_rgb, txt, (cx-70, cy+20), font, 3.0, (0,0,0), 15, cv2.LINE_AA)
            # 文字（白）
            cv2.putText(annotated_rgb, txt, (cx-70, cy+20), font, 3.0, (255,255,255), 4, cv2.LINE_AA)
        
        results.append(f"{int(empty_ratio)}%")
    return annotated_rgb, results

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

st.markdown("<style>div[data-testid='stSelectbox'], div[data-testid='stTextInput'] { display: none !important; }</style>", unsafe_allow_html=True)

head_c1, head_c2 = st.columns([1, 4])
with head_c1:
    try: st.image("header1_pc.png", width=180)
    except: st.write("### GLUG")
with head_c2:
    st.markdown("<h1 style='margin:0; padding-top:10px;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ パラメータ調整")
    ok_th = st.slider("OK判定 上限(%)", 0, 50, 20)
    st.divider()
    rice_v_min = st.slider("明度下限 (ご飯の白さ)", 0, 255, 160)
    rice_s_max = st.slider("彩度上限", 0, 255, 80)

uploads = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploads:
    results_list = []
    previews = {}
    
    for up in uploads:
        try:
            up.seek(0)
            file_bytes = np.frombuffer(up.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            b_mask = largest_contour_mask(gray)
            if b_mask is None: continue
            
            f_mask = food_mask_from_hsv(img, b_mask, rice_v_min, rice_s_max)
            comps = split_bento_compartments(img, b_mask)
            ann_img, area_txts = calc_detailed_ratios(img, comps, f_mask)
            
            total_b = np.count_nonzero(b_mask)
            total_f = np.count_nonzero(f_mask)
            total_ratio = (total_b - total_f) / total_b * 100
            
            results_list.append({
                "ファイル名": up.name,
                "全体空白率": f"{total_ratio:.1f}%",
                "エリア別詳細": ", ".join(area_txts),
                "判定": "OK" if total_ratio < ok_th else "NG"
            })
            previews[up.name] = ann_img
        except Exception as e:
            st.error(f"解析エラー ({up.name}): {e}")

    if results_list:
        df = pd.DataFrame(results_list)
        col_list, col_prev = st.columns([1, 1.3])
        
        with col_list:
            st.subheader("📋 判定一覧")
            event = st.dataframe(df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        
        with col_prev:
            selected_rows = event.get("selection", {}).get("rows", [])
            idx = selected_rows[0] if len(selected_rows) > 0 else 0
            
            if not df.empty and idx < len(df):
                fname = df.iloc[idx]["ファイル名"]
                st.subheader(f"🔍 解析プレビュー: {fname}")
                st.image(previews[fname], use_container_width=True)
