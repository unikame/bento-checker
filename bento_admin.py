import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

# =========================
# 画像処理（仕切り解析追加版）
# =========================
def largest_contour_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask

def food_mask_from_hsv(img_bgr, bento_mask, rice_v_min, rice_s_max, **kwargs) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s_blur = cv2.GaussianBlur(s, (7, 7), 0)
    _, color_mask = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rice_mask = cv2.inRange(hsv, (0, 0, rice_v_min), (180, rice_s_max, 255))
    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)
    k = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    return combined

def split_bento_compartments(img_bgr: np.ndarray, bento_mask: np.ndarray) -> List[np.ndarray]:
    """容器内部の仕切りを検出し、各エリアのマスクリストを返す"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 容器の縁や仕切りの影を強調
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    thresh = cv2.bitwise_and(thresh, thresh, mask=bento_mask)
    
    # 領域を太らせて結合を切り離す（距離変換）
    dist_transform = cv2.distanceTransform(255-thresh, cv2.DIST_L2, 5)
    ret, markers = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    
    markers = np.uint8(markers)
    cnts, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    compartment_masks = []
    for c in cnts:
        if cv2.contourArea(c) < (img_bgr.shape[0] * img_bgr.shape[1] * 0.02): continue # 小さすぎる領域除外
        m = np.zeros_like(gray)
        cv2.drawContours(m, [c], -1, 255, -1)
        # 容器の形に合わせる
        m = cv2.dilate(m, np.ones((11,11), np.uint8), iterations=3)
        m = cv2.bitwise_and(m, bento_mask)
        compartment_masks.append(m)
    return compartment_masks

def calc_detailed_ratios(img_bgr, comp_masks, food_mask):
    """仕切りごとの空白率を計算し、画像に描画する"""
    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = []
    
    # 枠線の色リスト（赤, 青, 緑, 黄）
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
    
    for i, m in enumerate(comp_masks):
        total_px = np.count_nonzero(m)
        if total_px == 0: continue
        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        empty_ratio = max(0, (total_px - food_px) / total_px * 100)
        
        # 輪郭とテキストの描画
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(annotated_rgb, cnts, -1, color, 8)
        
        M = cv2.moments(m)
        if M['m00'] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            # 文字の視認性を上げるため縁取り
            cv2.putText(annotated_rgb, f"{int(empty_ratio)}%", (cx-40, cy), cv2.FONT_HERSHEY_BOLD, 2.0, (0,0,0), 10)
            cv2.putText(annotated_rgb, f"{int(empty_ratio)}%", (cx-40, cy), cv2.FONT_HERSHEY_BOLD, 2.0, (255,255,255), 3)
        
        results.append(round(empty_ratio, 1))
    return annotated_rgb, results

# =========================
# UI 設定
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 3.0rem !important; }
    .custom-card { background: white; border: 1px solid #ddd; border-radius: 12px; padding: 20px; margin-bottom: 1rem; }
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ヘッダー
col_head1, col_head2 = st.columns([1.5, 4])
with col_head1:
    try: st.image("header1_pc.png", width=190)
    except: st.write("### GLUG")
with col_head2:
    st.markdown("<h1 style='margin:0; padding-top:5px; font-size: 2.2rem;margin-left:-30px;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    ok_th = st.slider("OK 上限（%）", 0.0, 50.0, 20.0, 0.5)
    warn_th = st.slider("注意 上限（%）", 0.0, 80.0, 30.0, 0.5)
    st.divider()
    rice_v_min = st.slider("V（明度）下限", 0, 255, 180)
    rice_s_max = st.slider("S（彩度）上限", 0, 255, 60)

uploads = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not uploads:
    st.info("画像をアップロードすると解析が始まります。")
    st.stop()

# --- 解析フェーズ ---
results = []
preview_images = {}

for up in uploads:
    try:
        up.seek(0)
        file_bytes = np.frombuffer(up.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 1. 基本マスク作成
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bento_mask = largest_contour_mask(gray)
        food_mask = food_mask_from_hsv(img_bgr, bento_mask, rice_v_min, rice_s_max)
        
        # 2. 仕切り分割と個別集計
        comp_masks = split_bento_compartments(img_bgr, bento_mask)
        annotated_img, area_ratios = calc_detailed_ratios(img_bgr, comp_masks, food_mask)
        
        # 全体比率
        total_ratio = (np.count_nonzero(bento_mask) - np.count_nonzero(food_mask)) / np.count_nonzero(bento_mask) * 100
        
        results.append({
            "ファイル名": up.name, 
            "全体空白率": round(total_ratio, 1),
            "エリア別": str(area_ratios),
            "判定": "OK" if total_ratio < ok_th else "NG"
        })
        preview_images[up.name] = annotated_img
        
    except Exception as e:
        results.append({"ファイル名": up.name, "判定": "ERROR", "error": str(e)})

df = pd.DataFrame(results)

# --- 表示 ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 判定一覧")
    selection = st.dataframe(df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")

with col_right:
    selected_rows = selection.get("selection", {}).get("rows", [])
    selected_idx = selected_rows[0] if selected_rows else 0
    selected_file = df.iloc[selected_idx]["ファイル名"]
    
    st.subheader(f"🔍 解析プレビュー: {selected_file}")
    if selected_file in preview_images:
        st.image(preview_images[selected_file], use_container_width=True)
        st.write("※ 枠線ごとに計算された空白率が表示されています。")
