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
# 画像処理（仕切り検出の安定化版）
# =========================

def largest_contour_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    """容器の外枠を検出してマスクを作成"""
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

def food_mask_from_hsv(img_bgr, bento_mask, rice_v_min, rice_s_max) -> np.ndarray:
    """HSV空間を用いて食材部分のマスクを作成"""
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
    """仕切りを検出し、領域を分割する（コントラスト強調版）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # コントラストを強調して仕切りの影を拾いやすくする
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 適応的二値化（近傍サイズを広げて大きな仕切りに対応）
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    thresh = cv2.bitwise_and(thresh, thresh, mask=bento_mask)
    
    # 距離変換による領域分離
    dist_transform = cv2.distanceTransform(255 - thresh, cv2.DIST_L2, 5)
    # 分割のしきい値を調整
    _, markers = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    
    markers = np.uint8(markers)
    cnts, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    compartment_masks = []
    min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.01 
    
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        m = np.zeros_like(gray)
        cv2.drawContours(m, [c], -1, 255, -1)
        # 枠線ギリギリまで拡張
        m = cv2.dilate(m, np.ones((15, 15), np.uint8), iterations=3)
        m = cv2.bitwise_and(m, bento_mask)
        compartment_masks.append(m)
        
    if not compartment_masks:
        compartment_masks.append(bento_mask)
        
    return compartment_masks

def calc_detailed_ratios(img_bgr, comp_masks, food_mask):
    """詳細比率を計算し画像に描画"""
    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = []
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
    
    for i, m in enumerate(comp_masks):
        total_px = np.count_nonzero(m)
        if total_px == 0: continue
        food_px = np.count_nonzero(cv2.bitwise_and(food_mask, m))
        empty_ratio = max(0, (total_px - food_px) / total_px * 100)
        
        # 枠線の描画
        cnt_list, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(annotated_rgb, cnt_list, -1, color, 12)
        
        # 中心座標の取得とテキスト描画
        M = cv2.moments(m)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            txt = f"{int(empty_ratio)}%"
            # 太めの縁取りで視認性確保
            cv2.putText(annotated_rgb, txt, (cx-60, cy), cv2.FONT_HERSHEY_BOLD, 3.0, (0,0,0), 18)
            cv2.putText(annotated_rgb, txt, (cx-60, cy), cv2.FONT_HERSHEY_BOLD, 3.0, (255,255,255), 6)
        
        results.append(f"{int(empty_ratio)}%")
    return annotated_rgb, results

# =========================
# UI 設定
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.0rem !important; }
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ヘッダー
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    try: st.image("header1_pc.png", width=180)
    except: st.write("### GLUG")
with col_h2:
    st.markdown("<h1 style='margin:0; padding-top:10px;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 設定")
    ok_th = st.slider("OK 上限（%）", 0.0, 50.0, 20.0)
    rice_v_min = st.slider("明度下限", 0, 255, 180)
    rice_s_max = st.slider("彩度上限", 0, 255, 60)

uploads = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploads:
    results_data = []
    previews = {}
    
    for up in uploads:
        try:
            up.seek(0)
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bento_m = largest_contour_mask(gray)
            if bento_m is None: continue
            
            food_m = food_mask_from_hsv(img, bento_m, rice_v_min, rice_s_max)
            comps = split_bento_compartments(img, bento_m)
            ann_img, area_res = calc_detailed_ratios(img, comps, food_m)
            
            total_ratio = (np.count_nonzero(bento_m) - np.count_nonzero(food_m)) / np.count_nonzero(bento_m) * 100
            
            results_data.append({
                "ファイル名": up.name,
                "全体空白率": f"{total_ratio:.1f}%",
                "エリア別詳細": ", ".join(area_res),
                "判定": "OK" if total_ratio < ok_th else "NG"
            })
            previews[up.name] = ann_img
        except Exception as e:
            st.error(f"解析エラー ({up.name}): {e}")

    if results_data:
        df = pd.DataFrame(results_data)
        cl, cr = st.columns([1, 1.2])
        
        with cl:
            st.subheader("📋 判定一覧")
            sel = st.dataframe(df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        
        with cr:
            # --- IndexError 対策の修正箇所 ---
            selected_rows = sel.get("selection", {}).get("rows", [])
            # 行が選択されていない場合は 0 番目（一番上）を表示、選択されていればその行を表示
            idx = selected_rows[0] if len(selected_rows) > 0 else 0
            
            if not df.empty and idx < len(df):
                fn = df.iloc[idx]["ファイル名"]
                st.subheader(f"🔍 解析: {fn}")
                st.image(previews[fn], use_container_width=True)
                st.info("※ 枠線ごとに計算された空白率が表示されています。数字が出ない場合はサイドバーの設定を調整してください。")
