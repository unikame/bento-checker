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
# 画像処理（仕切り解析・描画機能）
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
    """容器内部の仕切りを検出し、各エリアの個別マスクを作成"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 適応的二値化で仕切りの影や境界を抽出
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    thresh = cv2.bitwise_and(thresh, thresh, mask=bento_mask)
    
    # 距離変換を用いて各領域の中心を特定し、分離する
    dist_transform = cv2.distanceTransform(255 - thresh, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    
    markers = np.uint8(markers)
    cnts, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    compartment_masks = []
    # 画像サイズに対して一定以上の面積を持つ領域のみを採用
    min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.02
    
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        m = np.zeros_like(gray)
        cv2.drawContours(m, [c], -1, 255, -1)
        # 膨張させて仕切りギリギリまで領域を広げる
        m = cv2.dilate(m, np.ones((11, 11), np.uint8), iterations=3)
        m = cv2.bitwise_and(m, bento_mask)
        compartment_masks.append(m)
    return compartment_masks

def calc_detailed_ratios(img_bgr, comp_masks, food_mask):
    """各エリアの空白率を計算し、枠線と数値を画像に書き込む"""
    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = []
    
    # 枠線の色（緑, 白, 赤, 青の順でループ）
    colors = [(0, 255, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]
    
    for i, m in enumerate(comp_masks):
        total_px = np.count_nonzero(m)
        if total_px == 0: continue
        
        # エリア内の食材面積
        food_in_area = cv2.bitwise_and(food_mask, m)
        food_px = np.count_nonzero(food_in_area)
        
        # 空白率の算出
        empty_ratio = max(0, (total_px - food_px) / total_px * 100)
        
        # 枠線（輪郭）の描画
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(annotated_rgb, cnts, -1, color, 12)
        
        # 中心座標を計算してテキストを描画
        M = cv2.moments(m)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            label = f"{int(empty_ratio)}%"
            # 視認性向上のための縁取り
            cv2.putText(annotated_rgb, label, (cx - 50, cy), cv2.FONT_HERSHEY_BOLD, 2.5, (0, 0, 0), 12)
            cv2.putText(annotated_rgb, label, (cx - 50, cy), cv2.FONT_HERSHEY_BOLD, 2.5, (255, 255, 255), 4)
        
        results.append(f"{int(empty_ratio)}%")
        
    return annotated_rgb, results

# =========================
# Streamlit UI 設定
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; }
    .custom-card { background: white; border: 1px solid #ddd; border-radius: 12px; padding: 20px; }
    /* 不要なUI要素の非表示 */
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ヘッダーエリア
col_head1, col_head2 = st.columns([1, 4])
with col_head1:
    try: st.image("header1_pc.png", width=180)
    except: st.write("### GLUG")
with col_head2:
    st.markdown("<h1 style='margin:0; padding-top:10px;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

st.caption("画像をアップロードすると、仕切りごとに解析を行い、空白率を表示します。")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 判定閾値")
    ok_th = st.slider("OK 上限（%）", 0.0, 50.0, 20.0)
    st.divider()
    st.header("🔍 解析パラメータ")
    rice_v_min = st.slider("明度(V)下限", 0, 255, 180)
    rice_s_max = st.slider("彩度(S)上限", 0, 255, 60)

# ファイルアップローダー
uploads = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not uploads:
    st.info("画像をアップロードすると解析が始まります。")
    st.stop()

# --- メイン解析処理 ---
results_data = []
previews = {}

for up in uploads:
    try:
        up.seek(0)
        file_bytes = np.frombuffer(up.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 1. 容器全体と食材のマスク作成
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bento_mask = largest_contour_mask(gray)
        if bento_mask is None: raise ValueError("容器を検出できませんでした")
        
        food_mask = food_mask_from_hsv(img_bgr, bento_mask, rice_v_min, rice_s_max)
        
        # 2. 仕切りごとの分割と詳細解析
        compartments = split_bento_compartments(img_bgr, bento_mask)
        annotated_img, area_results = calc_detailed_ratios(img_bgr, compartments, food_mask)
        
        # 3. 全体統計の算出
        total_bento = np.count_nonzero(bento_mask)
        total_food = np.count_nonzero(food_mask)
        total_empty_ratio = (total_bento - total_food) / total_bento * 100
        
        results_data.append({
            "ファイル名": up.name,
            "全体空白率": f"{total_empty_ratio:.1f}%",
            "エリア別詳細": ", ".join(area_results),
            "判定": "OK" if total_empty_ratio < ok_th else "NG"
        })
        previews[up.name] = annotated_img

    except Exception as e:
        results_data.append({"ファイル名": up.name, "判定": "ERROR", "error": str(e)})

# --- 画面表示 ---
df = pd.DataFrame(results_data)
col_l, col_r = st.columns([1, 1.2])

with col_l:
    st.subheader("📋 判定一覧")
    selection = st.dataframe(
        df[["ファイル名", "全体空白率", "エリア別詳細", "判定"]], 
        use_container_width=True, 
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

with col_r:
    # 選択された行の画像を表示（デフォルトは1枚目）
    selected_rows = selection.get("selection", {}).get("rows", [])
    idx = selected_rows[0] if selected_rows else 0
    if not df.empty:
        fname = df.iloc[idx]["ファイル名"]
        
        st.subheader(f"🔍 解析プレビュー: {fname}")
        if fname in previews:
            st.image(previews[fname], use_container_width=True)
            st.info("※ 枠線で囲まれた各エリア内の数値を計算しています。")
