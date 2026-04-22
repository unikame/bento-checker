import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import anthropic
import base64
import json
import io
import re

# --- 1. APIキーの設定 ---
ANTHROPIC_API_KEY = "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"

st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 履歴・補正ロジック ---
def load_shared_history():
    if os.path.exists(DB_FILE):
        try: return pd.read_csv(DB_FILE).to_dict('records')
        except: return []
    return []

def normalize_image(img_bgr):
    res = img_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(res[:,:,0]), np.mean(res[:,:,1]), np.mean(res[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    res[:,:,0] *= (avg_gray / (avg_b + 1e-6))
    res[:,:,1] *= (avg_gray / (avg_g + 1e-6))
    res[:,:,2] *= (avg_gray / (avg_r + 1e-6))
    res = np.clip(res, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b_chan)), cv2.COLOR_LAB2BGR)

# --- 3. スタイル (UI修正) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f0ede8 0%, #e8e4de 100%); }
.title-block { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1a1a1a; margin-bottom: 20px; }
.metric-card { background: white; border-radius: 15px; padding: 15px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); margin-bottom: 10px; border-left: 5px solid #ccc; }
.metric-card.pass { border-left-color: #2ecc71; }
.metric-card.fail { border-left-color: #e74c3c; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #1a1a1a; }
.status-badge { display: inline-block; padding: 6px 20px; border-radius: 999px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.2rem; }
.status-badge.pass { background: #d4f5e2; color: #1a8a4a; }
.status-badge.fail { background: #fde8e8; color: #c0392b; }

/* サイドバーの文字色・デザイン修正 */
[data-testid="stSidebar"] { background: #1a1a1a !important; }
[data-testid="stSidebar"] * { color: white !important; } /* 全て白文字に */
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.9rem; font-weight: 600; color: white !important; }
[data-testid="stSidebar"] .stButton button { background-color: #333 !important; border: 1px solid #444 !important; color: white !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 4. メイン処理 ---
st.markdown('<div class="title-block">🍱 Bento Checker Pro</div>', unsafe_allow_html=True)

@st.cache_resource
def get_anthropic_client():
    if "sk-ant" not in ANTHROPIC_API_KEY: return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

client = get_anthropic_client()
if 'selected_idx' not in st.session_state: st.session_state.selected_idx = None

# サイドバー
with st.sidebar:
    if st.button("＋ 新規スキャン", use_container_width=True):
        st.session_state.selected_idx = None
        st.rerun()
    st.markdown("---")
    st.markdown("📷 **空容器登録(精度向上)**") # ラベルを白文字かつ強調
    up_ref = st.file_uploader("写真をアップロード", type=['jpg', 'png', 'jpeg'], key="ref_up", label_visibility="collapsed")
    if up_ref:
        Image.open(up_ref).convert("RGB").save(REFERENCE_FILE)
        st.success("容器を学習しました")
    st.markdown("📏 **色許容度**")
    tolerance = st.slider("tolerance", 10.0, 50.0, 25.0, label_visibility="collapsed")

# 履歴読み込み
history = load_shared_history()

if st.session_state.selected_idx is None:
    up = st.file_uploader("お弁当の写真をアップロードしてください", type=['jpg', 'png', 'jpeg'])
    if up:
        if st.button("🚀 解析を開始する", type="primary", use_container_width=True):
            with st.spinner("解析中..."):
                img_orig = Image.open(up).convert("RGB")
                img_np = np.array(img_orig)
                h, w = img_np.shape[:2]
                area_defs = {
                    "右上（メイン）": (int(h*0.08), int(h*0.48), int(w*0.48), int(w*0.95)),
                    "左上（副菜）": (int(h*0.08), int(h*0.48), int(w*0.05), int(w*0.45)),
                    "右下（副菜）": (int(h*0.5), int(h*0.95), int(w*0.52), int(w*0.95))
                }
                ref_stats = None
                if os.path.exists(REFERENCE_FILE):
                    ref_b = cv2.imread(REFERENCE_FILE)
                    if ref_b is not None:
                        ref_lab = cv2.cvtColor(normalize_image(ref_b), cv2.COLOR_BGR2LAB)
                        ref_stats = {"mean": np.mean(ref_lab, axis=(0,1)).tolist()}

                results, draw_np = [], img_np.copy()
                for name, (y1, y2, x1, x2) in area_defs.items():
                    roi = img_orig.crop((x1, y1, x2, y2))
                    # --- 解析実行 ---
                    roi_bgr = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2BGR)
                    roi_norm = normalize_image(roi_bgr)
                    roi_lab = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2LAB).astype(np.float32)
                    if ref_stats:
                        dist = np.sqrt(np.sum((roi_lab - np.array(ref_stats["mean"]))**2 * [0.6, 1.2, 1.2], axis=2))
                        mask = (dist < tolerance).astype(np.uint8) * 255
                    else:
                        hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
                        mask = cv2.bitwise_or(cv2.inRange(hsv, (0,70,40), (15,255,255)), cv2.inRange(hsv, (165,70,40), (180,255,255)))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
                    pct = round((np.sum(mask > 0) / mask.size) * 100.0, 1)
                    results.append(f"{name}@{pct}")
                    t_roi = draw_np[y1:y1+mask.shape[0], x1:x1+mask.shape[1]]
                    t_roi[mask > 0] = t_roi[mask > 0] * 0.5 + np.array([255, 230, 0]) * 0.5
                
                path = f"{SAVE_DIR}/res_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                Image.fromarray(draw_np).save(path)
                avg_v = np.mean([float(r.split("@")[1]) for r in results])
                new_rec = {"time": datetime.now().strftime("%m/%d %H:%M"), "status": "PASS" if avg_v < 15 else "FAIL", "img_path": path, "avg_emptiness": round(avg_v, 1), "detail_text": " / ".join(results)}
                pd.DataFrame([new_rec]).to_csv(DB_FILE, mode='a', header=not os.path.exists(DB_FILE), index=False)
                st.session_state.selected_idx = len(load_shared_history()) - 1
                st.rerun()
else:
    # 履歴詳細画面
    if st.session_state.selected_idx < len(history):
        data = history[st.session_state.selected_idx]
        c1, c2 = st.columns([1.3, 0.7])
        with c1: st.image(data['img_path'], use_container_width=True)
        with c2:
            st.markdown(f'<div class="status-badge {data["status"].lower()}">{data["status"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><small>平均空き率</small><div class="metric-value">{data["avg_emptiness"]}%</div></div>', unsafe_allow_html=True)
            for part in data['detail_text'].split(" / "):
                if "@" in part:
                    nm, pct = part.split("@")
                    st.markdown(f'<div class="metric-card"><small>{nm}</small><div class="metric-value">{pct}%</div></div>', unsafe_allow_html=True)
            if st.button("← 戻る", use_container_width=True):
                st.session_state.selected_idx = None
                st.rerun()
