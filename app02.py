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

# --- 1. APIキーの設定 (新しく作成したキーをここに貼り付け) ---
ANTHROPIC_API_KEY = "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"

st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. スタイル (バグ修正版) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a1a1a; }
.stApp { background: linear-gradient(135deg, #f0ede8 0%, #e8e4de 100%); }
.title-block { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1a1a1a; margin-bottom: 20px; }
.metric-card { background: white; border-radius: 15px; padding: 15px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); margin-bottom: 10px; border-left: 5px solid #ccc; }
.metric-card.pass { border-left-color: #2ecc71; }
.metric-card.fail { border-left-color: #e74c3c; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #1a1a1a; }
.status-badge { display: inline-block; padding: 6px 20px; border-radius: 999px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.2rem; }
.status-badge.pass { background: #d4f5e2; color: #1a8a4a; }
.status-badge.fail { background: #fde8e8; color: #c0392b; }
/* 左メニューのボタンと文字色の修正 */
[data-testid="stSidebar"] { background: #1a1a1a !important; color: #f0ede8 !important; }
[data-testid="stSidebar"] .stButton button { background-color: #333 !important; color: white !important; border: 1px solid #444 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #f0ede8 !important; }
[data-testid="stSidebar"] .stSlider label { color: #f0ede8 !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. 画像補正エンジン ---
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
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b_chan)), cv2.COLOR_LAB2BGR)

# --- 4. 解析ロジック ---
@st.cache_resource
def get_anthropic_client():
    if "sk-ant" not in ANTHROPIC_API_KEY:
        st.error("有効な Anthropic API キーを入力してください。")
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def compute_emptiness_hybrid(client, roi_pil, area_name, ref_stats, tolerance):
    roi_bgr = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    roi_norm = normalize_image(roi_bgr)
    roi_lab = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2LAB).astype(np.float32)

    if ref_stats:
        mean = np.array(ref_stats["mean"], dtype=np.float32)
        dist = np.sqrt(np.sum((roi_lab - mean)**2 * [0.6, 1.2, 1.2], axis=2))
        mask = (dist < tolerance).astype(np.uint8) * 255
    else:
        hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, (0,70,40), (15,255,255)), cv2.inRange(hsv, (165,70,40), (180,255,255)))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cv_pct = round((np.sum(mask > 0) / mask.size) * 100.0, 1)

    ai_pct, reason = cv_pct, "CV判定"
    if client:
        try:
            b64 = pil_to_base64(roi_pil)
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},{"type":"text","text":"Return JSON ONLY: {\"pct\": number, \"reason\": \"string\"}"}]}]
            )
            data = json.loads(re.search(r'\{.*\}', msg.content[0].text).group())
            ai_pct, reason = data.get("pct", cv_pct), data.get("reason", "AI判定")
        except: pass

    return {"pct": round(cv_pct * 0.3 + ai_pct * 0.7, 1), "reason": reason, "mask": mask}

# --- 5. メイン UI ---
st.markdown('<div class="title-block">🍱 Bento Checker Pro</div>', unsafe_allow_html=True)
client = get_anthropic_client()
if 'selected_idx' not in st.session_state: st.session_state.selected_idx = None

with st.sidebar:
    if st.button("＋ 新規スキャン", use_container_width=True):
        st.session_state.selected_idx = None
        st.rerun()
    st.markdown("### 設定")
    up_ref = st.file_uploader("空容器登録(精度向上)", type=['jpg', 'png'])
    if up_ref:
        Image.open(up_ref).save(REFERENCE_FILE)
        st.rerun()
    tolerance = st.slider("色許容度", 10.0, 50.0, 25.0)

# 履歴読み込み
history = pd.read_csv(DB_FILE).to_dict('records') if os.path.exists(DB_FILE) else []

if st.session_state.selected_idx is None:
    up = st.file_uploader("お弁当の写真をアップロード", type=['jpg', 'png'])
    if up:
        img_orig = Image.open(up).convert("RGB")
        img_np = np.array(img_orig)
        h, w = img_np.shape[:2]
        # 解析エリアの定義 (右上, 左上, 右下)
        area_defs = {
            "右上（メイン）": (int(h*0.1), int(h*0.48), int(w*0.45), int(w*0.95)),
            "左上（副菜）": (int(h*0.1), int(h*0.48), int(w*0.05), int(w*0.45)),
            "右下（副菜）": (int(h*0.5), int(h*0.95), int(w*0.5), int(w*0.95))
        }
        
        ref_stats = None
        if os.path.exists(REFERENCE_FILE):
            ref_b = cv2.imread(REFERENCE_FILE)
            ref_lab = cv2.cvtColor(normalize_image(ref_b), cv2.COLOR_BGR2LAB)
            ref_stats = {"mean": np.mean(ref_lab, axis=(0,1)).tolist()}

        results, draw_np = [], img_np.copy()
        for name, (y1, y2, x1, x2) in area_defs.items():
            roi = img_orig.crop((x1, y1, x2, y2))
            res = compute_emptiness_hybrid(client, roi, name, ref_stats, tolerance)
            results.append(f"{name}@{res['pct']}") # 区切りを@に変更
            m = res["mask"]
            target = draw_np[y1:y1+m.shape[0], x1:x1+m.shape[1]]
            target[m > 0] = target[m > 0] * 0.5 + np.array([255, 230, 0]) * 0.5
            
        path = f"{SAVE_DIR}/res_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        Image.fromarray(draw_np).save(path)
        avg_v = np.mean([float(r.split("@")[1]) for r in results])
        new_rec = {"time": datetime.now().strftime("%m/%d %H:%M"), "status": "PASS" if avg_v < 15 else "FAIL", "img_path": path, "avg_emptiness": round(avg_v, 1), "detail_text": " / ".join(results)}
        pd.DataFrame([new_rec]).to_csv(DB_FILE, mode='a', header=not os.path.exists(DB_FILE), index=False)
        st.session_state.selected_idx = len(history)
        st.rerun()
else:
    data = history[st.session_state.selected_idx]
    c1, c2 = st.columns([1.3, 0.7])
    with c1: st.image(data['img_path'], use_container_width=True)
    with c2:
        st.markdown(f'<div class="status-badge {data["status"].lower()}">{data["status"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><small>平均空き率</small><div class="metric-value">{data["avg_emptiness"]}%</div></div>', unsafe_allow_html=True)
        # エラー箇所修正：区切り文字@で分割
        for part in data['detail_text'].split(" / "):
            if "@" in part:
                nm, pct = part.split("@")
                st.markdown(f'<div class="metric-card"><small>{nm}</small><div class="metric-value">{pct}%</div></div>', unsafe_allow_html=True)
        if st.button("← 戻る"):
            st.session_state.selected_idx = None
            st.rerun()
