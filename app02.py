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

# --- 1. APIキーの設定 (Secretsから安全に取得) ---
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except:
    ANTHROPIC_API_KEY = ""

st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

# フォルダ設定
DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 補助関数 ---
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

def pil_to_base64(img: Image.Image) -> str:
    img_rgb = img.convert("RGB")
    img_rgb.thumbnail((800, 800), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@st.cache_resource
def get_anthropic_client():
    if not ANTHROPIC_API_KEY: return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- 3. スタイル設定 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&family=Space+Mono:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f0ede8; }
.metric-card { background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 10px; border-left: 8px solid #ccc; }
.metric-card.fail { border-left-color: #e74c3c; }
.metric-card.pass { border-left-color: #2ecc71; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #1a1a1a; }
.status-badge { display: inline-block; padding: 8px 24px; border-radius: 50px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.3rem; }
.status-badge.pass { background: #d4f5e2; color: #1a8a4a; }
.status-badge.fail { background: #fde8e8; color: #c0392b; }
/* サイドバーのグレー化 */
[data-testid="stSidebar"] { background: #333333 !important; }
[data-testid="stSidebar"] * { color: #f0ede8 !important; }
</style>
""", unsafe_allow_html=True)

# --- 4. メイン処理 ---
st.markdown('<h2 style="color:#1a1a1a; font-family:Space Mono;">🍱 Bento Checker Pro</h2>', unsafe_allow_html=True)
client = get_anthropic_client()
if 'selected_idx' not in st.session_state: st.session_state.selected_idx = None

with st.sidebar:
    if st.button("＋ 新規スキャン", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("📷 **空容器登録**")
    up_ref = st.file_uploader("参考画像をアップロード", type=['jpg', 'png', 'jpeg'], key="ref_up", label_visibility="collapsed")
    if up_ref:
        Image.open(up_ref).convert("RGB").save(REFERENCE_FILE)
        st.success("容器の色を学習しました")

if st.session_state.selected_idx is None:
    up = st.file_uploader("お弁当の写真をアップロードしてください", type=['jpg', 'png', 'jpeg'])
    if up:
        if not ANTHROPIC_API_KEY:
            st.error("⚠️ APIキーが設定されていません。StreamlitのSecrets画面を確認してください。")
        else:
            with st.spinner("AIが隙間を厳密に分析中..."):
                img_orig = Image.open(up).convert("RGB")
                h, w = np.array(img_orig).shape[:2]
                area_defs = {
                    "右上（メイン）": (int(h*0.08), int(h*0.48), int(w*0.35), int(w*0.95)),
                    "左上（副菜）": (int(h*0.08), int(h*0.48), int(w*0.05), int(w*0.35)),
                    "右下（副菜）": (int(h*0.5), int(h*0.95), int(w*0.52), int(w*0.95))
                }
                results = []
                for name, (y1, y2, x1, x2) in area_defs.items():
                    roi = img_orig.crop((x1, y1, x2, y2))
                    try:
                        b64 = pil_to_base64(roi)
                        # AIへの指示（極限まで厳しく）
                        prompt = "Analyze tray emptiness. CRITICAL: If you see the RED plastic bottom (especially the wide gap next to the egg roll), return AT LEAST 15-20%. Overestimate gaps rather than underestimate. JSON ONLY: {\"pct\": number, \"reason\": \"string\"}"
                        msg = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=300,
                            messages=[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},{"type":"text","text":prompt}]}]
                        )
                        data = json.loads(re.search(r'\{.*\}', msg.content[0].text).group())
                        results.append({"name": name, "pct": data.get("pct", 0), "reason": data.get("reason", "")})
                    except Exception as e:
                        results.append({"name": name, "pct": 0, "reason": "通信失敗"})
                
                path = f"{SAVE_DIR}/res_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                img_orig.save(path)
                
                is_fail = any(r["pct"] >= 15.0 for r in results)
                new_rec = {
                    "time": datetime.now().strftime("%m/%d %H:%M"),
                    "status": "FAIL" if is_fail else "PASS",
                    "img_path": path,
                    "avg_emptiness": round(np.mean([r["pct"] for r in results]), 1),
                    "detail_text": " / ".join([f"{r['name']}@{r['pct']}" for r in results])
                }
                pd.DataFrame([new_rec]).to_csv(DB_FILE, mode='a', header=not os.path.exists(DB_FILE), index=False)
                st.session_state.selected_idx = len(pd.read_csv(DB_FILE)) - 1
                st.rerun()
else:
    # 詳細表示
    history = pd.read_csv(DB_FILE).to_dict('records')
    data = history[st.session_state.selected_idx]
    c1, c2 = st.columns([1.3, 0.7])
    with c1: st.image(data['img_path'], use_container_width=True)
    with c2:
        status = data['status'].lower()
        st.markdown(f'<div class="status-badge {status}">{data["status"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><small>平均空き率</small><div class="metric-value">{data["avg_emptiness"]}%</div></div>', unsafe_allow_html=True)
        for part in data['detail_text'].split(" / "):
            if "@" in part:
                nm, pct = part.split("@")
                st.markdown(f'<div class="metric-card {"fail" if float(pct)>=15 else "pass"}"><small>{nm}</small><div class="metric-value">{pct}%</div></div>', unsafe_allow_html=True)
        if st.button("← 戻る", use_container_width=True):
            st.session_state.selected_idx = None
            st.rerun()
