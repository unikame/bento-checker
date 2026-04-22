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

# --- 1. APIキーの設定 (ここを一番確認してください！) ---
# 画像で見えていた sk-ant-api03-... の【すべての文字列】をここに貼り付けます。
ANTHROPIC_API_KEY = "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"

st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. スタイル設定 ---
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
[data-testid="stSidebar"] { background: #333333 !important; }
[data-testid="stSidebar"] * { color: #f0ede8 !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. 画像処理・AIエンジン ---
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
    if "sk-ant" not in ANTHROPIC_API_KEY:
        st.error("⚠️ APIキーが正しく設定されていません。sk-ant- で始まるキーを入力してください。")
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- 4. メイン UI ---
st.markdown('<h2 style="color:#1a1a1a; font-family:Space Mono;">🍱 Bento Checker Pro</h2>', unsafe_allow_html=True)

client = get_anthropic_client()

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

up = st.file_uploader("お弁当の写真をアップロードしてください", type=['jpg', 'png', 'jpeg'])

if up and "processed" not in st.session_state:
    with st.spinner("AIと通信中..."):
        img_orig = Image.open(up).convert("RGB")
        h, w = np.array(img_orig).shape[:2]
        
        # 判定エリアをより厳密に設定
        area_defs = {
            "右上（メイン）": (int(h*0.08), int(h*0.48), int(w*0.35), int(w*0.95)),
            "左上（副菜）": (int(h*0.08), int(h*0.48), int(w*0.05), int(w*0.35)),
            "右下（副菜）": (int(h*0.5), int(h*0.95), int(w*0.52), int(w*0.95))
        }
        
        results, debug_logs = [], []
        
        for name, (y1, y2, x1, x2) in area_defs.items():
            roi = img_orig.crop((x1, y1, x2, y2))
            final_pct, reason = 0.0, "通信エラー"
            
            if client:
                try:
                    b64 = pil_to_base64(roi)
                    msg = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=300,
                        messages=[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},{"type":"text","text":"Analyze tray emptiness (RED plastic area). If there is a clear red gap next to food, return at least 18%. JSON ONLY: {\"pct\": number, \"reason\": \"string\"}"}]}]
                    )
                    data = json.loads(re.search(r'\{.*\}', msg.content[0].text).group())
                    final_pct = data.get("pct", 0.0)
                    reason = data.get("reason", "判定完了")
                    debug_logs.append(f"✅ {name}: {final_pct}%")
                except Exception as e:
                    # エラーの詳細をログに残す
                    reason = f"エラー: {str(e)[:50]}"
                    debug_logs.append(f"❌ {name}: 通信失敗 ({str(e)})")
            
            results.append({"name": name, "pct": final_pct, "reason": reason})
        
        st.session_state.processed = {"results": results, "logs": debug_logs, "img": img_orig}
        st.rerun()

# --- 結果表示 ---
if "processed" in st.session_state:
    p = st.session_state.processed
    c1, c2 = st.columns([1.2, 0.8])
    
    with c1:
        st.image(p["img"], use_container_width=True)
        with st.expander("🛠 AI通信ログ（接続エラーの確認）"):
            for log in p["logs"]:
                st.write(log)
            if not p["logs"]:
                st.error("AIとの通信が一度も行われませんでした。APIキーを確認してください。")

    with c2:
        # 15%以上が一つでもあればFAIL
        is_fail = any(r["pct"] >= 15.0 for r in p["results"])
        status = "FAIL" if is_fail else "PASS"
        st.markdown(f'<div class="status-badge {status.lower()}">{status}</div>', unsafe_allow_html=True)
        
        for r in p["results"]:
            cls = "fail" if r["pct"] >= 15.0 else "pass"
            st.markdown(f"""
            <div class="metric-card {cls}">
                <small style="color:#666;">{r['name']}</small>
                <div class="metric-value">{r['pct']}%</div>
                <div style="font-size:0.8rem; color:#888;">{r['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
