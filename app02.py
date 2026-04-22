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
# 画像の sk-ant-... から始まるキーを正確に貼り付けてください。
ANTHROPIC_API_KEY = "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"

st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 補正ロジック ---
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

# --- 3. スタイル設定 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&family=Space+Mono:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f0ede8; }
.title-block { font-family: 'Space Mono', monospace; font-size: 2.2rem; color: #1a1a1a; margin-bottom: 20px; }
.metric-card { background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 10px; border-left: 8px solid #ccc; }
.metric-card.pass { border-left-color: #2ecc71; }
.metric-card.fail { border-left-color: #e74c3c; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.status-badge { display: inline-block; padding: 8px 24px; border-radius: 50px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.3rem; }
.status-badge.pass { background: #d4f5e2; color: #1a8a4a; }
.status-badge.fail { background: #fde8e8; color: #c0392b; }
.debug-log { background: #1a1a1a; color: #00ff00; font-family: 'Space Mono', monospace; padding: 15px; border-radius: 10px; font-size: 0.8rem; margin-top: 10px; }
[data-testid="stSidebar"] { background: #1a1a1a !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 4. 解析コア ---
def pil_to_base64(img: Image.Image) -> str:
    img_rgb = img.convert("RGB")
    img_rgb.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@st.cache_resource
def get_anthropic_client():
    if "sk-ant" not in ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

client = get_anthropic_client()

# --- メイン画面 ---
st.markdown('<div class="title-block">🍱 Bento Checker Pro Max</div>', unsafe_allow_html=True)

with st.sidebar:
    if st.button("＋ 新規スキャン", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    up_ref = st.file_uploader("📷 空容器登録", type=['jpg', 'png', 'jpeg'])
    if up_ref:
        Image.open(up_ref).convert("RGB").save(REFERENCE_FILE)
        st.success("容器を再学習しました")
    tolerance = st.slider("色許容度 (低=厳格)", 10.0, 50.0, 22.0)

up = st.file_uploader("お弁当の写真をアップロード", type=['jpg', 'png', 'jpeg'])

if up:
    if "processed" not in st.session_state:
        with st.spinner("AIが隙間を逃さずチェックしています..."):
            img_orig = Image.open(up).convert("RGB")
            img_np = np.array(img_orig)
            h, w = img_np.shape[:2]
            
            # エリアをさらに内側に絞り、隙間を逃さない設定
            area_defs = {
                "右上（メイン）": (int(h*0.12), int(h*0.46), int(w*0.35), int(w*0.92)),
                "左上（副菜）": (int(h*0.12), int(h*0.46), int(w*0.08), int(w*0.35)),
                "右下（副菜）": (int(h*0.52), int(h*0.92), int(w*0.55), int(w*0.92))
            }
            
            results, draw_np, debug_data = [], img_np.copy(), []
            
            for name, (y1, y2, x1, x2) in area_defs.items():
                roi = img_orig.crop((x1, y1, x2, y2))
                
                # --- AI判定 (ここがメイン) ---
                ai_pct, ai_reason = 0, "通信失敗"
                if client:
                    try:
                        b64_roi = pil_to_base64(roi)
                        # AIに「15%以下にするな」と強く念押しするプロンプト
                        prompt = """
                        あなたはプロのお弁当品質検査官です。
                        画像内の「赤いプラスチックの底面」が露出している割合（0-100%）を判定してください。
                        
                        【厳格ルール】
                        1. 卵焼きの横などの「赤い大きな隙間」は、食材1品分に相当するため、最低でも15%〜25%以上と判定せよ。
                        2. 4%や5%といった過小評価は、検品ミスに繋がるため厳禁。
                        3. カップの柄は空白ではないが、カップの中に何も入っていない場合は、底面の露出とみなす。
                        
                        回答は以下のJSON形式のみで行え：
                        {"pct": 数値, "reason": "理由を日本語で"}
                        """
                        msg = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=200,
                            messages=[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64_roi}},{"type":"text","text":prompt}]}]
                        )
                        raw_text = msg.content[0].text
                        data = json.loads(re.search(r'\{.*\}', raw_text).group())
                        ai_pct = data.get("pct", 0)
                        ai_reason = data.get("reason", "")
                        debug_data.append(f"[{name}] AI Response: {ai_pct}% - {ai_reason}")
                    except Exception as e:
                        st.error(f"AI通信エラー ({name}): {e}")
                        ai_pct = 0

                results.append({"name": name, "pct": ai_pct, "reason": ai_reason})
            
            avg_v = np.mean([r["pct"] for r in results])
            st.session_state.processed = {
                "results": results,
                "avg": round(avg_v, 1),
                "debug": debug_data,
                "img": img_orig
            }
            st.rerun()

# --- 結果表示 ---
if "processed" in st.session_state:
    p = st.session_state.processed
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.image(p["img"], use_container_width=True)
        with st.expander("🛠 AI解析ログ（ここが表示されればAIは動いています）"):
            for line in p["debug"]:
                st.code(line)

    with col2:
        # いずれかが15%以上なら即座にNG
        is_fail = any(r["pct"] >= 15.0 for r in p["results"])
        status = "FAIL" if is_fail else "PASS"
        st.markdown(f'<div class="status-badge {status.lower()}">{status}</div>', unsafe_allow_html=True)
        
        for r in p["results"]:
            cls = "fail" if r["pct"] >= 15.0 else "pass"
            st.markdown(f"""
            <div class="metric-card {cls}">
                <small style="color:#888;">{r['name']}</small>
                <div class="metric-value">{r['pct']}%</div>
                <div style="font-size:0.8rem; color:#666; margin-top:5px;">{r['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
