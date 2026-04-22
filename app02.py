import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
from datetime import datetime
import anthropic
import base64
import json
import io
import re

# --- 初期設定 ---
st.set_page_config(page_title="Bento Checker Pro Max", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- スタイル設定 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f8f9fa; }
.title-block { font-family: 'Space Mono', monospace; font-size: 2.2rem; color: #1e1e1e; margin-bottom: 0.5rem; }
.metric-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #ddd; }
.metric-card.pass { border-top-color: #28a745; }
.metric-card.fail { border-top-color: #dc3545; }
.metric-card.warn { border-top-color: #ffc107; }
.status-badge { padding: 8px 24px; border-radius: 50px; font-weight: 700; font-family: 'Space Mono'; display: inline-block; }
.status-badge.pass { background: #e7f3eb; color: #28a745; }
.status-badge.fail { background: #fbebed; color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- Anthropic クライアント ---
@st.cache_resource
def get_anthropic_client():
    try:
        return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except:
        st.error("APIキーが設定されていません。")
        st.stop()

# --- 高度な画像補正ロジック (商用グレード) ---
def normalize_image(img_bgr):
    """
    1. Gray World Assumption によるホワイトバランス補正
    2. CLAHE によるコントラスト最適化（影の誤検出防止）
    """
    # ホワイトバランス補正
    res = img_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(res[:, :, 0]), np.mean(res[:, :, 1]), np.mean(res[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    res[:, :, 0] *= (avg_gray / (avg_b + 1e-6))
    res[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    res[:, :, 2] *= (avg_gray / (avg_r + 1e-6))
    res = np.clip(res, 0, 255).astype(np.uint8)

    # 局所コントラスト補正 (CLAHE)
    lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    res = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return res

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- 面積計算ロジック ---
def compute_emptiness_cv(roi_pil, area_name, ref_stats, tolerance):
    roi_bgr = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    roi_norm = normalize_image(roi_bgr)
    
    # LAB色空間での距離判定
    roi_lab = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    if ref_stats:
        mean = np.array(ref_stats["mean"], dtype=np.float32)
        diff = roi_lab - mean
        dist = np.sqrt(np.sum(diff**2 * [0.6, 1.2, 1.2], axis=2))
        mask = (dist < tolerance).astype(np.uint8) * 255
    else:
        # フォールバック: HSV
        hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([165, 70, 40]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 比率計算
    pct = (np.sum(mask > 0) / mask.size) * 100.0
    return {"pct": round(pct, 1), "mask": mask}

def compute_emptiness_vision(client, roi_pil, area_name):
    """Claude Vision による意味的解析"""
    try:
        b64 = pil_to_base64(roi_pil)
        prompt = f"お弁当の{area_name}です。赤いトレーの底面が見えている面積（空き率）を0-100%で判定し、JSON形式で返してください。回答例: {{\"pct\": 15.5, \"reason\": \"食材の隙間から底面が露出\"}}"
        
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}
            ]}]
        )
        data = json.loads(re.search(r'\{.*\}', msg.content[0].text).group())
        return data
    except:
        return None

# --- メインロジック ---
def analyze_bento(client, img_orig, ref_stats, tolerance):
    img_bgr = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    # 簡易エリア分割 (右上, 左上, 右下)
    areas = {
        "左上（副菜）": (int(h*0.1), int(h*0.45), int(w*0.1), int(w*0.45)),
        "右上（主菜）": (int(h*0.1), int(h*0.45), int(w*0.45), int(w*0.9)),
        "右下（副菜）": (int(h*0.5), int(h*0.9), int(w*0.55), int(w*0.9)),
    }
    
    results = []
    overlay_mask = np.zeros((h, w), dtype=np.uint8)

    for name, (y1, y2, x1, x2) in areas.items():
        roi = img_orig.crop((x1, y1, x2, y2))
        
        # ハイブリッド解析
        cv_res = compute_emptiness_cv(roi, name, ref_stats, tolerance)
        vision_res = compute_emptiness_vision(client, roi, name)
        
        # 最終パーセントの決定 (Visionが利用できれば70%の重みで合成)
        if vision_res:
            final_pct = (cv_res["pct"] * 0.3) + (vision_res["pct"] * 0.7)
            reason = vision_res["reason"]
        else:
            final_pct = cv_res["pct"]
            reason = "CV解析のみ"
            
        results.append({"name": name, "pct": final_pct, "reason": reason})
        
        # マスクの合成
        m = cv_res["mask"]
        overlay_mask[y1:y1+m.shape[0], x1:x1+m.shape[1]] = m

    return results, overlay_mask

# --- Streamlit UI ---
def main():
    st.markdown('<div class="title-block">Bento Checker Pro Max</div>', unsafe_allow_html=True)
    client = get_anthropic_client()
    
    # サイドバー設定
    with st.sidebar:
        st.header("Settings")
        tolerance = st.slider("Color Tolerance", 10.0, 50.0, 25.0)
        up_ref = st.file_uploader("Reference (Empty Tray)", type=['jpg', 'png'])
        
        ref_stats = None
        if up_ref:
            ref_img = Image.open(up_ref).convert("RGB")
            ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
            ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
            ref_stats = {"mean": np.mean(ref_lab, axis=(0, 1)).tolist()}
            st.success("Reference Loaded")

    # メインアップローダー
    up_main = st.file_uploader("Analyze Bento Image", type=['jpg', 'png'])
    
    if up_main:
        img = Image.open(up_main).convert("RGB")
        with st.spinner("Analyzing with Hybrid AI..."):
            results, mask = analyze_bento(client, img, ref_stats, tolerance)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            # 視覚化
            img_res = np.array(img)
            img_res[mask > 0] = [255, 255, 0] # 空きを黄色でハイライト
            st.image(img_res, caption="Detection Result", use_container_width=True)
            
        with col2:
            st.subheader("Analysis Metrics")
            for r in results:
                status = "fail" if r["pct"] > 15 else "pass"
                st.markdown(f"""
                <div class="metric-card {status}">
                    <small>{r['name']}</small>
                    <h3>{r['pct']:.1f}%</h3>
                    <p style='font-size:0.8rem; color:#666;'>{r['reason']}</p>
                </div><br>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
