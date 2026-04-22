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
st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
REFERENCE_FILE = "reference_empty.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- スタイル (元のデザインを維持) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #f0ede8; }
.stApp { background: linear-gradient(135deg, #f0ede8 0%, #e8e4de 100%); }
.title-block { font-family: 'Space Mono', monospace; font-size: 2rem; letter-spacing: -1px; color: #1a1a1a; margin-bottom: 4px; }
.subtitle-block { font-size: 0.85rem; color: #888; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 24px; }
.metric-card { background: white; border-radius: 20px; padding: 20px 24px; box-shadow: 0 2px 16px rgba(0,0,0,0.07); margin-bottom: 12px; border-left: 5px solid #ccc; transition: transform 0.2s; }
.metric-card:hover { transform: translateY(-2px); }
.metric-card.pass { border-left-color: #2ecc71; }
.metric-card.fail { border-left-color: #e74c3c; }
.metric-card.warn { border-left-color: #f39c12; }
.metric-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: #999; margin-bottom: 4px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #1a1a1a; }
.metric-label { font-size: 0.8rem; color: #888; margin-top: 2px; }
.status-badge { display: inline-block; padding: 6px 20px; border-radius: 999px; font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700; letter-spacing: 2px; margin-bottom: 16px; }
.status-badge.pass { background: #d4f5e2; color: #1a8a4a; }
.status-badge.fail { background: #fde8e8; color: #c0392b; }
.ai-comment { background: #fafafa; border: 1px solid #e8e8e8; border-radius: 14px; padding: 16px 20px; font-size: 0.9rem; color: #444; line-height: 1.7; margin-top: 8px; }
.section-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #bbb; margin-bottom: 6px; }
div[data-testid="stFileUploader"] { background: white; border: 2px dashed #d0ccc5; border-radius: 20px; padding: 16px; }
[data-testid="stSidebar"] { background: #1a1a1a !important; }
[data-testid="stSidebar"] * { color: #f0ede8 !important; }
</style>
""", unsafe_allow_html=True)

# --- 商用グレードの色補正ロジック ---
def normalize_image(img_bgr):
    """Gray World ホワイトバランス補正 + CLAHE コントラスト補正"""
    res = img_bgr.astype(np.float32)
    # ホワイトバランス
    avg_b, avg_g, avg_r = np.mean(res[:, :, 0]), np.mean(res[:, :, 1]), np.mean(res[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    res[:, :, 0] *= (avg_gray / (avg_b + 1e-6))
    res[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    res[:, :, 2] *= (avg_gray / (avg_r + 1e-6))
    res = np.clip(res, 0, 255).astype(np.uint8)
    # コントラスト(CLAHE)
    lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# --- Anthropic クライアント ---
@st.cache_resource
def get_anthropic_client():
    try:
        return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except:
        st.error("ANTHROPIC_API_KEY が設定されていません。")
        st.stop()

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- 座標クランプ ---
def clamp_rect(y1, y2, x1, x2, w, h):
    x1i, y1i = max(0, min(int(x1), w-1)), max(0, min(int(y1), h-1))
    x2i, y2i = max(0, min(int(x2), w-1)), max(0, min(int(y2), h-1))
    if x2i <= x1i or y2i <= y1i: return None
    return x1i, y1i, x2i, y2i

# --- 空容器リファレンス管理 ---
def load_reference_image():
    if os.path.exists(REFERENCE_FILE):
        return Image.open(REFERENCE_FILE).convert("RGB")
    return None

def get_reference_tray_lab_current():
    ref_img = load_reference_image()
    if ref_img is None: return None
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_norm = normalize_image(ref_bgr)
    ref_lab = cv2.cvtColor(ref_norm, cv2.COLOR_BGR2LAB)
    return {"mean": np.mean(ref_lab, axis=(0, 1)).tolist()}

# --- ハイブリッド解析 (CV + Vision) ---
def compute_emptiness_cv(roi_pil, area_name, ref_stats, tolerance):
    roi_bgr = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    roi_norm = normalize_image(roi_bgr)
    roi_lab = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2LAB).astype(np.float32)

    if ref_stats:
        mean = np.array(ref_stats["mean"], dtype=np.float32)
        diff = roi_lab - mean
        dist = np.sqrt(np.sum(diff**2 * [0.6, 1.2, 1.2], axis=2))
        mask = (dist < tolerance).astype(np.uint8) * 255
    else:
        hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, (0,70,40), (15,255,255)), cv2.inRange(hsv, (165,70,40), (180,255,255)))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    pct = (np.sum(mask > 0) / mask.size) * 100.0
    return {"pct": round(pct, 1), "mask": mask}

def compute_emptiness_vision(client, roi_pil, area_name):
    try:
        b64 = pil_to_base64(roi_pil)
        prompt = f"お弁当の「{area_name}」エリアの画像です。赤いトレー底面が見えている割合を0-100の数値で判定し、以下のJSON形式のみで回答してください。{{\"emptiness_pct\": 数値, \"reason\": \"理由(20字以内)\"}}"
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": [{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},{"type":"text","text":prompt}]}]
        )
        return json.loads(re.search(r'\{.*\}', msg.content[0].text).group())
    except: return None

# --- エリア分割 (既存のロジックを商用向けに調整) ---
def detect_bento_areas(img_bgr):
    h, w = img_bgr.shape[:2]
    # デフォルトの3分割（右上・左上・右下）
    return {
        "上左（小おかず）": (int(h*0.12), int(h*0.46), int(w*0.12), int(w*0.42)),
        "上右（大おかず）": (int(h*0.12), int(h*0.46), int(w*0.42), int(w*0.88)),
        "下右（小おかず）": (int(h*0.52), int(h*0.88), int(w*0.58), int(w*0.88)),
        "下左（ごはん）": (int(h*0.52), int(h*0.88), int(w*0.12), int(w*0.58))
    }

# --- 描画系 ---
def draw_results_on_image(img_pil, areas, results, area_masks):
    draw_img = np.array(img_pil.convert("RGB"))
    for name, (y1, y2, x1, x2) in areas.items():
        if name in area_masks:
            m = area_masks[name]
            roi = draw_img[y1:y1+m.shape[0], x1:x1+m.shape[1]]
            roi[m > 0] = roi[m > 0] * 0.5 + np.array([255, 230, 0]) * 0.5
    return Image.fromarray(draw_img)

# --- 履歴管理系 ---
def load_shared_history():
    if os.path.exists(DB_FILE): return pd.read_csv(DB_FILE).to_dict('records')
    return []

def save_history(record):
    pd.DataFrame([record]).to_csv(DB_FILE, mode='a', header=not os.path.exists(DB_FILE), index=False)

# --- メイン UI ---
st.markdown('<div class="title-block">🍱 Bento Checker Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-block">Hybrid CV & AI Analysis</div>', unsafe_allow_html=True)

client = get_anthropic_client()
if 'selected_idx' not in st.session_state: st.session_state.selected_idx = None

# サイドバー
with st.sidebar:
    st.markdown('<div style="padding:10px 0;">🍱 Bento Checker</div>', unsafe_allow_html=True)
    if st.button("＋ 新規スキャン", use_container_width=True):
        st.session_state.selected_idx = None
        st.rerun()
    
    st.markdown("---")
    ref_stats = get_reference_tray_lab_current()
    if ref_stats: st.success("空容器: 登録済")
    else: st.warning("空容器: 未登録")
    
    up_ref = st.file_uploader("空容器を登録", type=['jpg', 'png'])
    if up_ref:
        Image.open(up_ref).save(REFERENCE_FILE)
        st.rerun()
    
    tolerance = st.slider("色許容度", 10.0, 45.0, 25.0)

# メイン処理
history = load_shared_history()
if st.session_state.selected_idx is None:
    up = st.file_uploader("お弁当の写真をアップロード", type=['jpg', 'png'])
    if up:
        img_orig = Image.open(up).convert("RGB")
        progress = st.progress(0, "解析中...")
        
        areas = detect_bento_areas(np.array(img_orig))
        results = []
        area_masks = {}
        
        for i, (name, (y1, y2, x1, x2)) in enumerate(areas.items()):
            if "ごはん" in name: continue
            roi = img_orig.crop((x1, y1, x2, y2))
            cv_res = compute_emptiness_cv(roi, name, ref_stats, tolerance)
            vis_res = compute_emptiness_vision(client, roi, name)
            
            # ハイブリッド重み付け判定
            final_pct = (cv_res["pct"] * 0.3 + vis_res["emptiness_pct"] * 0.7) if vis_res else cv_res["pct"]
            results.append({"name": name, "pct": round(final_pct, 1), "reason": vis_res["reason"] if vis_res else "CV"})
            area_masks[name] = cv_res["mask"]
            progress.progress((i+1)/len(areas))

        # 保存と表示
        out_pil = draw_results_on_image(img_orig, areas, results, area_masks)
        path = f"{SAVE_DIR}/res_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        out_pil.save(path)
        
        avg_pct = np.mean([r["pct"] for r in results])
        status = "PASS" if all(r["pct"] < 15 for r in results) else "FAIL"
        
        new_rec = {
            "time": datetime.now().strftime("%m/%d %H:%M"),
            "status": status,
            "img_path": path,
            "avg_emptiness": round(avg_pct, 1),
            "detail_text": " / ".join([f"{r['name']}:{r['pct']}%" for r in results])
        }
        save_history(new_rec)
        st.session_state.selected_idx = len(load_shared_history()) - 1
        st.rerun()
else:
    # 詳細表示画面 (元のデザインを維持)
    data = history[st.session_state.selected_idx]
    col_img, col_info = st.columns([1.3, 0.7])
    with col_img:
        st.image(data['img_path'], use_container_width=True)
    with col_info:
        badge_cls = "pass" if data['status'] == "PASS" else "fail"
        st.markdown(f'<div class="status-badge {badge_cls}">{data["status"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">平均空き率</div><div class="metric-value">{data["avg_emptiness"]}%</div>', unsafe_allow_html=True)
        
        for part in data['detail_text'].split(" / "):
            nm, pct = part.split(":")
            st.markdown(f'''<div class="metric-card">
                <div class="metric-title">{nm}</div>
                <div class="metric-value">{pct}</div>
            </div>''', unsafe_allow_html=True)
        
        if st.button("← 戻る"):
            st.session_state.selected_idx = None
            st.rerun()
