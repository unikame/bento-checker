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

# --- 初期設定 ---
st.set_page_config(page_title="Bento Checker Pro", layout="wide", page_icon="🍱")

DB_FILE = "shared_history.csv"
SAVE_DIR = "history_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- スタイル ---
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
[data-testid="stSidebar"] .stButton button { background: #2a2a2a; border: 1px solid #3a3a3a; color: #f0ede8; border-radius: 10px; font-size: 0.8rem; }
[data-testid="stSidebar"] .stButton button:hover { background: #3a3a3a; border-color: #555; }
.new-scan-btn button { background: #f0ede8 !important; color: #1a1a1a !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; border-radius: 10px !important; border: none !important; }
</style>
""", unsafe_allow_html=True)


# --- Anthropic クライアント ---
@st.cache_resource
def get_anthropic_client():
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        st.error("ANTHROPIC_API_KEY が設定されていません。")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)



    if not api_key:
        st.error("ANTHROPIC_API_KEY が設定されていません。")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def pil_to_base64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# --- Claude Vision による充填率判定 ---
def analyze_area_with_claude(client, area_img: Image.Image, area_name: str) -> dict:
    prompt = f"""あなたはお弁当の品質検査AIです。
この画像は「{area_name}」エリアを真上から撮影・補正したものです。
空き率（食材が占めていない面積の割合）を厳密に計算してJSON形式のみで返してください。

判定ルール：
1. トレーの赤い底面・菱形模様が見えている面積 → 空き
2. 食材が偏っていても、空いている部分は空きとしてカウント
3. カップ・仕切り紙は容器なので空き率の計算に含めない
4. カップの中に食材が入っていれば埋まっているとみなす
5. 食材（お米・おかず）が占めている面積のみ埋まっているとカウント

空き率の目安：
- 0〜15%：ほぼ全面に食材が詰まっている
- 15〜30%：一部に隙間がある
- 30〜50%：食材が少なく隙間が目立つ
- 50%以上：かなり空いている

返答形式（JSONのみ・前後に文字を入れない）:
{{"emptiness_pct": 数値, "confidence": "high/medium/low", "reason": "理由を30字以内で"}}"""

    b64 = pil_to_base64(area_img)
    try:
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        raw = msg.content[0].text.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return {
            "emptiness_pct": float(data.get("emptiness_pct", 50)),
            "confidence": data.get("confidence", "medium"),
            "reason": data.get("reason", "")
        }
    except Exception as e:
        return {"emptiness_pct": 50.0, "confidence": "low", "reason": f"解析エラー: {e}"}


def analyze_overall_with_claude(client, img: Image.Image, results: list) -> str:
    summary = "\n".join([f"- {r['name']}: 空き率{r['emptiness_pct']:.1f}%" for r in results])
    prompt = f"""お弁当の品質検査結果です：
{summary}

品質検査員として充填状況の総評を日本語で2〜3文で述べてください。改善が必要な点があれば具体的に指摘してください。"""
    b64 = pil_to_base64(img)
    try:
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"総評の生成に失敗しました: {e}"


# --- トレー検出・射影変換 ---
# --- エリア分割ロジック（食材エリア直接検出）---
def detect_bento_areas(img_bgr: np.ndarray):
    """
    トレーの赤い仕切りで囲まれた食材エリアを直接検出。
    戻り値: areas（矩形座標 y1,y2,x1,x2）
    失敗時は固定比率フォールバック。
    """
    h, w = img_bgr.shape[:2]

    # トレー色マスク（暗めの赤茶色）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    tray1 = cv2.inRange(hsv, np.array([0,  120, 60]),  np.array([10, 255, 160]))
    tray2 = cv2.inRange(hsv, np.array([170,120, 60]),  np.array([180,255, 160]))
    tray_mask = cv2.bitwise_or(tray1, tray2)
    kernel = np.ones((20,20), np.uint8)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, kernel)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_OPEN, kernel)

    # トレー全体を塗りつぶし
    contours, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        filled = np.zeros_like(tray_mask)
        cv2.drawContours(filled, [contours[0]], -1, 255, -1)

        # 食材エリア = トレー内側の非赤部分
        food_mask = cv2.bitwise_and(filled, cv2.bitwise_not(tray_mask))
        food_contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = h * w * 0.03
        food_contours = [c for c in food_contours if cv2.contourArea(c) > min_area]
        food_contours = sorted(food_contours, key=cv2.contourArea, reverse=True)[:4]

        if len(food_contours) == 4:
            boxes = []
            for c in food_contours:
                fx, fy, fw, fh = cv2.boundingRect(c)
                boxes.append({'x1':fx,'y1':fy,'x2':fx+fw,'y2':fy+fh,'cx':fx+fw/2,'cy':fy+fh/2})

            cy_med = np.median([b['cy'] for b in boxes])
            top = sorted([b for b in boxes if b['cy'] < cy_med], key=lambda b: b['cx'])
            bot = sorted([b for b in boxes if b['cy'] >= cy_med], key=lambda b: b['cx'])

            if len(top) == 2 and len(bot) == 2:
                tl, tr, bl, br = top[0], top[1], bot[0], bot[1]
                # 上右の右端を下右の右端に揃える
                tr_x2 = br['x2']
                areas = {
                    "上左（小おかず）": (tl['y1'], tl['y2'], tl['x1'], tl['x2']),
                    "上右（大おかず）": (tr['y1'], tr['y2'], tr['x1'], tr_x2),
                    "下左（ごはん）":   (bl['y1'], bl['y2'], bl['x1'], bl['x2']),
                    "下右（小おかず）": (br['y1'], br['y2'], br['x1'], br['x2']),
                }
                return areas

    # フォールバック: 固定比率
    y1          = int(h * 0.10)
    h_split     = int(h * 0.46)
    x1_top      = int(w * 0.10)
    x2_top      = int(w * 0.92)
    v_top       = int(w * 0.36)
    h_split_bot = int(h * 0.50)
    y2          = int(h * 0.92)
    x1_bot      = int(w * 0.13)
    x2_bot      = int(w * 0.92)
    v_bot       = int(w * 0.63)
    return {
        "上左（小おかず）": (y1,          h_split,     x1_top, v_top),
        "上右（大おかず）": (y1,          h_split,     v_top,  x2_top),
        "下左（ごはん）":   (h_split_bot, y2,          x1_bot, v_bot),
        "下右（小おかず）": (h_split_bot, y2,          v_bot,  x2_bot),
    }


def draw_results_on_image(img_pil: Image.Image, areas: dict, results: list) -> Image.Image:
    output = img_pil.copy()
    result_map = {r["name"]: r for r in results}

    w, _ = output.size
    font_size = max(20, int(w * 0.035))
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "c:/Windows/Fonts/arialbd.ttf",
    ]
    font = ImageFont.load_default()
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
            except:
                pass
            break

    for name, (y1, y2, x1, x2) in areas.items():
        r = result_map.get(name, {})
        pct = r.get("emptiness_pct", 0)

        if pct < 15:
            color = (46, 204, 113)
            alpha = 30
        elif pct < 30:
            color = (243, 156, 18)
            alpha = 40
        else:
            color = (231, 76, 60)
            alpha = 50

        overlay = Image.new('RGBA', output.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha))
        output = Image.alpha_composite(output.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(output, 'RGBA')

        line_w = max(3, int(w * 0.004))
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 230), width=line_w)

        tx, ty = x1 + 10, y1 + 10
        text = f"{pct:.1f}%"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([tx-4, ty-4, tx+tw+8, ty+th+8], fill=(0,0,0,160), outline=(255,255,255,80))
        draw.text((tx+2, ty+2), text, font=font, fill=(0,0,0,120))
        draw.text((tx, ty), text, font=font, fill=(255,255,255,240))

    return output


# --- 履歴管理 ---
def load_shared_history():
    if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
        try:
            return pd.read_csv(DB_FILE).to_dict('records')
        except:
            return []
    return []


def save_history(record: dict):
    df = pd.DataFrame([record])
    df.to_csv(DB_FILE, mode='a', header=not os.path.exists(DB_FILE), index=False)


def delete_history_item(idx: int) -> bool:
    history = load_shared_history()
    if 0 <= idx < len(history):
        item = history.pop(idx)
        img_p = item.get('img_path', '')
        if img_p and os.path.exists(str(img_p)):
            try:
                os.remove(img_p)
            except:
                pass
        if history:
            pd.DataFrame(history).to_csv(DB_FILE, index=False)
        else:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
        return True
    return False


# --- セッション初期化 ---
for key, val in [('last_processed_file', None), ('selected_idx', None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# =====================
# サイドバー
# =====================
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #f0ede8; font-weight: 700;">🍱 Bento Checker</div>
        <div style="font-size: 0.7rem; color: #888; letter-spacing: 2px; margin-top: 2px;">HISTORY</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="new-scan-btn">', unsafe_allow_html=True)
        if st.button("＋  新規スキャン", use_container_width=True):
            st.session_state.selected_idx = None
            st.session_state.last_processed_file = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #333; margin: 12px 0;">', unsafe_allow_html=True)

    history = load_shared_history()
    if not history:
        st.markdown('<div style="color: #555; font-size: 0.8rem; padding: 8px;">履歴なし</div>', unsafe_allow_html=True)

    for idx, item in enumerate(reversed(history)):
        real_idx = len(history) - 1 - idx
        col1, col2 = st.columns([0.82, 0.18])
        with col1:
            icon = "✅" if item.get('status') == "PASS" else "❌"
            avg = item.get('avg_emptiness', '?')
            label = f"{icon} {item.get('time')}  ({avg}%)"
            if st.button(label, key=f"h_{real_idx}", use_container_width=True):
                st.session_state.selected_idx = real_idx
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{real_idx}"):
                delete_history_item(real_idx)
                if st.session_state.selected_idx == real_idx:
                    st.session_state.selected_idx = None
                st.rerun()


# =====================
# メイン画面
# =====================
st.markdown('<div class="title-block">🍱 Bento Checker Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-block">AI-Powered Filling Analysis</div>', unsafe_allow_html=True)

client = get_anthropic_client()
history = load_shared_history()

# --- 履歴詳細表示 ---
if st.session_state.selected_idx is not None and st.session_state.selected_idx < len(history):
    data = history[st.session_state.selected_idx]
    col_img, col_info = st.columns([1.3, 0.7])

    with col_img:
        img_p = data.get('img_path', '')
        if os.path.exists(str(img_p)):
            st.image(img_p, use_container_width=True)
        else:
            st.error("画像ファイルが見つかりません。")

    with col_info:
        status_val = data.get('status', 'FAIL')
        badge_cls = "pass" if status_val == "PASS" else "fail"
        st.markdown(f'<div class="status-badge {badge_cls}">{status_val}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">分析日時</div><div style="margin-bottom:12px; font-size:0.95rem;">{data.get("time")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">平均空き率</div><div style="font-family:Space Mono,monospace; font-size:2rem; font-weight:700; margin-bottom:12px;">{data.get("avg_emptiness", "?")}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">エリア詳細</div>', unsafe_allow_html=True)
        detail_text = data.get('detail_text', '')
        for part in detail_text.split(" / "):
            if ":" in part:
                nm, pct_str = part.rsplit(":", 1)
                try:
                    pct = float(pct_str.replace("%", ""))
                    cls = "pass" if pct < 15 else ("warn" if pct < 30 else "fail")
                    st.markdown(f'''<div class="metric-card {cls}">
                        <div class="metric-title">{nm}</div>
                        <div class="metric-value">{pct:.1f}%</div>
                        <div class="metric-label">空き率</div>
                    </div>''', unsafe_allow_html=True)
                except:
                    st.write(part)

        ai_comment = data.get('ai_comment', '')
        if ai_comment:
            st.markdown('<div class="section-label" style="margin-top:12px;">AI総評</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-comment">{ai_comment}</div>', unsafe_allow_html=True)

        if st.button("← 戻る", use_container_width=True):
            st.session_state.selected_idx = None
            st.rerun()

# --- 新規スキャン ---
else:
    up = st.file_uploader(
        "お弁当の写真をアップロード",
        type=['jpg', 'jpeg', 'png'],
        help="JPG または PNG 形式の画像をアップロードしてください"
    )

    if up and up.name != st.session_state.last_processed_file:
        progress_bar = st.progress(0, text="画像を読み込んでいます...")

        img_orig = Image.open(up).convert("RGB")
        img_bgr  = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)

        progress_bar.progress(10, text="トレーを検出・補正中...")
        areas = detect_bento_areas(img_bgr)

        results = []
        total_areas = len(areas)

        for i, (name, (y1, y2, x1, x2)) in enumerate(areas.items()):
            progress_bar.progress(
                20 + int(60 * i / total_areas),
                text=f"Claude Vision で「{name}」を解析中... ({i+1}/{total_areas})"
            )
            roi = img_orig.crop((x1, y1, x2, y2))
            analysis = analyze_area_with_claude(client, roi, name)
            results.append({
                "name": name,
                "emptiness_pct": analysis["emptiness_pct"],
                "confidence": analysis["confidence"],
                "reason": analysis["reason"],
            })

        progress_bar.progress(85, text="全体評価を生成中...")
        avg_pct = np.mean([r["emptiness_pct"] for r in results])
        is_pass = avg_pct < 20.0 and all(r["emptiness_pct"] < 30.0 for r in results)

        ai_comment = analyze_overall_with_claude(client, img_orig, results)

        progress_bar.progress(92, text="結果を描画中...")
        output_pil = draw_results_on_image(img_orig, areas, results)

        path = f"{SAVE_DIR}/res_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_pil.save(path, quality=92)

        detail_text = " / ".join([f"{r['name']}:{r['emptiness_pct']:.1f}%" for r in results])
        new_rec = {
            "time": datetime.now().strftime("%m/%d %H:%M"),
            "status": "PASS" if is_pass else "FAIL",
            "img_path": path,
            "detail_text": detail_text,
            "avg_emptiness": f"{avg_pct:.1f}",
            "ai_comment": ai_comment,
        }
        save_history(new_rec)

        progress_bar.progress(100, text="完了！")
        st.session_state.last_processed_file = up.name
        st.session_state.selected_idx = len(load_shared_history()) - 1
        st.rerun()

    elif not up:
        st.markdown("""
        <div style="text-align:center; padding: 48px 24px; color: #aaa;">
            <div style="font-size: 3rem; margin-bottom: 16px;">📷</div>
            <div style="font-size: 0.9rem; line-height: 1.8;">
                お弁当の写真をアップロードすると<br>
                Claude Vision AI が各エリアの充填率を解析します<br><br>
                <span style="font-size:0.75rem; color:#ccc;">
                ✓ 斜め補正対応 &nbsp;|&nbsp; ✓ AI視覚判定 &nbsp;|&nbsp; ✓ 高精度解析
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
