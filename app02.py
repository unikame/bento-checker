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


def pil_to_base64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")



# --- Claude Vision による充填率判定 ---
def analyze_area_with_claude(client, area_img: Image.Image, area_name: str) -> dict:
    b64 = pil_to_base64(area_img)

    prompt = f"""You are a bento quality inspector analyzing the "{area_name}" compartment.

The tray is dark reddish-brown (like mahogany) with a diamond/grid embossed pattern.

STEP 1: Identify the tray bottom color and pattern in this image.
STEP 2: Find areas where the tray bottom is directly visible with NO food or cup covering it.
STEP 3: Calculate what percentage of the total compartment area those bare tray areas represent.

EMPTY = bare tray surface with absolutely nothing on it
FILLED = any of the following:
- Food items (beans, vegetables, meat, egg, pickles, etc.)
- Paper cups — AND the tray area surrounding/beneath each cup is also FILLED
  (if a cup exists in the compartment, treat the entire zone around it as filled)
- Dividers, paper liners
- Transparent or semi-transparent food (pickles, jellied items)

KEY RULE: If you see paper cups in the image, the tray area around those cups is NOT empty — count it as filled.
Only count as EMPTY: large bare tray areas where there are clearly NO cups and NO food at all.

emptiness_pct = (truly bare tray area with no food and no cups nearby) / (total compartment area) × 100

Give exact decimal value (e.g. 3.2, 8.7, 17.4). Do NOT round to multiples of 5.

Respond in JSON only:
{{"emptiness_pct": number, "confidence": "high/medium/low", "reason": "理由を30字以内で"}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
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
        import traceback
        print(f"[API ERROR] {area_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"emptiness_pct": 50.0, "confidence": "low", "reason": f"エラー: {str(e)[:20]}"}


def analyze_overall_with_claude(client, img: Image.Image, results: list) -> str:
    summary = "\n".join([f"- {r['name']}: 空き率{r['emptiness_pct']:.1f}%" for r in results])
    prompt = f"""お弁当の品質検査結果です：
{summary}

品質検査員として充填状況の総評を日本語で2〜3文で述べてください。改善が必要な点があれば具体的に指摘してください。"""
    b64 = pil_to_base64(img)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
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


# --- エリア分割ロジック（食材エリア直接検出）---
def find_vertical_divider(img_bgr, y1, y2, x1, x2):
    """上段エリア内の縦仕切り線のX座標を検出する"""
    h_roi = y2 - y1
    w_roi = x2 - x1
    roi = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    t1 = cv2.inRange(hsv, np.array([0,  60, 40]),  np.array([15, 255, 180]))
    t2 = cv2.inRange(hsv, np.array([165,60, 40]),  np.array([180,255, 180]))
    tray = cv2.bitwise_or(t1, t2)
    col_d = np.sum(tray > 0, axis=0) / h_roi
    # x=25%〜55%の範囲で最大密度の列を仕切りとする
    s = int(w_roi * 0.25)
    e = int(w_roi * 0.55)
    if e <= s:
        return x1 + w_roi // 2
    local_peak = int(np.argmax(col_d[s:e]))
    return x1 + s + local_peak

def detect_bento_areas(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    tray1 = cv2.inRange(hsv, np.array([0,  60, 40]),  np.array([15, 255, 180]))
    tray2 = cv2.inRange(hsv, np.array([165,60, 40]),  np.array([180,255, 180]))
    tray_mask = cv2.bitwise_or(tray1, tray2)
    kernel = np.ones((20,20), np.uint8)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, kernel)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        filled = np.zeros_like(tray_mask)
        cv2.drawContours(filled, [contours[0]], -1, 255, -1)
        food_mask = cv2.bitwise_and(filled, cv2.bitwise_not(tray_mask))
        food_contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = h * w * 0.03
        food_contours = [c for c in food_contours if cv2.contourArea(c) > min_area]
        food_contours = sorted(food_contours, key=cv2.contourArea, reverse=True)[:4]

        # 4エリア検出成功
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
                return {
                    "上左（小おかず）": (tl['y1'], tl['y2'], tl['x1'], tl['x2']),
                    "上右（大おかず）": (tr['y1'], tr['y2'], tr['x1'], br['x2']),
                    "下右（小おかず）": (br['y1'], br['y2'], br['x1'], br['x2']),
                }

        # 3エリア検出：欠けているエリアを補完
        if len(food_contours) == 3:
            boxes = []
            for c in food_contours:
                fx, fy, fw, fh = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                boxes.append({'x1':fx,'y1':fy,'x2':fx+fw,'y2':fy+fh,'cx':fx+fw/2,'cy':fy+fh/2,'area':area})

            # ごはん = 最大面積
            rice = max(boxes, key=lambda b: b['area'])
            rest = [b for b in boxes if b != rice]

            # x・y座標で分類
            tl = tr = br = None
            for b in rest:
                cx_r = b['cx'] / w
                cy_r = b['cy'] / h
                if cy_r < 0.5 and cx_r < 0.5:
                    tl = b  # 上左
                elif cy_r < 0.5:
                    tr = b  # 上右
                else:
                    br = b  # 下右

            # ごはんの位置を参考に上段y範囲・右端を補完
            rice_x2 = rice['x2']
            top_y1 = (tl or tr)['y1'] if (tl or tr) else int(h * 0.06)
            top_y2 = (tl or tr)['y2'] if (tl or tr) else int(h * 0.46)
            bot_y1 = rice['y1']
            bot_y2 = rice['y2']

            # 上左が未検出→上右のx1より左を上左として補完
            if tl is None and tr is not None:
                tl = {'y1': top_y1, 'y2': top_y2, 'x1': rice['x1'], 'x2': tr['x1']}

            # 下右が未検出→ごはんのx2より右を下右として補完
            if br is None:
                br_x1 = rice['x2']
                br_x2 = (tr or tl)['x2'] if (tr or tl) else int(w * 0.90)
                br = {'y1': bot_y1, 'y2': bot_y2, 'x1': br_x1, 'x2': br_x2}

            # 上右が未検出→上左のx2から右端まで
            if tr is None and tl is not None:
                tr = {'y1': top_y1, 'y2': top_y2, 'x1': tl['x2'], 'x2': br['x2']}

            if tl and tr and br:
                def safe(v): return int(v) if v is not None else 0
                return {
                    "上左（小おかず）": (safe(tl['y1']), safe(tl['y2']), safe(tl['x1']), safe(tl['x2'])),
                    "上右（大おかず）": (safe(tr['y1']), safe(tr['y2']), safe(tr['x1']), safe(br['x2'])),
                    "下右（小おかず）": (safe(br['y1']), safe(br['y2']), safe(br['x1']), safe(br['x2'])),
                }

    # フォールバック（固定比率）
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
        "下右（小おかず）": (h_split_bot, y2,          v_bot,  x2_bot),
    }


def draw_results_on_image(img_pil: Image.Image, areas: dict, results: list) -> Image.Image:
    result_map = {r["name"]: r for r in results}
    w, h = img_pil.size
    font_size = max(20, int(w * 0.035))
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "c:/Windows/Fonts/arialbd.ttf",
    ]
    font = ImageFont.load_default(size=font_size)
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except:
                pass

    # 半透明オーバーレイを一括で作成
    output = img_pil.convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    for name, (y1, y2, x1, x2) in areas.items():
        if name == "下左（ごはん）":
            continue
        r = result_map.get(name, {})
        pct = r.get("emptiness_pct", 0)
        if pct < 15:
            color = (46, 204, 113); alpha = 40
        elif pct < 30:
            color = (243, 156, 18); alpha = 50
        else:
            color = (231, 76, 60); alpha = 60
        ov_draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha))

    output = Image.alpha_composite(output, overlay).convert('RGB')
    draw = ImageDraw.Draw(output, 'RGBA')

    line_w = max(4, int(w * 0.005))
    for name, (y1, y2, x1, x2) in areas.items():
        if name == "下左（ごはん）":
            continue
        r = result_map.get(name, {})
        pct = r.get("emptiness_pct", 0)
        if pct < 15:
            color = (46, 204, 113)
        elif pct < 30:
            color = (243, 156, 18)
        else:
            color = (231, 76, 60)

        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=line_w)

        # テキストをエリア中央に表示
        text = f"{pct:.1f}%"
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # textbboxのオフセットを正確に考慮
        bbox = draw.textbbox((0, 0), text, font=font)
        bx0, by0, bx1, by1 = bbox
        tw = bx1 - bx0
        th = by1 - by0
        # 描画位置：中央に合わせてオフセット補正
        tx = cx - tw // 2 - bx0
        ty = cy - th // 2 - by0
        # 背景
        pad = max(14, int(font_size * 0.45))
        draw.rectangle(
            [cx - tw//2 - pad, cy - th//2 - pad,
             cx + tw//2 + pad, cy + th//2 + pad],
            fill=(0, 0, 0, 200)
        )
        # テキスト
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

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
                # ご飯エリアは表示しない
                if nm == "下左（ごはん）":
                    continue
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

        progress_bar.progress(10, text="トレーを検出中...")
        areas = detect_bento_areas(img_bgr)

        results = []
        total_areas = len(areas)

        for i, (name, (y1, y2, x1, x2)) in enumerate(areas.items()):
            progress_bar.progress(
                20 + int(60 * i / total_areas),
                text=f"Claude Vision で「{name}」を解析中... ({i+1}/{total_areas})"
            )
            # ご飯エリアは計測対象外
            if name == "下左（ごはん）":
                results.append({
                    "name": name,
                    "emptiness_pct": 0.0,
                    "confidence": "high",
                    "reason": "計測対象外",
                })
                continue
            # 座標バリデーション
            img_w, img_h = img_orig.size
            x1c = max(0, min(int(x1), img_w-1))
            y1c = max(0, min(int(y1), img_h-1))
            x2c = max(x1c+1, min(int(x2), img_w))
            y2c = max(y1c+1, min(int(y2), img_h))
            roi = img_orig.crop((x1c, y1c, x2c, y2c))
            analysis = analyze_area_with_claude(client, roi, name)
            results.append({
                "name": name,
                "emptiness_pct": analysis["emptiness_pct"],
                "confidence": analysis["confidence"],
                "reason": analysis["reason"],
            })

        progress_bar.progress(85, text="全体評価を生成中...")
        # 平均・PASS判定はご飯除外
        target_results = [r for r in results if r["name"] != "下左（ごはん）"]
        avg_pct = np.mean([r["emptiness_pct"] for r in target_results]) if target_results else 0.0
        is_pass = avg_pct < 20.0 and all(r["emptiness_pct"] < 30.0 for r in target_results)

        ai_comment = analyze_overall_with_claude(client, img_orig, target_results)

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
                ✓ エリア自動検出 &nbsp;|&nbsp; ✓ AI視覚判定 &nbsp;|&nbsp; ✓ 高精度解析
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
