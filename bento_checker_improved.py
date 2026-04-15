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
DEBUG_DIR = "debug_rois"  # デバッグ用ROI保存ディレクトリ
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

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


# =============================================================================
# 【改善①】Claude Vision プロンプトを「充填密度」ベースに全面改訂
# =============================================================================
def analyze_area_with_claude(client, area_img: Image.Image, area_name: str) -> dict:
    b64 = pil_to_base64(area_img)

    prompt = f"""You are a bento (Japanese lunch box) quality inspector. Analyze the image of the "{area_name}" section.

Calculate the EMPTY SPACE PERCENTAGE of this bento section.

## Definition of "empty space"
- EMPTY = visible tray surface (reddish-brown diamond-pattern tray bottom) with NO food on top
- FILLED = any area covered by: food items, paper cups, divider papers, sauce packets, garnishes

## Inspection rules
1. Paper cups (silicone/paper) → count the cup AND its contents as FILLED
2. Green divider paper (バラン) → count as FILLED
3. Small gaps between food items → count as FILLED (not empty)
4. Partial coverage → if >50% of a spot is covered, count as FILLED
5. Only count as EMPTY if the tray bottom is clearly and directly visible with no food above it

## Important
- If food fills the section wall-to-wall with no visible tray bottom → 0% empty
- If food covers most of the section but small tray areas are visible → 5-15%
- If there is a clearly visible unfilled region → 20-50%+

Report the empty percentage as a decimal with 1 decimal place (e.g., 3.2, 8.7, 17.4).
Do NOT round to multiples of 5.

Respond ONLY in JSON:
{{"emptiness_pct": number, "confidence": "high/medium/low", "reason": "reason in 20 chars or less in Japanese"}}"""

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
        reason = data.get("reason", "")
        pct = float(data.get("emptiness_pct", 50))
        print(f"[RESULT] {area_name}: {pct}% - {reason}")
        return {
            "emptiness_pct": pct,
            "confidence": data.get("confidence", "medium"),
            "reason": reason
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


# =============================================================================
# 【改善②】エリア検出ロジック改善
#   - トレー全体のバウンディングボックスを先に確定
#   - 仕切り線（暗い線）を検出して4分割
#   - フォールバックを画像サイズ比率ベースに改善
# =============================================================================
def detect_bento_areas(img_bgr: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]

    # --- Step 1: トレー全体領域を検出 ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 赤茶色のトレー色
    tray1 = cv2.inRange(hsv, np.array([0,  50, 60]),  np.array([15, 255, 200]))
    tray2 = cv2.inRange(hsv, np.array([160, 50, 60]), np.array([180, 255, 200]))
    tray_mask = cv2.bitwise_or(tray1, tray2)

    kernel = np.ones((15, 15), np.uint8)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, kernel)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tray_x1, tray_y1, tray_x2, tray_y2 = int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        tx, ty, tw, th = cv2.boundingRect(largest)
        # トレー全体が十分大きい場合のみ採用
        if tw * th > w * h * 0.3:
            tray_x1, tray_y1 = tx, ty
            tray_x2, tray_y2 = tx + tw, ty + th
            print(f"[TRAY] Detected: ({tray_x1},{tray_y1}) - ({tray_x2},{tray_y2})")

    tray_w = tray_x2 - tray_x1
    tray_h = tray_y2 - tray_y1

    # --- Step 2: 水平仕切り線検出（上段・下段の境界） ---
    # トレー内部のグレースケールでエッジ検出
    roi_full = img_bgr[tray_y1:tray_y2, tray_x1:tray_x2]
    gray = cv2.cvtColor(roi_full, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    # 水平方向の強いエッジ行を探す（仕切り板の位置）
    h_proj = np.sum(edges, axis=1)
    # 上から30%〜70%の範囲で探す
    search_y1 = int(tray_h * 0.30)
    search_y2 = int(tray_h * 0.70)
    h_proj_roi = h_proj[search_y1:search_y2]

    h_divider_local = search_y1 + int(np.argmax(h_proj_roi))
    h_divider = tray_y1 + h_divider_local
    print(f"[DIVIDER-H] Horizontal divider at y={h_divider} (local={h_divider_local})")

    # 検出が不安定な場合はデフォルト値
    if h_divider_local < tray_h * 0.35 or h_divider_local > tray_h * 0.65:
        h_divider = tray_y1 + int(tray_h * 0.50)
        print(f"[DIVIDER-H] Fallback: y={h_divider}")

    # --- Step 3: 垂直仕切り線検出（上段左右の境界） ---
    # 上段ROIで垂直エッジを探す
    top_roi = img_bgr[tray_y1:h_divider, tray_x1:tray_x2]
    gray_top = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
    edges_top = cv2.Canny(gray_top, 30, 100)
    v_proj = np.sum(edges_top, axis=0)

    # 左から20%〜55%の範囲で探す
    search_x1 = int(tray_w * 0.20)
    search_x2 = int(tray_w * 0.55)
    v_proj_roi = v_proj[search_x1:search_x2]

    v_divider_local = search_x1 + int(np.argmax(v_proj_roi))
    v_divider = tray_x1 + v_divider_local
    print(f"[DIVIDER-V] Vertical divider at x={v_divider} (local={v_divider_local})")

    # 検出が不安定な場合はデフォルト値
    if v_divider_local < tray_w * 0.22 or v_divider_local > tray_w * 0.53:
        v_divider = tray_x1 + int(tray_w * 0.38)
        print(f"[DIVIDER-V] Fallback: x={v_divider}")

    # --- Step 4: 下段の垂直仕切り線（ごはん｜小おかず） ---
    bot_roi = img_bgr[h_divider:tray_y2, tray_x1:tray_x2]
    gray_bot = cv2.cvtColor(bot_roi, cv2.COLOR_BGR2GRAY)
    edges_bot = cv2.Canny(gray_bot, 30, 100)
    v_proj_bot = np.sum(edges_bot, axis=0)

    # 左から45%〜75%の範囲で探す（ごはんが大きいので右寄り）
    search_bx1 = int(tray_w * 0.45)
    search_bx2 = int(tray_w * 0.75)
    v_proj_bot_roi = v_proj_bot[search_bx1:search_bx2]

    v_bot_divider_local = search_bx1 + int(np.argmax(v_proj_bot_roi))
    v_bot_divider = tray_x1 + v_bot_divider_local
    print(f"[DIVIDER-V-BOT] Bottom vertical divider at x={v_bot_divider}")

    if v_bot_divider_local < tray_w * 0.47 or v_bot_divider_local > tray_w * 0.73:
        v_bot_divider = tray_x1 + int(tray_w * 0.63)
        print(f"[DIVIDER-V-BOT] Fallback: x={v_bot_divider}")

    # --- Step 5: 座標確定（マージン付き） ---
    margin = 6  # ピクセル単位の内側マージン（仕切り線自体を除外）

    areas = {
        "上左（小おかず）": (
            tray_y1 + margin,
            h_divider - margin,
            tray_x1 + margin,
            v_divider - margin,
        ),
        "上右（大おかず）": (
            tray_y1 + margin,
            h_divider - margin,
            v_divider + margin,
            tray_x2 - margin,
        ),
        "下左（ごはん）": (
            h_divider + margin,
            tray_y2 - margin,
            tray_x1 + margin,
            v_bot_divider - margin,
        ),
        "下右（小おかず）": (
            h_divider + margin,
            tray_y2 - margin,
            v_bot_divider + margin,
            tray_x2 - margin,
        ),
    }

    # 座標クリッピング
    clipped = {}
    for name, (y1, y2, x1, x2) in areas.items():
        y1c = max(0, min(int(y1), h - 1))
        y2c = max(y1c + 10, min(int(y2), h))
        x1c = max(0, min(int(x1), w - 1))
        x2c = max(x1c + 10, min(int(x2), w))
        clipped[name] = (y1c, y2c, x1c, x2c)
        print(f"[AREA] {name}: y={y1c}-{y2c}, x={x1c}-{x2c}")

    return clipped


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

    output = img_pil.convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    for name, (y1, y2, x1, x2) in areas.items():
        if name == "下左（ごはん）":
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1+1, min(x2, w)), max(y1+1, min(y2, h))
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
        reason = r.get("reason", "")
        if pct < 15:
            color = (46, 204, 113)
        elif pct < 30:
            color = (243, 156, 18)
        else:
            color = (231, 76, 60)

        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=line_w)

        text = f"{pct:.1f}%"
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        bx0, by0, bx1, by1 = bbox
        tw = bx1 - bx0
        th = by1 - by0
        tx = cx - tw // 2 - bx0
        ty = cy - th // 2 - by0
        pad = max(14, int(font_size * 0.45))
        draw.rectangle(
            [cx - tw//2 - pad, cy - th//2 - pad,
             cx + tw//2 + pad, cy + th//2 + pad],
            fill=(0, 0, 0, 200)
        )
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

        # 理由テキストも表示
        if reason:
            small_font = ImageFont.load_default(size=max(14, int(font_size * 0.6)))
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        small_font = ImageFont.truetype(fp, max(14, int(font_size * 0.6)))
                        break
                    except:
                        pass
            rbbox = draw.textbbox((0, 0), reason, font=small_font)
            rtw = rbbox[2] - rbbox[0]
            rth = rbbox[3] - rbbox[1]
            ry = cy + th // 2 + pad + 6
            draw.rectangle(
                [cx - rtw//2 - 8, ry,
                 cx + rtw//2 + 8, ry + rth + 6],
                fill=(0, 0, 0, 180)
            )
            draw.text((cx - rtw//2 - rbbox[0], ry + 3), reason, font=small_font, fill=(255, 220, 100, 255))

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
    # ==========================================================================
    # 【改善③】デバッグモード（開発者向け）
    # ==========================================================================
    with st.expander("🔧 デバッグ設定", expanded=False):
        debug_mode = st.checkbox("デバッグモード（切り出しROIを表示）", value=False)

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

        # デバッグ：切り出しROIを表示
        if debug_mode:
            st.subheader("🔍 検出エリア（ROI確認）")
            debug_cols = st.columns(len(areas))
            for i, (name, (y1, y2, x1, x2)) in enumerate(areas.items()):
                roi = img_orig.crop((x1, y1, x2, y2))
                with debug_cols[i]:
                    st.image(roi, caption=f"{name}\n({x2-x1}×{y2-y1}px)", use_container_width=True)

        results = []
        total_areas = len(areas)

        for i, (name, (y1, y2, x1, x2)) in enumerate(areas.items()):
            progress_bar.progress(
                20 + int(60 * i / total_areas),
                text=f"Claude Vision で「{name}」を解析中... ({i+1}/{total_areas})"
            )
            if name == "下左（ごはん）":
                results.append({
                    "name": name,
                    "emptiness_pct": 0.0,
                    "confidence": "high",
                    "reason": "計測対象外",
                })
                continue

            img_w, img_h = img_orig.size
            x1c = max(0, min(int(x1), img_w-1))
            y1c = max(0, min(int(y1), img_h-1))
            x2c = max(x1c+1, min(int(x2), img_w))
            y2c = max(y1c+1, min(int(y2), img_h))
            roi = img_orig.crop((x1c, y1c, x2c, y2c))

            # デバッグ保存
            roi.save(f"{DEBUG_DIR}/{name}.jpg", quality=92)

            analysis = analyze_area_with_claude(client, roi, name)
            results.append({
                "name": name,
                "emptiness_pct": analysis["emptiness_pct"],
                "confidence": analysis["confidence"],
                "reason": analysis["reason"],
            })

        progress_bar.progress(85, text="全体評価を生成中...")
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
