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
REFERENCE_FILE = "reference_empty.jpg"
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
.ref-status-ok { background: #1f3a2a; color: #7fdba0 !important; padding: 8px 12px; border-radius: 8px; font-size: 0.75rem; }
.ref-status-ng { background: #3a2a1f; color: #e0a070 !important; padding: 8px 12px; border-radius: 8px; font-size: 0.75rem; }
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


# --- 座標クランプ ヘルパー ---
def clamp_rect(y1, y2, x1, x2, w: int, h: int):
    """座標を [0, w-1] × [0, h-1] にクランプ。退化した矩形は None を返す。"""
    try:
        x1i = max(0, min(int(x1), w - 1))
        y1i = max(0, min(int(y1), h - 1))
        x2i = max(0, min(int(x2), w - 1))
        y2i = max(0, min(int(y2), h - 1))
    except (TypeError, ValueError):
        return None
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i


# --- リファレンス画像の管理 ---
def load_reference_image():
    if os.path.exists(REFERENCE_FILE):
        try:
            return Image.open(REFERENCE_FILE).convert("RGB")
        except Exception:
            return None
    return None


def save_reference_image(img_pil: Image.Image):
    max_side = 1600
    w, h = img_pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img_pil.save(REFERENCE_FILE, quality=92)


def delete_reference_image():
    if os.path.exists(REFERENCE_FILE):
        try:
            os.remove(REFERENCE_FILE)
        except Exception:
            pass


# --- トレー色のリファレンス学習 (LAB 色空間) ---
@st.cache_data(show_spinner=False)
def get_reference_tray_lab(ref_mtime: float):
    """リファレンス画像から「トレー底面色」の LAB 平均値と分散を学習する。
    ref_mtime は cache 無効化用（ファイルが更新されると自動再計算）。"""
    ref_img = load_reference_image()
    if ref_img is None:
        return None
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)

    ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(ref_hsv, np.array([0, 60, 30]), np.array([15, 255, 230]))
    m2 = cv2.inRange(ref_hsv, np.array([165, 60, 30]), np.array([180, 255, 230]))
    tray_candidate = cv2.bitwise_or(m1, m2)

    kernel = np.ones((5, 5), np.uint8)
    tray_candidate = cv2.morphologyEx(tray_candidate, cv2.MORPH_OPEN, kernel)

    if np.sum(tray_candidate > 0) < 1000:
        return None

    lab_pixels = ref_lab[tray_candidate > 0]
    lab_mean = np.mean(lab_pixels, axis=0).astype(np.float32)
    lab_std = np.std(lab_pixels, axis=0).astype(np.float32)
    return {
        "mean": tuple(float(x) for x in lab_mean),
        "std": tuple(float(x) for x in lab_std),
    }


def get_reference_tray_lab_current():
    """現在のリファレンスファイルから LAB 統計値を取得（mtime でキャッシュ）"""
    if not os.path.exists(REFERENCE_FILE):
        return None
    mtime = os.path.getmtime(REFERENCE_FILE)
    return get_reference_tray_lab(mtime)


# --- 空き率計算（OpenCV ベース）---
def compute_emptiness_cv(roi_pil: Image.Image, area_name: str,
                         ref_lab_stats: dict = None,
                         tolerance: float = 25.0) -> dict:
    """
    ROI 内で「トレー色（赤い菱形模様）」のピクセル比率を計算。

    - リファレンス学習済 → LAB 色空間での距離ベース判定
    - 未学習 → HSV 範囲ベース判定（フォールバック）

    小おかずと大おかずで、食材/カップ間の構造的な小ギャップを吸収するため
    食材マスクに MORPH_CLOSE を適用する。過剰補正しないよう kernel は控えめ。
    さらに大おかずではコンパートメント壁（リム）を狭く除外する。
    """
    roi_bgr = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    h_roi, w_roi = roi_bgr.shape[:2]

    if ref_lab_stats is not None:
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_mean = np.array(ref_lab_stats["mean"], dtype=np.float32)
        diff = roi_lab - ref_mean
        weights = np.array([0.5, 1.2, 1.2], dtype=np.float32)
        weighted = diff * weights
        dist = np.sqrt(np.sum(weighted * weighted, axis=2))
        tray_mask = (dist < tolerance).astype(np.uint8) * 255
    else:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 90, 50]),   np.array([12, 230, 180]))
        m2 = cv2.inRange(hsv, np.array([168, 90, 50]), np.array([180, 230, 180]))
        tray_mask = cv2.bitwise_or(m1, m2)

    # 小さな孤立ノイズ除去
    kernel_small = np.ones((3, 3), np.uint8)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_OPEN, kernel_small)

    # --- 構造ギャップを埋める MORPH_CLOSE ---
    # 本物の空きは残すため kernel は控えめに
    if "小おかず" in area_name:
        # 小おかずはカップ同士の構造ギャップが大きい
        bridge_size = max(21, int(min(h_roi, w_roi) * 0.12))
    elif "大おかず" in area_name:
        # 食材間の小〜中ギャップのみ吸収（≒ 5% 程度）
        bridge_size = max(9, int(min(h_roi, w_roi) * 0.05))
    else:
        bridge_size = 0

    if bridge_size > 0:
        if bridge_size % 2 == 0:
            bridge_size += 1
        food_mask = cv2.bitwise_not(tray_mask)
        kernel_bridge = np.ones((bridge_size, bridge_size), np.uint8)
        food_bridged = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel_bridge)
        tray_mask = cv2.bitwise_not(food_bridged)

    # --- 大おかず: コンパートメント壁（リム）を狭く除外 ---
    # 外縁に必ず写るリムがノイズ的に空き判定を膨らませるのを防ぐ
    total_px = tray_mask.size
    if "大おかず" in area_name:
        pad_x     = max(3, int(w_roi * 0.025))
        pad_y_top = max(3, int(h_roi * 0.025))
        pad_y_bot = max(3, int(h_roi * 0.025))
        tray_mask[:pad_y_top, :] = 0
        tray_mask[-pad_y_bot:, :] = 0
        tray_mask[:, :pad_x] = 0
        tray_mask[:, -pad_x:] = 0
        inner_w = max(1, w_roi - 2 * pad_x)
        inner_h = max(1, h_roi - pad_y_top - pad_y_bot)
        total_px = inner_w * inner_h

    tray_px = int(np.sum(tray_mask > 0))
    pct = (tray_px / total_px * 100.0) if total_px > 0 else 0.0
    pct = max(pct, 0.0)

    return {
        "emptiness_pct": round(pct, 1),
        "confidence": "high",
        "reason": f"CV検出 {tray_px:,}/{total_px:,}px",
        "tray_mask": tray_mask,
    }


def overlay_tray_mask(img_pil: Image.Image, areas: dict, area_masks: dict) -> Image.Image:
    """検出された空き（トレー露出）ピクセルを黄色で可視化する"""
    w, h = img_pil.size
    output = img_pil.convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))

    for name, (y1, y2, x1, x2) in areas.items():
        if name == "下左（ごはん）":
            continue
        mask = area_masks.get(name)
        if mask is None:
            continue
        clamped = clamp_rect(y1, y2, x1, x2, w, h)
        if clamped is None:
            continue
        x1i, y1i, x2i, y2i = clamped
        area_w = x2i - x1i
        area_h = y2i - y1i
        mh, mw = mask.shape[:2]
        if (mw, mh) != (area_w, area_h):
            try:
                mask = cv2.resize(mask, (area_w, area_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                continue
        rgba = np.zeros((area_h, area_w, 4), dtype=np.uint8)
        rgba[mask > 0] = [255, 230, 0, 170]
        patch = Image.fromarray(rgba, 'RGBA')
        overlay.paste(patch, (x1i, y1i), patch)

    return Image.alpha_composite(output, overlay).convert('RGB')


# --- Claude Vision による空き率推定（高精度）---
def compute_emptiness_vision(client, roi_pil: Image.Image, area_name: str) -> dict:
    """
    Claude Vision (Opus 4.7) を使って ROI の空き率を意味的に判定する。

    CV によるピクセル比較では、食材間の隙間・仕切り壁・リム・影を
    「空き」として誤検出するが、Vision は対象を意味的に理解するため
    「食材で覆われていないトレー露出部」を精度良く推定できる。

    返り値: {"emptiness_pct": float, "reason": str} / 失敗時 None
    """
    try:
        b64 = pil_to_base64(roi_pil)
        threshold_txt = "15%" if "大おかず" in area_name else "20%"
        prompt = f"""この画像はお弁当の 1 つのコンパートメント（{area_name}エリア）の切り出しです。
このエリア内で、**赤いトレー底面が食材やカップで覆われずに露出している部分の割合**を 0〜100 の数値（%）で精密に推定してください。

判定ルール:
- 食材本体・カップライナー・葉飾り（バラン）・卵焼き等は「充填済み」
- コンパートメントを区切る壁（リム）は「構造部分」なのでカウントしない
- 赤いトレー底面が見えている部分のみを「空き」としてカウント
- 端の影や細いリムは除外し、実質的な空きスペースのみを評価

判定閾値: このエリアは空き率 {threshold_txt} 以上で NG です。

必ず以下の JSON 形式のみで回答してください（説明文や前置きは一切不要）:
{{"emptiness_pct": 数値, "reason": "簡潔な根拠(20字以内)"}}"""

        msg = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=400,
            thinking={"type": "adaptive"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        # thinking ブロックをスキップして text ブロックを取得
        text = ""
        for block in msg.content:
            if getattr(block, "type", "") == "text":
                text = block.text
                break
        if not text:
            return None

        import re
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        pct = float(data.get("emptiness_pct", 0))
        pct = max(0.0, min(100.0, pct))
        return {
            "emptiness_pct": round(pct, 1),
            "reason": f"Vision: {data.get('reason', '')}",
        }
    except Exception as e:
        print(f"[Vision ERROR {area_name}] {type(e).__name__}: {e}")
        return None


# --- AI 総評（全体コメント用のみ Claude を使用）---
def analyze_overall_with_claude(client, img: Image.Image, results: list) -> str:
    summary = "\n".join([f"- {r['name']}: 空き率{r['emptiness_pct']:.1f}%" for r in results])
    prompt = f"""お弁当の品質検査結果です（画像処理で計算された空き率）：
{summary}

※空き率は、各エリア内で「トレーの赤い底面が露出しているピクセル比率」を画像処理で算出した客観値です。
※判定基準：大おかずエリアは15%以上で NG、小おかずエリアは20%以上で NG。

品質検査員として充填状況の総評を日本語で2〜3文で述べてください。改善が必要な点があれば具体的に指摘してください。"""
    b64 = pil_to_base64(img)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
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


# --- エリア分割ロジック ---
def build_tray_mask_rough(img_bgr: np.ndarray) -> np.ndarray:
    """エリア検出用の赤色マスク（広めの HSV 範囲）"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,   60, 40]),  np.array([15,  255, 220]))
    m2 = cv2.inRange(hsv, np.array([165, 60, 40]),  np.array([180, 255, 220]))
    return cv2.bitwise_or(m1, m2)


def _fallback_areas(w: int, h: int) -> dict:
    """全ての検出が失敗した場合の固定比率フォールバック"""
    y1          = int(h * 0.10)
    h_split     = int(h * 0.46)
    x1_top      = int(w * 0.10)
    x2_top      = int(w * 0.92)
    v_top       = int(w * 0.36)
    h_split_bot = int(h * 0.50)
    y2          = int(h * 0.92)
    x2_bot      = int(w * 0.92)
    v_bot       = int(w * 0.63)
    return {
        "上左（小おかず）": (y1,          h_split,     x1_top, v_top),
        "上右（大おかず）": (y1,          h_split,     v_top,  x2_top),
        "下右（小おかず）": (h_split_bot, y2,          v_bot,  x2_bot),
    }


def detect_bento_areas(img_bgr: np.ndarray) -> dict:
    """
    標準レイアウト（上段=小・大、下段=ごはん・小）想定で 3 エリアを返す。

    手順:
      1. トレー全体の外周 bbox を取得
      2. トレー左下の「ごはん」を厳密な HSV + 領域限定で検出
         （S<55, V>160 かつ トレー左 65% × 下 60% のみ）
         - 他エリアの白系食材（玉子焼き、衣付きフライ）との混同を防ぐ
      3. ごはん bbox の妥当性チェック（位置・アスペクト比）
      4. ごはん上端 = 上下段境界、ごはん右端 = 下右エリア左端
      5. 上段の左右分割点は、上段内の食材輪郭の水平ギャップから推定
      6. どこかで失敗したら輪郭ベース/固定比率にフォールバック
    """
    h, w = img_bgr.shape[:2]

    # Step 1: トレー全体 bbox
    tray_mask_rough = build_tray_mask_rough(img_bgr)
    kernel_big = np.ones((25, 25), np.uint8)
    tray_closed = cv2.morphologyEx(tray_mask_rough, cv2.MORPH_CLOSE, kernel_big)
    tray_closed = cv2.morphologyEx(tray_closed, cv2.MORPH_OPEN, kernel_big)

    tray_contours, _ = cv2.findContours(tray_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not tray_contours:
        return _fallback_areas(w, h)

    tray_c = max(tray_contours, key=cv2.contourArea)
    if cv2.contourArea(tray_c) < w * h * 0.15:
        return _fallback_areas(w, h)

    tx, ty, tw, th = cv2.boundingRect(tray_c)

    inside_tray = np.zeros_like(tray_closed)
    cv2.drawContours(inside_tray, [tray_c], -1, 255, -1)

    # Step 2: ごはん領域検出（厳密条件 + 領域制限）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    rice_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 55, 255]))
    rice_mask = cv2.bitwise_and(rice_mask, inside_tray)

    # トレー左 65% × 下 60% に限定（ごはんは左下が定位置）
    bottom_left_region = np.zeros_like(rice_mask)
    y_start = ty + int(th * 0.40)
    x_end   = tx + int(tw * 0.65)
    bottom_left_region[y_start: ty + th, tx: x_end] = 255
    rice_mask = cv2.bitwise_and(rice_mask, bottom_left_region)

    rk = np.ones((15, 15), np.uint8)
    rice_mask = cv2.morphologyEx(rice_mask, cv2.MORPH_CLOSE, rk)
    rice_mask = cv2.morphologyEx(rice_mask, cv2.MORPH_OPEN, rk)

    rice_contours, _ = cv2.findContours(rice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rice_contours = [c for c in rice_contours if cv2.contourArea(c) > tw * th * 0.06]

    # Step 3: 妥当性チェック
    rice_ok = False
    rx = ry = rw = rh = 0
    if rice_contours:
        rice = max(rice_contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(rice)
        aspect = rw / max(rh, 1)
        rice_rel_top  = (ry - ty) / max(th, 1)
        rice_rel_left = (rx - tx) / max(tw, 1)
        # ごはんは横長気味 + トレー左下に位置
        if aspect >= 0.7 and rice_rel_top >= 0.30 and rice_rel_left <= 0.15:
            rice_ok = True

    if rice_ok:
        inset = max(2, int(min(tw, th) * 0.015))

        # Step 4: 上下段境界（ごはん上端に合わせる）
        up_y1 = ty + inset
        up_y2 = ry
        lo_y1 = ry
        lo_y2 = ty + th - inset

        # Step 5: 上段の左右分割点
        split_x = tx + int(tw * 0.37)  # 既定値（小:大 = 37:63）

        food_mask_full = cv2.bitwise_and(inside_tray, cv2.bitwise_not(tray_mask_rough))
        upper_food = food_mask_full.copy()
        upper_food[:up_y1, :] = 0
        upper_food[up_y2:, :] = 0
        upper_food[:, :tx] = 0
        upper_food[:, tx + tw:] = 0
        uk = np.ones((9, 9), np.uint8)
        upper_food = cv2.morphologyEx(upper_food, cv2.MORPH_OPEN, uk)

        uc, _ = cv2.findContours(upper_food, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        uc_bbox = [cv2.boundingRect(c) for c in uc if cv2.contourArea(c) > tw * th * 0.012]
        if len(uc_bbox) >= 2:
            uc_bbox_sorted = sorted(uc_bbox, key=lambda b: b[0] + b[2] / 2)
            leftmost = uc_bbox_sorted[0]
            second   = uc_bbox_sorted[1]
            left_right_edge  = leftmost[0] + leftmost[2]
            second_left_edge = second[0]
            candidate = (left_right_edge + second_left_edge) / 2
            low_lim  = tx + tw * 0.22
            high_lim = tx + tw * 0.50
            split_x = int(max(low_lim, min(high_lim, candidate)))

        # Step 6: 下右エリア
        lr_x1 = rx + rw
        lr_x2 = tx + tw - inset
        if lr_x2 - lr_x1 < tw * 0.15:
            lr_x1 = tx + int(tw * 0.63)
            lr_x2 = tx + tw - inset

        return {
            "上左（小おかず）": (int(up_y1), int(up_y2), int(tx + inset), int(split_x)),
            "上右（大おかず）": (int(up_y1), int(up_y2), int(split_x),    int(tx + tw - inset)),
            "下右（小おかず）": (int(lo_y1), int(lo_y2), int(lr_x1),      int(lr_x2)),
        }

    # --- ごはん検出失敗: 輪郭ベースのフォールバック ---
    kernel = np.ones((20, 20), np.uint8)
    tm = cv2.morphologyEx(tray_mask_rough, cv2.MORPH_CLOSE, kernel)
    tm = cv2.morphologyEx(tm, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        filled = np.zeros_like(tm)
        cv2.drawContours(filled, [contours[0]], -1, 255, -1)
        food_mask = cv2.bitwise_and(filled, cv2.bitwise_not(tm))
        food_contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = h * w * 0.03
        food_contours = [c for c in food_contours if cv2.contourArea(c) > min_area]
        food_contours = sorted(food_contours, key=cv2.contourArea, reverse=True)[:4]

        if len(food_contours) == 4:
            boxes = []
            for c in food_contours:
                fx, fy, fw, fh = cv2.boundingRect(c)
                boxes.append({'x1': fx, 'y1': fy, 'x2': fx + fw, 'y2': fy + fh,
                              'cx': fx + fw / 2, 'cy': fy + fh / 2})
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

    return _fallback_areas(w, h)


# --- 判定閾値 ---
THRESHOLD_OOKAZU = 15.0   # 大おかず: 15%以上で NG
THRESHOLD_KOOKAZU = 20.0  # 小おかず: 20%以上で NG


def pct_color(pct: float, area_name: str) -> tuple:
    """空き率に応じた色（RGBタプル）を返す"""
    threshold = THRESHOLD_OOKAZU if "大おかず" in area_name else THRESHOLD_KOOKAZU
    warn_zone = threshold * 0.7
    if pct < warn_zone:
        return (46, 204, 113)  # green
    elif pct < threshold:
        return (243, 156, 18)  # yellow
    else:
        return (231, 76, 60)   # red


def draw_results_on_image(img_pil: Image.Image, areas: dict, results: list, area_masks: dict = None) -> Image.Image:
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

    if area_masks:
        base = overlay_tray_mask(img_pil, areas, area_masks)
    else:
        base = img_pil

    output = base.convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    clamped_areas = {}
    for name, (y1, y2, x1, x2) in areas.items():
        if name == "下左（ごはん）":
            continue
        clamped = clamp_rect(y1, y2, x1, x2, w, h)
        if clamped is None:
            continue
        clamped_areas[name] = clamped

    for name, (x1, y1, x2, y2) in clamped_areas.items():
        r = result_map.get(name, {})
        pct = r.get("emptiness_pct", 0)
        color = pct_color(pct, name)
        threshold = THRESHOLD_OOKAZU if "大おかず" in name else THRESHOLD_KOOKAZU
        alpha = 30 if pct < threshold * 0.7 else (40 if pct < threshold else 50)
        try:
            ov_draw.rectangle([x1, y1, x2, y2], fill=(color[0], color[1], color[2], alpha))
        except Exception as e:
            print(f"[DRAW ERROR fill {name}] {e}")

    output = Image.alpha_composite(output, overlay).convert('RGB')
    draw = ImageDraw.Draw(output, 'RGBA')

    line_w = max(4, int(w * 0.005))
    for name, (x1, y1, x2, y2) in clamped_areas.items():
        r = result_map.get(name, {})
        pct = r.get("emptiness_pct", 0)
        color = pct_color(pct, name)

        try:
            draw.rectangle([x1, y1, x2, y2], outline=(color[0], color[1], color[2], 255), width=line_w)
        except Exception as e:
            print(f"[DRAW ERROR outline {name}] {e}")
            continue

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

        bg_x1 = max(0, min(cx - tw//2 - pad, w - 1))
        bg_y1 = max(0, min(cy - th//2 - pad, h - 1))
        bg_x2 = max(bg_x1 + 1, min(cx + tw//2 + pad, w - 1))
        bg_y2 = max(bg_y1 + 1, min(cy + th//2 + pad, h - 1))
        try:
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))
            draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
        except Exception as e:
            print(f"[DRAW ERROR text {name}] {e}")

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
for key, val in [('last_processed_file', None), ('selected_idx', None),
                 ('show_ref_upload', False), ('color_tolerance', 25.0)]:
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
    st.markdown('<div style="font-size:0.7rem; color:#888; letter-spacing:2px; margin-bottom:6px;">REFERENCE (空容器)</div>', unsafe_allow_html=True)

    ref_img = load_reference_image()
    if ref_img is not None:
        ref_stats = get_reference_tray_lab_current()
        if ref_stats is not None:
            lm = ref_stats["mean"]
            st.markdown(f'<div class="ref-status-ok">✓ 空容器登録済 / LAB学習済<br><span style="font-size:0.65rem;opacity:.8;">L={lm[0]:.0f} A={lm[1]:.0f} B={lm[2]:.0f}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ref-status-ng">✓ 登録済（色抽出失敗・HSVフォールバック）</div>', unsafe_allow_html=True)
        st.image(ref_img, use_container_width=True)
        col_re, col_del = st.columns(2)
        with col_re:
            if st.button("更新", key="ref_update", use_container_width=True):
                st.session_state.show_ref_upload = True
                st.rerun()
        with col_del:
            if st.button("削除", key="ref_delete", use_container_width=True):
                delete_reference_image()
                st.session_state.show_ref_upload = False
                st.rerun()
    else:
        st.markdown('<div class="ref-status-ng">未登録</div>', unsafe_allow_html=True)
        st.session_state.show_ref_upload = True

    if st.session_state.show_ref_upload:
        ref_up = st.file_uploader(
            "空容器の写真を登録",
            type=['jpg', 'jpeg', 'png'],
            key="ref_uploader",
            help="食材を入れる前の空容器を撮影して登録してください"
        )
        if ref_up is not None:
            try:
                ref_pil = Image.open(ref_up).convert("RGB")
                save_reference_image(ref_pil)
                st.session_state.show_ref_upload = False
                st.success("リファレンスを登録しました")
                st.rerun()
            except Exception as e:
                st.error(f"登録失敗: {e}")

    st.markdown('<hr style="border-color: #333; margin: 12px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#888; letter-spacing:2px; margin-bottom:4px;">色許容度 (TOLERANCE)</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem; color:#666; margin-bottom:6px;">小さい→厳密判定 / 大きい→甘い判定</div>', unsafe_allow_html=True)
    st.session_state.color_tolerance = st.slider(
        "tolerance", min_value=10.0, max_value=45.0,
        value=float(st.session_state.color_tolerance), step=1.0,
        label_visibility="collapsed"
    )

    st.markdown('<hr style="border-color: #333; margin: 12px 0;">', unsafe_allow_html=True)
    st.markdown(f'''<div style="font-size:0.7rem; color:#888; letter-spacing:2px; margin-bottom:6px;">判定閾値 (NG)</div>
    <div style="font-size:0.78rem; color:#ccc; line-height:1.7;">
      大おかず: <b>≥ {THRESHOLD_OOKAZU:.0f}%</b><br>
      小おかず: <b>≥ {THRESHOLD_KOOKAZU:.0f}%</b>
    </div>''', unsafe_allow_html=True)

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
st.markdown('<div class="subtitle-block">CV-Based Filling Analysis</div>', unsafe_allow_html=True)

client = get_anthropic_client()
history = load_shared_history()

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
                    threshold = THRESHOLD_OOKAZU if "大おかず" in nm else THRESHOLD_KOOKAZU
                    if pct < threshold * 0.7:
                        cls = "pass"
                    elif pct < threshold:
                        cls = "warn"
                    else:
                        cls = "fail"
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

        ref_lab_stats = get_reference_tray_lab_current()
        tolerance = float(st.session_state.color_tolerance)

        progress_bar.progress(10, text="トレーを検出中...")
        areas = detect_bento_areas(img_bgr)

        results = []
        area_masks = {}
        total_areas = len(areas)

        for i, (name, (y1, y2, x1, x2)) in enumerate(areas.items()):
            progress_bar.progress(
                20 + int(50 * i / total_areas),
                text=f"画像処理で「{name}」を解析中... ({i+1}/{total_areas})"
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
            clamped = clamp_rect(y1, y2, x1, x2, img_w, img_h)
            if clamped is None:
                results.append({
                    "name": name,
                    "emptiness_pct": 0.0,
                    "confidence": "low",
                    "reason": "座標不正",
                })
                continue
            x1c, y1c, x2c, y2c = clamped
            roi = img_orig.crop((x1c, y1c, x2c, y2c))

            # CV でマスクを生成（黄色オーバーレイ表示用）
            cv_analysis = compute_emptiness_cv(roi, name, ref_lab_stats=ref_lab_stats, tolerance=tolerance)
            area_masks[name] = cv_analysis.pop("tray_mask", None)

            # Vision API で空き率を意味的に判定（高精度）
            vision_analysis = compute_emptiness_vision(client, roi, name)

            if vision_analysis is not None:
                pct = vision_analysis["emptiness_pct"]
                reason = vision_analysis["reason"]
                conf = "high"
            else:
                # Vision 失敗時は CV 結果にフォールバック
                pct = cv_analysis["emptiness_pct"]
                reason = f"CVフォールバック: {cv_analysis['reason']}"
                conf = cv_analysis["confidence"]

            results.append({
                "name": name,
                "emptiness_pct": pct,
                "confidence": conf,
                "reason": reason,
            })

        progress_bar.progress(80, text="全体評価を生成中...")
        target_results = [r for r in results if r["name"] != "下左（ごはん）"]
        avg_pct = np.mean([r["emptiness_pct"] for r in target_results]) if target_results else 0.0

        def area_passes(r):
            pct = r["emptiness_pct"]
            if "大おかず" in r["name"]:
                return pct < THRESHOLD_OOKAZU
            return pct < THRESHOLD_KOOKAZU

        is_pass = all(area_passes(r) for r in target_results)

        ai_comment = analyze_overall_with_claude(client, img_orig, target_results)

        progress_bar.progress(92, text="結果を描画中...")
        try:
            output_pil = draw_results_on_image(img_orig, areas, results, area_masks=area_masks)
        except Exception as e:
            print(f"[DRAW ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            output_pil = img_orig

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
                画像処理で各エリアの空き率を解析します<br><br>
                <span style="font-size:0.75rem; color:#ccc;">
                ✓ エリア自動検出 &nbsp;|&nbsp; ✓ トレー露出検出 &nbsp;|&nbsp; ✓ 決定的判定
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
