import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image


# =========================
# 画像処理
# =========================
def largest_contour_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask


def food_mask_from_hsv(
    img_bgr: np.ndarray,
    bento_mask: np.ndarray,
    rice_v_min: int = 180,
    rice_s_max: int = 60,
    close_holes_kernel: int = 5,
    close_holes_iter: int = 2,
    fill_micro_gaps_kernel: int = 15,
    fill_micro_gaps_iter: int = 1,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    s_blur = cv2.GaussianBlur(s, (7, 7), 0)
    _, color_mask = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    rice_mask = cv2.inRange(hsv, (0, 0, rice_v_min), (180, rice_s_max, 255))

    combined = cv2.bitwise_or(color_mask, rice_mask)
    combined = cv2.bitwise_and(combined, combined, mask=bento_mask)

    k1 = np.ones((close_holes_kernel, close_holes_kernel), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k1, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k1, iterations=close_holes_iter)

    k2 = np.ones((fill_micro_gaps_kernel, fill_micro_gaps_kernel), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k2, iterations=fill_micro_gaps_iter)

    return combined


def calc_empty_ratio_and_debug(img_bgr: np.ndarray, params: Dict) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bento_mask = largest_contour_mask(gray)
    if bento_mask is None:
        raise ValueError("Bento container not detected (外枠検出に失敗)。")

    food_mask = food_mask_from_hsv(
        img_bgr, bento_mask,
        rice_v_min=params["rice_v_min"],
        rice_s_max=params["rice_s_max"],
        close_holes_kernel=params["close_holes_kernel"],
        close_holes_iter=params["close_holes_iter"],
        fill_micro_gaps_kernel=params["fill_micro_gaps_kernel"],
        fill_micro_gaps_iter=params["fill_micro_gaps_iter"],
    )

    bento_area = int(np.count_nonzero(bento_mask))
    food_area = int(np.count_nonzero(food_mask))
    empty_area = max(bento_area - food_area, 0)
    empty_ratio = empty_area / bento_area * 100.0
    empty_mask = cv2.bitwise_and(255 - food_mask, 255 - food_mask, mask=bento_mask)

    return float(empty_ratio), bento_mask, food_mask, empty_mask


def judge(empty_ratio: float, ok_th: float, warn_th: float) -> str:
    if empty_ratio < ok_th: return "OK"
    elif empty_ratio < warn_th: return "注意"
    else: return "NG"


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def mask_to_preview(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    rgb = bgr_to_rgb(img_bgr)
    m = (mask > 0).astype(np.uint8)
    overlay = rgb.copy()
    overlay[m == 1] = np.clip(overlay[m == 1] + 70, 0, 255)
    return (rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)


def read_upload_to_bgr(uploaded_file) -> np.ndarray:
    data = uploaded_file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("画像読み込みに失敗しました")
    return img


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def build_zip(results_df: pd.DataFrame, debug_images: Dict[str, Dict[str, np.ndarray]]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("result.csv", df_to_csv_bytes(results_df))
        for fname, imgs in debug_images.items():
            stem = Path(fname).stem
            for key, img_rgb in imgs.items():
                ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                if ok: z.writestr(f"debug/{stem}_{key}.png", buf.tobytes())
    return bio.getvalue()


@st.cache_resource
def load_ai_model(model_path: str = "bento_ai.pt"):
    ckpt = torch.load(model_path, map_location="cpu")
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = transforms.Compose([transforms.Resize((ckpt["img_size"], ckpt["img_size"])), transforms.ToTensor()])
    return model, tf, ckpt["class_names"]


def ai_ng_probability_percent(img_bgr: np.ndarray) -> float:
    model, tf, class_names = load_ai_model("bento_ai.pt")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = tf(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx_ng = class_names.index("ng")
    return float(prob[idx_ng] * 100.0)


# =========================
# UI 設定
# =========================
st.set_page_config(page_title="スカスカ弁当 判定管理", layout="wide")

# CSS修正: ロゴが切れないように padding-top を 0.5rem に減らし、全体の位置を調整
st.markdown("""
<style>
    .block-container { 
        padding-top: 3.0rem !important; 
        padding-bottom: 2rem;
        max-width: 1200px; 
    }
    
    .header-box {
        display: flex;
        align-items: center;
        padding-top: 10px;
        margin-bottom: 20px;
    }

    .custom-card {
        background: white;
        border: 1px solid rgba(49,51,63,.15);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,.05);
        margin-bottom: 1rem;
    }
    
    .card-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px; }
    .card-sub { color: #666; font-size: 0.85rem; margin-bottom: 1.2rem; }

    [data-testid="stVerticalBlock"] { gap: 0.8rem !important; }
    
    /* 検索ボックス等を隠す */
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- ヘッダー（ロゴとタイトル） ---
col_head1, col_head2 = st.columns([1.5, 4])
with col_head1:
    try:
        # ロゴを表示（paddingの影響を受けないよう配置）
        st.image("header1_pc.png", width=190)
    except:
        st.write("### GLUG")
with col_head2:
    st.markdown("<h1 style='margin:0; padding-top:5px; font-size: 2.2rem;'>スカスカ弁当 判定管理</h1>", unsafe_allow_html=True)

st.caption("画像をアップロードし、表の行をクリックするとプレビューが切り替わります。")

with st.sidebar:
    st.header("⚙️ 設定")
    ok_th = st.slider("OK 上限（%）", 0.0, 50.0, 20.0, 0.5)
    warn_th = st.slider("注意 上限（%）", 0.0, 80.0, 30.0, 0.5)
    st.divider()
    rice_v_min = st.slider("V（明度）下限", 0, 255, 180)
    rice_s_max = st.slider("S（彩度）上限", 0, 255, 60)
    debug_preview = st.checkbox("詳細画像を表示", value=True)
    overlay_alpha = st.slider("オーバーレイ濃度", 0.0, 0.9, 0.35)

params = {
    "rice_v_min": rice_v_min, "rice_s_max": rice_s_max,
    "close_holes_kernel": 5, "close_holes_iter": 2,
    "fill_micro_gaps_kernel": 15, "fill_micro_gaps_iter": 1,
}

uploads = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not uploads:
    st.info("画像をアップロードすると解析が始まります。")
    st.stop()

# --- 解析フェーズ ---
results = []
debug_images = {}
original_images = {}

for up in uploads:
    try:
        up.seek(0)
        img_bgr = read_upload_to_bgr(up)
        original_images[up.name] = img_bgr.copy()
        
        ratio, b_mask, f_mask, e_mask = calc_empty_ratio_and_debug(img_bgr, params)
        ai_p = ai_ng_probability_percent(img_bgr)
        j = judge(ratio, ok_th, warn_th)
        
        results.append({
            "ファイル名": up.name, 
            "空白率": round(ratio, 2), 
            "AI判定NG率": round(ai_p, 2), 
            "判定": j
        })
        debug_images[up.name] = {
            "overlay": overlay_mask(img_bgr, f_mask, overlay_alpha),
            "bento": mask_to_preview(b_mask), "food": mask_to_preview(f_mask), "empty": mask_to_preview(e_mask)
        }
    except Exception as e:
        results.append({"ファイル名": up.name, "判定": "ERROR", "error": str(e)})

df = pd.DataFrame(results)

# --- 表示レイアウト ---
col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.subheader("📋 判定結果一覧")
    selection = st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        on_select="rerun",  
        selection_mode="single-row"
    )
    
    st.download_button("⬇️ 全判定CSVを保存", data=df_to_csv_bytes(df), file_name="result.csv", mime="text/csv", use_container_width=True)
    if debug_preview:
        st.download_button("⬇️ 解析全データ(ZIP)を保存", data=build_zip(df, debug_images), file_name="debug_data.zip", use_container_width=True)

with col_right:
    selected_rows = selection.get("selection", {}).get("rows", [])
    selected_idx = selected_rows[0] if selected_rows else 0
    selected_file = df.iloc[selected_idx]["ファイル名"]
    row = df.iloc[selected_idx]

    st.markdown(f"""
    <div class="custom-card">
        <div class="card-title">🔍 プレビュー: {selected_file}</div>
        <div class="card-sub">選択中の画像の判定詳細です。</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("空白率", f"{row.get('空白率')}%")
    m2.metric("AI判定NG率", f"{row.get('AI判定NG率')}%")
    m3.metric("判定", row.get("判定"))

    # 単体保存ボタン
    if selected_file in original_images:
        img_to_save = original_images[selected_file]
        is_success, buffer = cv2.imencode(".jpg", img_to_save)
        if is_success:
            st.download_button(
                label=f"📥 この画像をデスクトップに保存 ({row.get('判定')})",
                data=buffer.tobytes(),
                file_name=f"{row.get('判定')}_{selected_file}",
                mime="image/jpeg",
                use_container_width=True
            )

    if debug_preview and selected_file in debug_images:
        imgs = debug_images[selected_file]
        st.image(imgs["overlay"], caption=f"食材オーバーレイ ({selected_file})", use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.image(imgs["bento"], caption="容器外枠", use_container_width=True)
        c2.image(imgs["food"], caption="食材領域", use_container_width=True)
        c3.image(imgs["empty"], caption="空白領域", use_container_width=True)

st.divider()
st.caption("💡 運用：学習用データとして残したい写真は「📥 この画像を保存」ボタンでデスクトップに保存してください。")







