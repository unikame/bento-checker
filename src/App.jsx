import { useState, useRef, useCallback, useEffect } from "react";
import cv from "@techstark/opencv-js";

const AREAS = [
  { name: "右上（メイン）", key: "main" },
  { name: "左上（副菜A）", key: "sub1" },
  { name: "右下（副菜B）", key: "sub2" },
];

// サンプル画像から実測した正解座標（参考値）
// 右上（メイン）: y: 0.085-0.433, x: 0.376-0.926
// 左上（副菜A）: y: 0.085-0.434, x: 0.079-0.338
// 右下（副菜B）: y: 0.479-0.894, x: 0.661-0.923

const DEFAULT_CROP_DEFS = [
  [0.085, 0.433, 0.376, 0.926],  // 右上（メイン）: y1, y2, x1, x2
  [0.085, 0.434, 0.140, 0.360],  // 左上（副菜A）
  [0.479, 0.894, 0.661, 0.923],  // 右下（副菜B）
];

// OpenCV.jsの準備を待つ
function waitForOpenCV() {
  return new Promise((resolve) => {
    if (cv && cv.Mat) {
      resolve();
      return;
    }
    // モジュールが初期化されるまで待つ
    const check = setInterval(() => {
      if (cv && cv.Mat) {
        clearInterval(check);
        resolve();
      }
    }, 100);
    setTimeout(() => { clearInterval(check); resolve(); }, 15000);
  });
}

// 保存された座標をlocalStorageから読み込み
function loadSavedCropDefs() {
  try {
    const saved = localStorage.getItem("savedCropDefs");
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length === 3) {
        return parsed;
      }
    }
  } catch {}
  return null;
}

function saveCropDefs(cropDefs) {
  try {
    // 配列のコピー（プロパティを除外）
    const cleaned = cropDefs.map(d => [...d]);
    localStorage.setItem("savedCropDefs", JSON.stringify(cleaned));
  } catch {}
}

// OpenCVが検出した容器外枠を一時保存（デバッグ用）
let lastContainerBox = null;

// 画像を容器の範囲で切り抜いて、新しいFileオブジェクトを返す
async function cropImageToContainer(file) {
  try {
    await waitForOpenCV();
    if (!cv || !cv.Mat) {
      console.warn("[OpenCV] Not loaded, returning original file");
      return { croppedFile: file, success: false };
    }
    console.log("[OpenCV] Detecting container for cropping");

    const bitmap = await createImageBitmap(file);
    const maxSize = 1500;
    const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
    const w = Math.floor(bitmap.width * scale);
    const h = Math.floor(bitmap.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();

    const src = cv.imread(canvas);
    const hsv = new cv.Mat();
    cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);

    const mask1 = new cv.Mat();
    const mask2 = new cv.Mat();
    const redMask = new cv.Mat();
    const low1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 80, 50, 0]);
    const high1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [15, 255, 200, 255]);
    const low2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [160, 80, 50, 0]);
    const high2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [180, 255, 200, 255]);
    cv.inRange(hsv, low1, high1, mask1);
    cv.inRange(hsv, low2, high2, mask2);
    cv.bitwise_or(mask1, mask2, redMask);

    const kernel = cv.Mat.ones(7, 7, cv.CV_8U);
    cv.morphologyEx(redMask, redMask, cv.MORPH_CLOSE, kernel);
    cv.morphologyEx(redMask, redMask, cv.MORPH_OPEN, kernel);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(redMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let containerRect = null;
    for (let i = 0; i < contours.size(); i++) {
      const area = cv.contourArea(contours.get(i));
      if (area > maxArea) {
        maxArea = area;
        containerRect = cv.boundingRect(contours.get(i));
      }
    }

    const cleanup = () => {
      src.delete(); hsv.delete();
      mask1.delete(); mask2.delete(); redMask.delete();
      low1.delete(); high1.delete(); low2.delete(); high2.delete();
      kernel.delete();
      contours.delete(); hierarchy.delete();
    };

    if (!containerRect || maxArea < (w * h) * 0.2) {
      console.warn("[OpenCV] Container not found, using original");
      cleanup();
      return { croppedFile: file, success: false };
    }

    console.log("[OpenCV] Container detected for cropping:", containerRect);

    cleanup();

    // 容器領域を元の画像から切り出し（高解像度のまま）
    const origW = await createImageBitmap(file).then(b => { const w = b.width; b.close(); return w; });
    const origH = await createImageBitmap(file).then(b => { const h = b.height; b.close(); return h; });
    const scaleBack = origW / w;

    const cropX = Math.floor(containerRect.x * scaleBack);
    const cropY = Math.floor(containerRect.y * scaleBack);
    const cropW = Math.floor(containerRect.width * scaleBack);
    const cropH = Math.floor(containerRect.height * scaleBack);

    const origBitmap = await createImageBitmap(file);
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.drawImage(origBitmap, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    origBitmap.close();

    // BlobをFileに変換
    const blob = await new Promise(resolve => cropCanvas.toBlob(resolve, "image/jpeg", 0.92));
    const croppedFile = new File([blob], "cropped.jpg", { type: "image/jpeg" });

    return { croppedFile, success: true };
  } catch (e) {
    console.error("[OpenCV] Crop failed:", e);
    return { croppedFile: file, success: false };
  }
}

// OpenCVで弁当容器の4区画を自動検出
async function detectRegions(file) {
  try {
    await waitForOpenCV();
    if (!cv || !cv.Mat) {
      console.warn("[OpenCV] Not loaded, using default coordinates");
      return DEFAULT_CROP_DEFS;
    }
    console.log("[OpenCV] Loaded, starting detection");

    // 画像をCanvasに読み込む
    const bitmap = await createImageBitmap(file);
    const maxSize = 1200;
    const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
    const w = Math.floor(bitmap.width * scale);
    const h = Math.floor(bitmap.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();

    const src = cv.imread(canvas);
    const hsv = new cv.Mat();
    cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);

    // 赤色マスク（HSVで2つの範囲 + OR）
    const mask1 = new cv.Mat();
    const mask2 = new cv.Mat();
    const redMask = new cv.Mat();
    const low1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 80, 50, 0]);
    const high1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [15, 255, 200, 255]);
    const low2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [160, 80, 50, 0]);
    const high2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [180, 255, 200, 255]);
    cv.inRange(hsv, low1, high1, mask1);
    cv.inRange(hsv, low2, high2, mask2);
    cv.bitwise_or(mask1, mask2, redMask);

    // モルフォロジー処理でノイズ除去 & 穴埋め
    const kernel = cv.Mat.ones(7, 7, cv.CV_8U);
    cv.morphologyEx(redMask, redMask, cv.MORPH_CLOSE, kernel);
    cv.morphologyEx(redMask, redMask, cv.MORPH_OPEN, kernel);

    // 最大の赤い輪郭を容器とみなす
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(redMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let containerRect = null;
    for (let i = 0; i < contours.size(); i++) {
      const area = cv.contourArea(contours.get(i));
      if (area > maxArea) {
        maxArea = area;
        containerRect = cv.boundingRect(contours.get(i));
      }
    }

    // クリーンアップ
    const cleanup = () => {
      src.delete(); hsv.delete();
      mask1.delete(); mask2.delete(); redMask.delete();
      low1.delete(); high1.delete(); low2.delete(); high2.delete();
      kernel.delete();
      contours.delete(); hierarchy.delete();
    };

    if (!containerRect || maxArea < (w * h) * 0.2) {
      console.warn("[OpenCV] Container not found or too small, using default. maxArea:", maxArea, "threshold:", w * h * 0.2);
      cleanup();
      return DEFAULT_CROP_DEFS;
    }

    console.log("[OpenCV] Container detected:", containerRect, "area:", maxArea);

    // === 仕切り線検出 ===
    // 容器内の暗い線（黒い溝）を検出
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);

    // 容器内の領域を切り出し
    const roi = gray.roi(containerRect);

    // 閾値処理で暗い部分（黒い仕切り線）を抽出
    const darkMask = new cv.Mat();
    cv.threshold(roi, darkMask, 60, 255, cv.THRESH_BINARY_INV);

    // 水平線検出用のカーネル（横長）
    const horKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(40, 1));
    const horLines = new cv.Mat();
    cv.morphologyEx(darkMask, horLines, cv.MORPH_OPEN, horKernel);

    // 垂直線検出用のカーネル（縦長）
    const verKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 40));
    const verLines = new cv.Mat();
    cv.morphologyEx(darkMask, verLines, cv.MORPH_OPEN, verKernel);

    // 各行・列のピクセル数を集計
    const rows = horLines.rows;
    const cols = horLines.cols;

    // 水平線の行を探す（ピクセルが集中している行）
    const rowSums = [];
    for (let y = 0; y < rows; y++) {
      let sum = 0;
      for (let x = 0; x < cols; x++) {
        if (horLines.ucharPtr(y, x)[0] > 0) sum++;
      }
      rowSums.push({ y, sum });
    }

    // 容器の中央付近（40-60%）で最もピクセル数が多い行を水平仕切りとする
    let horSepY = Math.floor(rows * 0.5);
    let maxRowSum = 0;
    for (let y = Math.floor(rows * 0.4); y < Math.floor(rows * 0.6); y++) {
      if (rowSums[y].sum > maxRowSum) {
        maxRowSum = rowSums[y].sum;
        horSepY = y;
      }
    }

    // 垂直線の列を探す
    const colSums = [];
    for (let x = 0; x < cols; x++) {
      let sum = 0;
      for (let y = 0; y < rows; y++) {
        if (verLines.ucharPtr(y, x)[0] > 0) sum++;
      }
      colSums.push({ x, sum });
    }

    // 上段の垂直仕切り（x: 20%-45%の範囲で最大）
    let upperVerX = Math.floor(cols * 0.33);
    let maxUpperSum = 0;
    for (let x = Math.floor(cols * 0.2); x < Math.floor(cols * 0.45); x++) {
      // 上段のみで集計
      let sum = 0;
      for (let y = 0; y < horSepY; y++) {
        if (verLines.ucharPtr(y, x)[0] > 0) sum++;
      }
      if (sum > maxUpperSum) {
        maxUpperSum = sum;
        upperVerX = x;
      }
    }

    // 下段の垂直仕切り（x: 55%-75%の範囲で最大）
    let lowerVerX = Math.floor(cols * 0.65);
    let maxLowerSum = 0;
    for (let x = Math.floor(cols * 0.55); x < Math.floor(cols * 0.75); x++) {
      let sum = 0;
      for (let y = horSepY; y < rows; y++) {
        if (verLines.ucharPtr(y, x)[0] > 0) sum++;
      }
      if (sum > maxLowerSum) {
        maxLowerSum = sum;
        lowerVerX = x;
      }
    }

    console.log("[OpenCV] Dividers - horSepY:", horSepY, "/", rows, "upperVerX:", upperVerX, "lowerVerX:", lowerVerX);

    // 仕切り位置を容器内相対座標に変換
    const horSepRel = horSepY / rows;
    const upperVerRel = upperVerX / cols;
    const lowerVerRel = lowerVerX / cols;

    // パディング（仕切り線の内側に少し余白を入れる）
    const padY = 0.02;
    const padX = 0.02;

    // 検出した仕切り位置に基づく区画定義
    const detectedRegions = {
      main:  {
        x1: upperVerRel + padX,
        y1: 0.04,
        x2: 0.97,
        y2: horSepRel - padY,
      },
      sub1:  {
        x1: 0.03,
        y1: 0.04,
        x2: upperVerRel - padX,
        y2: horSepRel - padY,
      },
      sub2:  {
        x1: lowerVerRel + padX,
        y1: horSepRel + padY,
        x2: 0.97,
        y2: 0.96,
      },
    };

    // クリーンアップ
    gray.delete();
    roi.delete();
    darkMask.delete();
    horKernel.delete();
    horLines.delete();
    verKernel.delete();
    verLines.delete();

    // 容器の位置（画像全体に対する比率）
    const cx1 = containerRect.x / w;
    const cy1 = containerRect.y / h;
    const cx2 = (containerRect.x + containerRect.width) / w;
    const cy2 = (containerRect.y + containerRect.height) / h;
    const cw_rel = cx2 - cx1;
    const ch_rel = cy2 - cy1;

    // 検出した仕切り位置を使う（容器内相対 → 画像全体座標に変換）
    const toImageCoord = (r) => [
      cy1 + r.y1 * ch_rel,
      cy1 + r.y2 * ch_rel,
      cx1 + r.x1 * cw_rel,
      cx1 + r.x2 * cw_rel,
    ];

    cleanup();

    // 容器の矩形情報を保存（デバッグ用）
    lastContainerBox = { x1: cx1, y1: cy1, x2: cx2, y2: cy2 };

    return [
      toImageCoord(detectedRegions.main),
      toImageCoord(detectedRegions.sub1),
      toImageCoord(detectedRegions.sub2),
    ];
  } catch (e) {
    console.error("OpenCV detection failed:", e);
    lastContainerBox = null;
    return DEFAULT_CROP_DEFS;
  }
}

async function cropFileToBase64(file, x1r, y1r, x2r, y2r) {
  const bitmap = await createImageBitmap(file);
  const sw = bitmap.width;
  const sh = bitmap.height;
  const x1 = Math.floor(sw * x1r);
  const y1 = Math.floor(sh * y1r);
  const cw = Math.max(1, Math.floor(sw * (x2r - x1r)));
  const ch = Math.max(1, Math.floor(sh * (y2r - y1r)));
  const maxSize = 800;
  const scale = Math.min(1, maxSize / Math.max(cw, ch));
  const ow = Math.max(1, Math.floor(cw * scale));
  const oh = Math.max(1, Math.floor(ch * scale));
  const canvas = new OffscreenCanvas(ow, oh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, x1, y1, cw, ch, 0, 0, ow, oh);
  bitmap.close();

  // 容器の側面（明るい赤）を黒く塗りつぶす（AIが側面を空きと誤認しないように）
  const imgData = ctx.getImageData(0, 0, ow, oh);
  const data = imgData.data;
  for (let i = 0; i < ow * oh; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const saturation = max === 0 ? 0 : (max - min) / max;
    const brightness = (r + g + b) / 3;

    // 容器の側面: 明るい赤茶色（底面より明るい）
    const isSide =
      r >= 150 &&
      g >= 30 && g <= 100 &&
      b >= 30 && b <= 100 &&
      r > g * 1.6 &&
      r > b * 1.6 &&
      Math.abs(g - b) < 30 &&
      saturation > 0.4 &&
      brightness > 80;

    if (isSide) {
      data[i * 4] = 0;
      data[i * 4 + 1] = 0;
      data[i * 4 + 2] = 0;
    }
  }
  ctx.putImageData(imgData, 0, 0);

  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.85 });
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const parts = reader.result.split(",");
      const b64 = parts[1];
      if (!b64 || b64.length === 0) reject(new Error("base64が空です"));
      else resolve(b64);
    };
    reader.onerror = () => reject(new Error("FileReader エラー"));
    reader.readAsDataURL(blob);
  });
}

// 画像内の赤茶色トレー底面部分をピクセル検出して、空きエリアのマスクを返す
async function detectTrayEmptyAreas(file, x1r, y1r, x2r, y2r) {
  const bitmap = await createImageBitmap(file);
  const sw = bitmap.width;
  const sh = bitmap.height;
  const x1 = Math.floor(sw * x1r);
  const y1 = Math.floor(sh * y1r);
  const cw = Math.max(1, Math.floor(sw * (x2r - x1r)));
  const ch = Math.max(1, Math.floor(sh * (y2r - y1r)));
  // 解析用に縮小
  const targetSize = 80;
  const scale = targetSize / Math.max(cw, ch);
  const ow = Math.max(10, Math.floor(cw * scale));
  const oh = Math.max(10, Math.floor(ch * scale));
  const canvas = new OffscreenCanvas(ow, oh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, x1, y1, cw, ch, 0, 0, ow, oh);
  bitmap.close();

  const imgData = ctx.getImageData(0, 0, ow, oh);
  const data = imgData.data;

  // 容器底面の色判定（厳密化）:
  // 底面は「暗い赤茶色」で、側面は「明るい赤」
  // 底面の特徴: R=80-150, G=20-55, B=20-55, R/G比とR/B比が大きい
  // 側面（除外）: R=180+, より明るい赤
  // 食材の茶色（除外）: G/Bがもう少し高い
  const mask = new Uint8Array(ow * oh);
  let emptyCount = 0;
  for (let i = 0; i < ow * oh; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const saturation = max === 0 ? 0 : (max - min) / max;
    const brightness = (r + g + b) / 3;

    const isTrayBottom =
      r >= 80 && r <= 150 &&         // 底面は中程度の暗さ
      g >= 18 && g <= 55 &&          // 緑は低い
      b >= 18 && b <= 55 &&          // 青も低い
      r > g * 2.2 &&                 // 赤が緑より強い（食材茶色を除外）
      r > b * 2.2 &&                 // 赤が青より強い
      Math.abs(g - b) < 18 &&        // 緑と青が近い
      saturation > 0.55 &&           // 彩度が十分
      brightness < 90 &&             // 全体的に暗い（明るい側面を除外）
      r < 155;                       // 赤すぎない（側面のハイライトを除外）

    if (isTrayBottom) {
      mask[i] = 1;
      emptyCount++;
    }
  }

  // グリッドで分割してトレー底面色が集中しているブロックを検出
  const gridSize = 10;
  const cellW = Math.floor(ow / gridSize);
  const cellH = Math.floor(oh / gridSize);
  const boxes = [];
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      let count = 0;
      let total = 0;
      for (let py = gy * cellH; py < (gy + 1) * cellH && py < oh; py++) {
        for (let px = gx * cellW; px < (gx + 1) * cellW && px < ow; px++) {
          if (mask[py * ow + px]) count++;
          total++;
        }
      }
      // 60%以上のピクセルが底面色の場合のみ空きとみなす
      if (total > 0 && count / total > 0.55) {
        boxes.push({
          x1: (gx * cellW) / ow,
          y1: (gy * cellH) / oh,
          x2: Math.min(((gx + 1) * cellW) / ow, 1),
          y2: Math.min(((gy + 1) * cellH) / oh, 1),
        });
      }
    }
  }

  const emptyRatio = emptyCount / (ow * oh);
  return { boxes, emptyRatio };
}

async function analyzeArea(b64, areaName) {
  const isMain = areaName.includes("メイン") || areaName.includes("右上");

  if (!isMain) {
    // 副菜はシンプルなプロンプト
    const prompt = `You are inspecting "${areaName}" section of a bento tray (not main section).

RULES:
1. If food is placed DIRECTLY on tray (not in cups):
   - Count large empty gaps as 15-25%

2. If ALL food is only in paper cups:
   - Return 0% (cup gaps are normal and acceptable)

3. DO NOT count small gaps between cups

Return ONLY JSON: {"pct": number, "reason": "Japanese text"}`;

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 300,
        messages: [{
          role: "user",
          content: [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: prompt },
          ],
        }],
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error?.message || `API error ${res.status}`);
    }

    const data = await res.json();
    const text = data.content?.[0]?.text || "{}";
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) throw new Error("JSON not found");
    return JSON.parse(match[0]);
  }

  // メインは2段階判定
  // STEP 1: 観察（数値を出さない）
  const observePrompt = `右上メイン区画の画像を観察してください。

【空きスペースの定義（重要）】
空きスペース = 食材もカップも置かれておらず、赤茶色のトレー「底面」がむき出しに見えている部分のみ

【画像の前処理について】
この画像では、容器の側面（明るい赤の斜面）は事前に黒く塗りつぶされています。
黒い部分は「容器の側面」なので、絶対に空きスペースとしてカウントしないでください。

【絶対にカウントしないもの】
- 黒く塗りつぶされた部分（容器の側面・斜面）※前処理済み
- 容器の縁（食材が置けない立ち上がった部分）
- 食材の表面や食材同士の凹凸（これは空きではない）
- カップの中の見える隙間（カップ内は食材の範囲）
- 仕切り板の溝や枠

【カウントするもの（重要）】
- 食材・カップの外側で、赤茶色のトレー底面が露出している部分
- 区画の端（上下左右）に食材が届いていない部分
- 食材間の隙間

【観察してください】
1. 食材は区画の中央/片寄って配置されているか？
2. 区画の上端/下端/左端/右端それぞれに食材は届いているか？
3. トレー底面が露出している箇所と、その大きさは？
4. 全体の印象として「ぎっしり」「やや余裕あり」「明確な空きあり」「スカスカ」のどれか？

【分類の基準（厳密に判定してください）】

◆ ぎっしり (0-5%)
- 食材が区画の端（上下左右すべて）にほぼ届いている
- トレー底面の露出はほぼない
- 例: パターンB（ヒレカツ弁当）のような詰まり方

◆ やや余裕あり (10-17%)
- 1箇所だけ小さな隙間がある
- 例: 卵焼きの横に手のひらサイズの小さな隙間

◆ 明確な空きあり (18-28%) ← 片寄り配置は必ずこれ
- 食材が中央や片側に寄り、反対側に明確な空きがある
- 区画の上/下/左/右のいずれかが食材で埋まっていない
- 食材間に複数の隙間があり合計面積が2割程度

◆ スカスカ (35-50%)
- 半分近くが露出

【重要な判定ルール】
- 食材が区画の「左側または中央だけ」にあり、「右側が明確に空いている」 → 明確な空きあり
- 食材が区画の「上側だけ」にあり、「下側が明確に空いている」 → 明確な空きあり
- トレー露出が区画の端に沿って帯状に続いている → 明確な空きあり

JSONで返してください: {"observation": "詳細な観察結果", "overall": "ぎっしり|やや余裕あり|明確な空きあり|スカスカ"}`;

  const obsRes = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 400,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: observePrompt },
        ],
      }],
    }),
  });

  if (!obsRes.ok) {
    const err = await obsRes.json();
    throw new Error(err.error?.message || "observation error");
  }

  const obsData = await obsRes.json();
  const obsText = obsData.content?.[0]?.text || "{}";
  const obsMatch = obsText.match(/\{[\s\S]*\}/);
  let observation = { observation: "", overall: "やや余裕あり" };
  if (obsMatch) {
    try { observation = JSON.parse(obsMatch[0]); } catch {}
  }

  // STEP 2: 観察結果に基づいてパーセンテージ決定（ルールベース＋AI補正）
  const baseRange = {
    "ぎっしり": [0, 5],
    "やや余裕あり": [10, 17],
    "明確な空きあり": [18, 28],
    "スカスカ": [35, 50],
  };
  const range = baseRange[observation.overall] || [10, 20];

  const decidePrompt = `先ほどあなたは以下のように観察しました:

観察結果: ${observation.observation}
全体評価: ${observation.overall}

この観察結果に基づき、空きスペース率を${range[0]}〜${range[1]}%の範囲で、整数で正確に決定してください。
"${observation.overall}"なら${range[0]}〜${range[1]}%が妥当です。
観察内容の具体性に応じて、この範囲内で適切な値を選んでください。

さらに、空きスペースが存在する位置を、この区画画像内の相対座標（0〜1）で1〜3個のボックスで示してください。
空きスペースがなければ空配列[]を返してください。

JSONで返してください:
{
  "pct": <${range[0]}〜${range[1]}の整数>,
  "reason": "観察結果の要約（日本語100文字以内）",
  "empty_boxes": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}]
}`;

  const decRes = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: decidePrompt },
        ],
      }],
    }),
  });

  if (!decRes.ok) {
    const err = await decRes.json();
    throw new Error(err.error?.message || "decision error");
  }

  const decData = await decRes.json();
  const decText = decData.content?.[0]?.text || "{}";
  const decMatch = decText.match(/\{[\s\S]*\}/);
  if (!decMatch) throw new Error("JSON not found in decision");
  const result = JSON.parse(decMatch[0]);

  // レンジ内にクリップ
  result.pct = Math.max(range[0], Math.min(range[1], result.pct));

  return result;
}

async function generateAdvice(areaResults) {
  const summary = areaResults.map(r => `${r.name}: ${r.pct}% (${r.reason})`).join("\n");
  const prompt = `以下は日本のお弁当の3区画それぞれの空きスペース評価結果です:

${summary}

この結果を踏まえて、より見栄えのする盛り付け方法について、総評とアドバイスを日本語で100-150文字程度で述べてください。
具体的な改善策（例: 副菜を足す、配置を変える、カップを大きくする等）を含めてください。
JSONで返してください: {"advice": "総評とアドバイス"}`;

  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 400,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!res.ok) return "総評の生成に失敗しました";
  const data = await res.json();
  const text = data.content?.[0]?.text || "{}";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return text;
  try {
    return JSON.parse(match[0]).advice || "";
  } catch {
    return text;
  }
}

export default function BentoCheckerPro() {
  const [imgSrc, setImgSrc] = useState(null);
  const [imgFile, setImgFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [viewHistory, setViewHistory] = useState(false);
  const [cropDefs, setCropDefs] = useState(() => loadSavedCropDefs() || DEFAULT_CROP_DEFS);
  const [containerBox, setContainerBox] = useState(null);
  const [debugMode, setDebugMode] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [dragState, setDragState] = useState(null);
  const imageContainerRef = useRef(null);
  const fileRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    setImgFile(file);
    setResults(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setImgSrc(url);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFile(file);
  }, [handleFile]);

  const analyze = useCallback(async () => {
    if (!imgFile) return;
    setAnalyzing(true);
    setError(null);
    setProgress(0);

    try {
      // STEP -1: 容器を切り抜いて正規化
      setProgressLabel("容器を検出して画像を正規化中...");
      const { croppedFile, success } = await cropImageToContainer(imgFile);

      if (success) {
        // 切り抜いた画像をプレビューに反映
        const url = URL.createObjectURL(croppedFile);
        setImgSrc(url);
      }

      // 解析対象ファイル（切り抜き済みまたは元のファイル）
      const targetFile = croppedFile;

      // STEP 0: 区画の位置を決定（保存座標 or デフォルト）
      const savedDefs = loadSavedCropDefs();
      let cropDefs;
      if (savedDefs) {
        setProgressLabel("保存された座標を使用中...");
        cropDefs = savedDefs;
      } else {
        // 切り抜き済み画像なら、容器全体に対する固定比率を使用
        cropDefs = success ? [
          [0.05, 0.45, 0.36, 0.97],   // 右上（メイン）
          [0.05, 0.45, 0.04, 0.34],   // 左上（副菜A）
          [0.50, 0.95, 0.65, 0.97],   // 右下（副菜B）
        ] : DEFAULT_CROP_DEFS;
      }
      setCropDefs(cropDefs);
      setContainerBox(lastContainerBox);

      const areaResults = [];
      for (let i = 0; i < AREAS.length; i++) {
        setProgressLabel(`${AREAS[i].name} を解析中...`);
        setProgress(Math.round(((i + 1) / (AREAS.length + 1)) * 100));

        const [y1r, y2r, x1r, x2r] = cropDefs[i];
        const b64 = await cropFileToBase64(targetFile, x1r, y1r, x2r, y2r);
        const res = await analyzeArea(b64, AREAS[i].name);

        // ピクセル検出で実際のトレー露出ボックスを取得（ハイライト表示のみに使用）
        const { boxes: detectedBoxes } = await detectTrayEmptyAreas(targetFile, x1r, y1r, x2r, y2r);

        areaResults.push({
          ...AREAS[i],
          pct: res.pct ?? 0,
          reason: res.reason ?? "",
          empty_boxes: detectedBoxes
        });
        setProgress(Math.round(((i + 2) / (AREAS.length + 1)) * 100));
      }

      const avg = areaResults.reduce((s, r) => s + r.pct, 0) / areaResults.length;
      const isFail = areaResults.some((r) => r.pct >= 15);

      setProgressLabel("総評を生成中...");
      const advice = await generateAdvice(areaResults);

      const record = {
        id: Date.now(),
        time: new Date().toLocaleString("ja-JP"),
        imgSrc,
        areas: areaResults,
        avg: Math.round(avg * 10) / 10,
        status: isFail ? "FAIL" : "PASS",
        advice,
      };
      setResults(record);
      setHistory((h) => [record, ...h]);
    } catch (e) {
      setError(e.message);
    } finally {
      setAnalyzing(false);
      setProgress(0);
      setProgressLabel("");
    }
  }, [imgFile, imgSrc]);

  const reset = () => {
    setImgSrc(null);
    setImgFile(null);
    setResults(null);
    setError(null);
    setCropDefs(loadSavedCropDefs() || DEFAULT_CROP_DEFS);
  };

  // ドラッグ開始
  const handleDragStart = (e, boxIndex, handle) => {
    if (!editMode) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = imageContainerRef.current.getBoundingClientRect();
    setDragState({ boxIndex, handle, rect });
  };

  // ドラッグ中
  useEffect(() => {
    if (!dragState) return;
    const handleMove = (e) => {
      const { boxIndex, handle, rect } = dragState;
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

      setCropDefs((prev) => {
        const newDefs = prev.map(d => [...d]);
        const [y1, y2, x1, x2] = newDefs[boxIndex];

        if (handle === "move") {
          // 中央ドラッグで移動
          const cx = (x1 + x2) / 2;
          const cy = (y1 + y2) / 2;
          const dx = x - cx;
          const dy = y - cy;
          newDefs[boxIndex] = [
            Math.max(0, y1 + dy),
            Math.min(1, y2 + dy),
            Math.max(0, x1 + dx),
            Math.min(1, x2 + dx),
          ];
        } else {
          // 角ドラッグでリサイズ
          if (handle.includes("n")) newDefs[boxIndex][0] = Math.min(y, y2 - 0.05);
          if (handle.includes("s")) newDefs[boxIndex][1] = Math.max(y, y1 + 0.05);
          if (handle.includes("w")) newDefs[boxIndex][2] = Math.min(x, x2 - 0.05);
          if (handle.includes("e")) newDefs[boxIndex][3] = Math.max(x, x1 + 0.05);
        }
        return newDefs;
      });
    };
    const handleUp = () => setDragState(null);
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [dragState]);

  const saveAndExitEdit = () => {
    saveCropDefs(cropDefs);
    setEditMode(false);
  };

  const resetSavedCoords = () => {
    localStorage.removeItem("savedCropDefs");
    setCropDefs(DEFAULT_CROP_DEFS);
  };

  useEffect(() => {
    if (imgFile && !results && !analyzing && !error) {
      analyze();
    }
  }, [imgFile]);

  // Apple風カラー
  const C = {
    bg: "#f5f5f7",
    card: "#ffffff",
    text: "#1d1d1f",
    textSub: "#6e6e73",
    border: "#d2d2d7",
    accent: "#0071e3",
    success: "#34c759",
    danger: "#ff3b30",
    successBg: "#e8f8ec",
    dangerBg: "#ffeceb",
  };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', 'Hiragino Sans', sans-serif", color: C.text }}>
      {/* Header */}
      <div style={{ background: C.card, borderBottom: `1px solid ${C.border}`, padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 32 }}>🍱</span>
          <div>
            <div style={{ fontSize: 20, fontWeight: 600, letterSpacing: "-0.02em" }}>Bento Checker Pro</div>
            <div style={{ fontSize: 12, color: C.textSub, marginTop: 2 }}>AI-Powered Quality Inspection</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          {imgSrc && !editMode && (
            <button
              onClick={() => setEditMode(true)}
              style={{ background: "#0071e3", border: "none", color: "white", borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 500 }}
            >
              ✏️ 枠を調整
            </button>
          )}
          {editMode && (
            <>
              <button
                onClick={saveAndExitEdit}
                style={{ background: "#34c759", border: "none", color: "white", borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 600 }}
              >
                ✓ 保存して終了
              </button>
              <button
                onClick={resetSavedCoords}
                style={{ background: "transparent", border: `1px solid ${C.border}`, color: C.textSub, borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14 }}
              >
                リセット
              </button>
            </>
          )}
          <button
            onClick={() => setDebugMode(!debugMode)}
            style={{ background: debugMode ? "#f5a623" : "transparent", border: `1px solid ${debugMode ? "#f5a623" : C.border}`, color: debugMode ? "white" : C.text, borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 500 }}
          >
            🔍 デバッグ {debugMode ? "ON" : "OFF"}
          </button>
          <button
            onClick={() => setViewHistory(!viewHistory)}
            style={{ background: "transparent", border: `1px solid ${C.border}`, color: C.text, borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 500 }}
          >
            📋 履歴 ({history.length})
          </button>
        </div>
      </div>

      <div style={{ display: "flex", minHeight: "calc(100vh - 72px)", maxWidth: 1800, margin: "0 auto", padding: "32px", gap: 32 }}>
        {/* Left Panel */}
        <div style={{ flex: "1 1 65%", minWidth: 0 }}>
          {!imgSrc && (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileRef.current?.click()}
              style={{ background: C.card, border: `2px dashed ${C.border}`, borderRadius: 20, padding: "80px 40px", textAlign: "center", cursor: "pointer", transition: "all 0.2s" }}
            >
              <div style={{ fontSize: 64, marginBottom: 20 }}>📷</div>
              <div style={{ color: C.text, fontSize: 18, fontWeight: 500, marginBottom: 8 }}>お弁当の写真をドロップ</div>
              <div style={{ color: C.textSub, fontSize: 14 }}>またはクリックして選択</div>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
            </div>
          )}

          {imgSrc && (
            <div style={{ textAlign: "center" }}>
              <div style={{ background: C.card, borderRadius: 20, padding: 12, boxShadow: "0 2px 10px rgba(0,0,0,0.05)" }}>
                <div ref={imageContainerRef} style={{ position: "relative", display: "inline-block", width: "85%" }}>
                  <img
                    src={imgSrc}
                    alt="bento"
                    style={{ width: "100%", borderRadius: 12, display: "block" }}
                  />
                  {/* デバッグ: OpenCVが検出した容器の外枠 */}
                  {debugMode && containerBox && (
                    <div style={{
                      position: "absolute",
                      left: `${containerBox.x1 * 100}%`,
                      top: `${containerBox.y1 * 100}%`,
                      width: `${(containerBox.x2 - containerBox.x1) * 100}%`,
                      height: `${(containerBox.y2 - containerBox.y1) * 100}%`,
                      border: "4px dashed #f5a623",
                      boxSizing: "border-box",
                      pointerEvents: "none",
                    }}>
                      <div style={{
                        position: "absolute",
                        top: -30,
                        left: 0,
                        background: "#f5a623",
                        color: "white",
                        fontSize: 11,
                        fontWeight: 600,
                        padding: "3px 10px",
                        borderRadius: 4,
                      }}>
                        🔍 OpenCV検出: 容器の外枠
                      </div>
                    </div>
                  )}

                  {/* 3つのエリア枠オーバーレイ */}
                  {cropDefs.map((def, i) => {
                    const [y1r, y2r, x1r, x2r] = def;
                    const colors = ["#ff3b30", "#0071e3", "#34c759"];
                    const labels = ["右上（メイン）", "左上（副菜A）", "右下（副菜B）"];
                    const area = results?.areas?.[i];
                    const pct = area?.pct;
                    const emptyBoxes = area?.empty_boxes || [];
                    const areaW = x2r - x1r;
                    const areaH = y2r - y1r;
                    const handleSize = 14;
                    return (
                      <div
                        key={i}
                        onMouseDown={editMode ? (e) => handleDragStart(e, i, "move") : undefined}
                        style={{
                          position: "absolute",
                          left: `${x1r * 100}%`,
                          top: `${y1r * 100}%`,
                          width: `${areaW * 100}%`,
                          height: `${areaH * 100}%`,
                          border: `3px solid ${colors[i]}`,
                          borderRadius: 8,
                          boxSizing: "border-box",
                          pointerEvents: editMode ? "auto" : "none",
                          cursor: editMode ? "move" : "default",
                          background: editMode ? `${colors[i]}15` : "transparent",
                        }}
                      >
                        {/* 空きスペースのハイライト（編集モード中は非表示） */}
                        {!editMode && emptyBoxes.map((box, j) => (
                          <div
                            key={j}
                            style={{
                              position: "absolute",
                              left: `${box.x1 * 100}%`,
                              top: `${box.y1 * 100}%`,
                              width: `${(box.x2 - box.x1) * 100}%`,
                              height: `${(box.y2 - box.y1) * 100}%`,
                              background: "rgba(255, 235, 59, 0.45)",
                              border: "2px dashed #f5a623",
                              boxSizing: "border-box",
                              pointerEvents: "none",
                            }}
                          />
                        ))}
                        <div style={{
                          position: "absolute",
                          top: -26,
                          left: -3,
                          background: colors[i],
                          color: "white",
                          fontSize: 11,
                          fontWeight: 600,
                          padding: "2px 8px",
                          borderRadius: 4,
                          whiteSpace: "nowrap",
                          pointerEvents: "none",
                        }}>
                          {labels[i]}{!editMode && pct !== undefined ? ` ${pct}%` : ""}
                        </div>
                        {/* 編集モード時のドラッグハンドル（4角） */}
                        {editMode && ["nw", "ne", "sw", "se"].map((pos) => {
                          const styles = {
                            position: "absolute",
                            width: handleSize,
                            height: handleSize,
                            background: colors[i],
                            border: "2px solid white",
                            borderRadius: "50%",
                            cursor: `${pos}-resize`,
                            boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
                          };
                          if (pos.includes("n")) styles.top = -handleSize / 2;
                          if (pos.includes("s")) styles.bottom = -handleSize / 2;
                          if (pos.includes("w")) styles.left = -handleSize / 2;
                          if (pos.includes("e")) styles.right = -handleSize / 2;
                          return (
                            <div
                              key={pos}
                              onMouseDown={(e) => handleDragStart(e, i, pos)}
                              style={styles}
                            />
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              </div>

              {analyzing && (
                <div style={{ marginTop: 24, background: C.card, borderRadius: 16, padding: 20, boxShadow: "0 2px 10px rgba(0,0,0,0.05)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, fontSize: 14 }}>
                    <span style={{ color: C.textSub }}>{progressLabel}</span>
                    <span style={{ color: C.accent, fontWeight: 600 }}>{progress}%</span>
                  </div>
                  <div style={{ background: "#f0f0f3", borderRadius: 8, height: 8, overflow: "hidden" }}>
                    <div style={{ width: `${progress}%`, height: "100%", background: C.accent, transition: "width 0.4s ease" }} />
                  </div>
                </div>
              )}

              {error && (
                <div style={{ marginTop: 24, background: C.dangerBg, border: `1px solid ${C.danger}`, borderRadius: 12, padding: 16, color: C.danger, fontSize: 14 }}>
                  ⚠️ {error}
                </div>
              )}

              {!analyzing && (
                <div style={{ display: "flex", justifyContent: "center", marginTop: 24 }}>
                  <button
                    onClick={reset}
                    style={{ background: C.accent, color: "white", border: "none", borderRadius: 30, padding: "14px 40px", cursor: "pointer", fontSize: 16, fontWeight: 500 }}
                  >
                    ＋ 新規スキャン
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div style={{ flex: "1 1 35%", minWidth: 400 }}>
          {!results && !viewHistory && !analyzing && (
            <div style={{ color: C.textSub, textAlign: "center", marginTop: 100, fontSize: 16 }}>
              画像をアップロードして<br />解析を開始してください
            </div>
          )}

          {results && !viewHistory && (
            <div>
              {/* Status Card */}
              <div style={{
                background: results.status === "FAIL" ? C.dangerBg : C.successBg,
                border: `2px solid ${results.status === "FAIL" ? C.danger : C.success}`,
                borderRadius: 20,
                padding: "20px 28px",
                marginBottom: 24,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between"
              }}>
                <div style={{ fontSize: 32, fontWeight: 700, color: results.status === "FAIL" ? C.danger : C.success, letterSpacing: "-0.02em" }}>
                  {results.status}
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 12, color: C.textSub, marginBottom: 2 }}>平均空き率</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: C.text, letterSpacing: "-0.02em" }}>{results.avg}%</div>
                </div>
              </div>

              {/* Area Cards */}
              {results.areas.map((area, i) => {
                const fail = area.pct >= 15;
                const frameColors = ["#ff3b30", "#0071e3", "#34c759"]; // 赤・青・緑
                const frameColor = frameColors[i];
                return (
                  <div key={area.key} style={{
                    background: C.card,
                    border: `1px solid ${C.border}`,
                    borderLeft: `6px solid ${frameColor}`,
                    borderRadius: 16,
                    padding: "20px 24px",
                    marginBottom: 16,
                    boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                  }}>
                    <div style={{ fontSize: 14, color: frameColor, marginBottom: 6, fontWeight: 600 }}>{area.name}</div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: fail ? C.danger : C.success, marginBottom: 10, letterSpacing: "-0.02em" }}>
                      {area.pct}%
                    </div>
                    <div style={{ background: "#f0f0f3", borderRadius: 4, height: 6, marginBottom: 14 }}>
                      <div style={{ width: `${Math.min(area.pct, 100)}%`, height: "100%", background: fail ? C.danger : C.success, borderRadius: 4, transition: "width 0.5s" }} />
                    </div>
                    <div style={{ fontSize: 15, color: C.text, lineHeight: 1.6 }}>{area.reason}</div>
                  </div>
                );
              })}

              <div style={{ fontSize: 13, color: C.textSub, marginTop: 12, textAlign: "right" }}>{results.time}</div>

              {/* 総評カード */}
              {results.advice && (
                <div style={{
                  background: "#fffbea",
                  border: `1px solid #f5d77a`,
                  borderLeft: `6px solid #f5a623`,
                  borderRadius: 16,
                  padding: "20px 24px",
                  marginTop: 20,
                  boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                }}>
                  <div style={{ fontSize: 14, color: "#8a6d00", marginBottom: 10, fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>
                    💡 総評・改善アドバイス
                  </div>
                  <div style={{ fontSize: 15, color: C.text, lineHeight: 1.7 }}>{results.advice}</div>
                </div>
              )}
            </div>
          )}

          {viewHistory && (
            <div>
              <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 20 }}>検査履歴</div>
              {history.length === 0 && <div style={{ color: C.textSub, fontSize: 14 }}>履歴なし</div>}
              {history.map((rec) => (
                <div key={rec.id} style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 16, padding: 16, marginBottom: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                    <span style={{ fontWeight: 700, color: rec.status === "FAIL" ? C.danger : C.success, fontSize: 18 }}>{rec.status}</span>
                    <span style={{ fontSize: 22, fontWeight: 700, color: C.text }}>{rec.avg}%</span>
                  </div>
                  <img src={rec.imgSrc} alt="" style={{ width: "100%", borderRadius: 10, marginBottom: 8 }} />
                  <div style={{ fontSize: 12, color: C.textSub }}>{rec.time}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
