import { useState, useRef, useCallback, useEffect } from "react";

const AREAS = [
  { name: "右上（メイン）", key: "main" },
  { name: "左上（副菜A）", key: "sub1" },
  { name: "右下（副菜B）", key: "sub2" },
];

// 青緑マスク画像（底面=青、側面=緑）から算出した正確な底面座標
const DEFAULT_CROP_DEFS = [
  [0.104, 0.411, 0.427, 0.929],  // 右上（メイン）: y1, y2, x1, x2
  [0.109, 0.409, 0.159, 0.345],  // 左上（副菜A）
  [0.546, 0.905, 0.712, 0.917],  // 右下（副菜B）
];

// メイン区画の空白率を圧縮する係数（体感より高めに出るため）
const MAIN_SCALE = 0.7;
// FAIL判定の閾値（％）
const FAIL_THRESHOLD = 10;

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
    const cleaned = cropDefs.map(d => [...d]);
    localStorage.setItem("savedCropDefs", JSON.stringify(cleaned));
  } catch {}
}

// 枠内の底面色をピクセル検出して空白率を計算（数値判定はこれに一本化）
// isSub: 副菜区画（カップ入り）なら true → カップ外（外周）の底面を空白から除外
async function calcEmptyRateByPixel(file, x1r, y1r, x2r, y2r, isSub = false) {
  const bitmap = await createImageBitmap(file);
  const sw = bitmap.width;
  const sh = bitmap.height;
  const x1 = Math.floor(sw * x1r);
  const y1 = Math.floor(sh * y1r);
  const cw = Math.max(1, Math.floor(sw * (x2r - x1r)));
  const ch = Math.max(1, Math.floor(sh * (y2r - y1r)));

  const targetSize = 200;
  const scale = targetSize / Math.max(cw, ch);
  const ow = Math.max(10, Math.floor(cw * scale));
  const oh = Math.max(10, Math.floor(ch * scale));
  const canvas = new OffscreenCanvas(ow, oh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, x1, y1, cw, ch, 0, 0, ow, oh);
  bitmap.close();

  const imgData = ctx.getImageData(0, 0, ow, oh);
  const data = imgData.data;
  const total = ow * oh;

  // 底面（赤茶色）の条件:
  // - 赤が強い（R > G*1.5 かつ R > B*1.5）
  // - 暗い赤茶（60 < R < 170）→ 側面の明るい赤を除外
  // - G と B が近い（|G-B| < 30）→ 純粋な赤茶色
  // - G < 85, B < 85 → 食材の茶色を除外
  const mask = new Uint8Array(total);
  let bottomCount = 0;
  for (let i = 0; i < total; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    if (
      r > g * 1.5 &&
      r > b * 1.5 &&
      r > 60 && r < 170 &&
      Math.abs(g - b) < 30 &&
      g < 85 && b < 85
    ) {
      mask[i] = 1;
      bottomCount++;
    }
  }

  let emptyCount = bottomCount;

  // 副菜区画: カップの外側（容器の底）の赤茶色は「仕方ない隙間」として除外する。
  // カップは区画の中央寄りにあるため、外周に固まっている底面色はカップ外とみなして無視する。
  if (isSub) {
    const marginX = Math.floor(ow * 0.15); // 外周15%はカップ外とみなして除外
    const marginY = Math.floor(oh * 0.15);
    let innerBottom = 0;
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const idx = y * ow + x;
        if (!mask[idx]) continue;
        const isOuter = x < marginX || x >= ow - marginX || y < marginY || y >= oh - marginY;
        if (isOuter) {
          mask[idx] = 0; // 外周の底面は空白から除外
        } else {
          innerBottom++;
        }
      }
    }
    emptyCount = innerBottom;
  }

  const rate = Math.round(emptyCount / total * 100);

  // 底面が集中しているブロックを「空きボックス」として抽出（デバッグ表示用）
  const gridSize = 10;
  const cellW = Math.floor(ow / gridSize);
  const cellH = Math.floor(oh / gridSize);
  const boxes = [];
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      let count = 0, cellTotal = 0;
      for (let py = gy * cellH; py < (gy + 1) * cellH && py < oh; py++) {
        for (let px = gx * cellW; px < (gx + 1) * cellW && px < ow; px++) {
          if (mask[py * ow + px]) count++;
          cellTotal++;
        }
      }
      if (cellTotal > 0 && count / cellTotal > 0.55) {
        boxes.push({
          x1: (gx * cellW) / ow,
          y1: (gy * cellH) / oh,
          x2: Math.min(((gx + 1) * cellW) / ow, 1),
          y2: Math.min(((gy + 1) * cellH) / oh, 1),
        });
      }
    }
  }

  console.log(`[PixelCalc]${isSub ? "[副菜]" : "[メイン]"} 空白率=${rate}% (${emptyCount}/${total}px)`);
  return { rate, count: emptyCount, boxes };
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

// AIには食材名だけを取得させる（空白率の計算はピクセル検出に任せる＝ブレない）
async function analyzeArea(b64, areaName) {
  const prompt = `お弁当の「${areaName}」区画の画像です。
この区画に入っている食材を大まかに挙げてください。空白率の計算は不要です。
JSONのみ出力: {"items": [{"name": "食材名", "pct": おおよその面積%}]}`;

  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [{ role: "user", content: [
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
        { type: "text", text: prompt },
      ]}],
    }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.error?.message || `API error ${res.status}`); }
  const data = await res.json();
  const text = data.content?.[0]?.text || "{}";

  let result = {};
  try {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      const cleaned = match[0]
        .replace(/：/g, ":")
        .replace(/，/g, ",")
        .replace(/（/g, "(")
        .replace(/）/g, ")")
        .replace(/,\s*}/g, "}")
        .replace(/,\s*]/g, "]");
      result = JSON.parse(cleaned);
    }
  } catch (e) {
    console.warn("[JSON parse error]", e.message, "\ntext:", text.slice(0, 200));
  }

  const itemsDesc = (result.items ?? [])
    .map(it => `${it.name}${it.pct ? it.pct + "%" : ""}`)
    .join("、");
  return { reason: itemsDesc };
}

async function generateAdvice(areaResults) {
  const summary = areaResults.map(r => `${r.name}: 空き率${r.pct}% (${r.reason})`).join("\n");
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
  const imgRef = useRef(null);
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
      const targetFile = imgFile;
      const savedDefs = loadSavedCropDefs();
      let usedCropDefs;
      if (savedDefs) {
        setProgressLabel("保存された座標を使用中...");
        usedCropDefs = savedDefs;
      } else {
        setProgressLabel("デフォルト座標を使用中...");
        usedCropDefs = DEFAULT_CROP_DEFS;
      }
      setCropDefs(usedCropDefs);

      const areaResults = [];
      for (let i = 0; i < AREAS.length; i++) {
        setProgressLabel(`${AREAS[i].name} を解析中...`);
        setProgress(Math.round(((i + 1) / (AREAS.length + 1)) * 100));

        const [y1r, y2r, x1r, x2r] = usedCropDefs[i];

        // メインは底面そのまま、副菜はカップ外の底面を除外して判定
        const isSub = AREAS[i].key !== "main";
        const pixel = await calcEmptyRateByPixel(targetFile, x1r, y1r, x2r, y2r, isSub);

        // メイン区画の空白率は体感より高めに出るため係数で圧縮
        let finalPct = pixel.rate;
        if (AREAS[i].key === "main") {
          finalPct = Math.round(pixel.rate * MAIN_SCALE);
        }

        // 食材名のみAIに取得させる（数値に影響しないので1回でOK）
        const b64 = await cropFileToBase64(targetFile, x1r, y1r, x2r, y2r);
        let aiReason = "";
        try {
          const ai = await analyzeArea(b64, AREAS[i].name);
          aiReason = ai.reason;
        } catch (e) {
          console.warn("[AI食材名取得失敗]", e.message);
        }

        areaResults.push({
          ...AREAS[i],
          pct: finalPct,
          reason: aiReason,
          empty_boxes: pixel.boxes || [],
        });
      }

      const avg = areaResults.reduce((s, r) => s + r.pct, 0) / areaResults.length;
      const isFail = areaResults.some((r) => r.pct >= FAIL_THRESHOLD);

      setProgressLabel("総評を生成中...");
      let advice = "";
      try {
        advice = await generateAdvice(areaResults);
      } catch (e) {
        console.warn("[総評生成失敗]", e.message);
      }

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

  const handleDragStart = (e, boxIndex, handle) => {
    if (!editMode) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = imgRef.current.getBoundingClientRect();
    const startX = (e.clientX - rect.left) / rect.width;
    const startY = (e.clientY - rect.top) / rect.height;
    setDragState({ boxIndex, handle, rect, startX, startY, startDef: [...cropDefs[boxIndex]] });
  };

  useEffect(() => {
    if (!dragState) return;
    const handleMove = (e) => {
      const { boxIndex, handle, rect, startX, startY, startDef } = dragState;
      const curX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const curY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      const dx = curX - startX;
      const dy = curY - startY;
      const [sy1, sy2, sx1, sx2] = startDef;
      setCropDefs(prev => {
        const nd = prev.map(d => [...d]);
        if (handle === "move") {
          nd[boxIndex] = [Math.max(0,sy1+dy), Math.min(1,sy2+dy), Math.max(0,sx1+dx), Math.min(1,sx2+dx)];
        } else {
          const n = [sy1,sy2,sx1,sx2];
          if (handle.includes("n")) n[0] = Math.min(curY, sy2-0.03);
          if (handle.includes("s")) n[1] = Math.max(curY, sy1+0.03);
          if (handle.includes("w")) n[2] = Math.min(curX, sx2-0.03);
          if (handle.includes("e")) n[3] = Math.max(curX, sx1+0.03);
          nd[boxIndex] = n;
        }
        return nd;
      });
    };
    const handleUp = () => setDragState(null);
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => { window.removeEventListener("mousemove", handleMove); window.removeEventListener("mouseup", handleUp); };
  }, [dragState]);

  const saveAndExitEdit = () => {
    saveCropDefs(cropDefs);
    setEditMode(false);
  };

  const resetSavedCoords = () => {
    localStorage.removeItem("savedCropDefs");
    setCropDefs(DEFAULT_CROP_DEFS);
  };

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
            <button onClick={() => setEditMode(true)}
              style={{ background: "#0071e3", border: "none", color: "white", borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 500 }}>
              ✏️ 枠を調整
            </button>
          )}
          {editMode && (
            <>
              <button onClick={saveAndExitEdit}
                style={{ background: "#34c759", border: "none", color: "white", borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 600 }}>
                ✓ 保存して終了
              </button>
              <button onClick={resetSavedCoords}
                style={{ background: "transparent", border: `1px solid ${C.border}`, color: C.textSub, borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14 }}>
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
                <div style={{ position: "relative", display: "inline-block", width: "85%" }}>
                  <img
                    ref={(el) => { imgRef.current = el; imageContainerRef.current = el?.parentElement || null; }}
                    src={imgSrc}
                    alt="bento"
                    style={{ width: "100%", borderRadius: 12, display: "block" }}
                  />
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
                        position: "absolute", top: -30, left: 0,
                        background: "#f5a623", color: "white",
                        fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 4,
                      }}>
                        🔍 OpenCV検出: 容器の外枠
                      </div>
                    </div>
                  )}

                  {cropDefs.map((def, i) => {
                    const [y1r, y2r, x1r, x2r] = def;
                    const colors = ["#ff3b30", "#0071e3", "#34c759"];
                    const labels = ["右上（メイン）", "左上（副菜A）", "右下（副菜B）"];
                    const area = results?.areas?.[i];
                    const pct = area?.pct;
                    const emptyBoxes = (debugMode && area?.empty_boxes) ? area.empty_boxes : [];
                    const hs = 14;
                    return (
                      <div key={i}
                        onMouseDown={editMode ? (e) => handleDragStart(e, i, "move") : undefined}
                        style={{
                          position: "absolute",
                          left: `${x1r*100}%`, top: `${y1r*100}%`,
                          width: `${(x2r-x1r)*100}%`, height: `${(y2r-y1r)*100}%`,
                          border: `3px solid ${colors[i]}`,
                          borderRadius: 8, boxSizing: "border-box",
                          pointerEvents: editMode ? "auto" : "none",
                          cursor: editMode ? "move" : "default",
                          background: editMode ? `${colors[i]}15` : "transparent",
                        }}>
                        {!editMode && emptyBoxes.map((box, j) => (
                          <div key={j} style={{
                            position: "absolute",
                            left: `${box.x1*100}%`, top: `${box.y1*100}%`,
                            width: `${(box.x2-box.x1)*100}%`, height: `${(box.y2-box.y1)*100}%`,
                            background: "rgba(255,235,59,0.45)", border: "2px dashed #f5a623",
                            boxSizing: "border-box", pointerEvents: "none",
                          }} />
                        ))}
                        <div style={{
                          position: "absolute", top: -26, left: -3,
                          background: colors[i], color: "white",
                          fontSize: 11, fontWeight: 600, padding: "2px 8px",
                          borderRadius: 4, whiteSpace: "nowrap", pointerEvents: "none",
                        }}>
                          {labels[i]}{!editMode && pct !== undefined ? ` ${pct}%` : ""}
                        </div>
                        {editMode && ["nw","ne","sw","se"].map(pos => {
                          const s = { position:"absolute", width:hs, height:hs, background:colors[i], border:"2px solid white", borderRadius:"50%", cursor:`${pos}-resize`, boxShadow:"0 1px 3px rgba(0,0,0,0.3)" };
                          if (pos.includes("n")) s.top = -hs/2;
                          if (pos.includes("s")) s.bottom = -hs/2;
                          if (pos.includes("w")) s.left = -hs/2;
                          if (pos.includes("e")) s.right = -hs/2;
                          return <div key={pos} onMouseDown={(e)=>handleDragStart(e,i,pos)} style={s} />;
                        })}
                      </div>
                    );
                  })}
                </div>
              </div>

              {!editMode && !analyzing && !results && imgFile && (
                <div style={{ marginTop: 20 }}>
                  <button
                    onClick={analyze}
                    style={{
                      background: "#0071e3", color: "white", border: "none",
                      borderRadius: 28, padding: "16px 48px", fontSize: 18, fontWeight: 600,
                      cursor: "pointer", boxShadow: "0 4px 16px rgba(0,113,227,0.35)", letterSpacing: 0.5,
                    }}
                  >
                    ▶ 解析開始
                  </button>
                  <div style={{ marginTop: 10, color: C.textSub, fontSize: 13 }}>
                    枠がズレている場合は「✏️ 枠を調整」で修正してから解析してください
                  </div>
                </div>
              )}

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
                <div style={{ display: "flex", justifyContent: "center", gap: 12, marginTop: 24 }}>
                  <button
                    onClick={() => { setResults(null); setError(null); }}
                    style={{ background: "transparent", color: C.accent, border: `2px solid ${C.accent}`, borderRadius: 30, padding: "14px 32px", cursor: "pointer", fontSize: 15, fontWeight: 500 }}
                  >
                    🔄 再解析
                  </button>
                  <button
                    onClick={reset}
                    style={{ background: C.accent, color: "white", border: "none", borderRadius: 30, padding: "14px 32px", cursor: "pointer", fontSize: 15, fontWeight: 500 }}
                  >
                    ＋ 新規スキャン
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        <div style={{ flex: "1 1 35%", minWidth: 400 }}>
          {!results && !viewHistory && !analyzing && (
            <div style={{ color: C.textSub, textAlign: "center", marginTop: 100, fontSize: 16 }}>
              画像をアップロードして<br />解析を開始してください
            </div>
          )}

          {results && !viewHistory && (
            <div>
              <div style={{
                background: results.status === "FAIL" ? C.dangerBg : C.successBg,
                border: `2px solid ${results.status === "FAIL" ? C.danger : C.success}`,
                borderRadius: 20, padding: "20px 28px", marginBottom: 24,
                display: "flex", alignItems: "center", justifyContent: "space-between"
              }}>
                <div style={{ fontSize: 32, fontWeight: 700, color: results.status === "FAIL" ? C.danger : C.success, letterSpacing: "-0.02em" }}>
                  {results.status}
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 12, color: C.textSub, marginBottom: 2 }}>平均空き率</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: C.text, letterSpacing: "-0.02em" }}>{results.avg}%</div>
                </div>
              </div>

              {results.areas.map((area, i) => {
                const fail = area.pct >= FAIL_THRESHOLD;
                const frameColors = ["#ff3b30", "#0071e3", "#34c759"];
                const frameColor = frameColors[i];
                return (
                  <div key={area.key} style={{
                    background: C.card, border: `1px solid ${C.border}`,
                    borderLeft: `6px solid ${frameColor}`, borderRadius: 16,
                    padding: "20px 24px", marginBottom: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
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

              {results.advice && (
                <div style={{
                  background: "#fffbea", border: `1px solid #f5d77a`,
                  borderLeft: `6px solid #f5a623`, borderRadius: 16,
                  padding: "20px 24px", marginTop: 20, boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
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
