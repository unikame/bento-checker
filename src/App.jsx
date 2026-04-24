import { useState, useRef, useCallback, useEffect } from "react";

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
  [0.085, 0.434, 0.079, 0.338],  // 左上（副菜A）
  [0.479, 0.894, 0.661, 0.923],  // 右下（副菜B）
];

// AIで容器と各区画の位置を検出
async function detectRegions(file) {
  const bitmap = await createImageBitmap(file);
  const maxSize = 1000;
  const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const canvas = new OffscreenCanvas(
    Math.floor(bitmap.width * scale),
    Math.floor(bitmap.height * scale)
  );
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.85 });
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error("FileReader error"));
    reader.readAsDataURL(blob);
  });

  const prompt = `この弁当画像の各区画を精密に検出してください。

【容器の構造】
赤い4区画の弁当容器です。区画は黒い仕切り線で分かれています。
- 上段: 左に小さい正方形に近い区画（副菜A）、右に大きい横長の区画（メイン）
- 下段: 左に大きい区画（ご飯）、右に小さい縦長の区画（副菜B）

【参考座標（標準的な撮影位置での正解例）】
同じ容器を真上から標準的に撮影した場合、以下の座標になります:
- 右上（メイン）: x1=0.376, y1=0.085, x2=0.926, y2=0.433
- 左上（副菜A）: x1=0.079, y1=0.085, x2=0.338, y2=0.434
- 右下（副菜B）: x1=0.661, y1=0.479, x2=0.923, y2=0.894

【今回の画像での検出方法】
今回アップロードされた画像で、上記の参考座標と比べて:
- 容器が画像のどの位置に写っているか確認
- 参考座標を基準に、実際の位置に合わせて調整
- 仕切り線の内側にぴったり合わせる

【重要な指示】
- 参考座標は一例です。実際の画像に合わせて調整してください
- 容器が中央寄りに写っていれば参考座標に近い値になります
- 食材ではなく、区画の壁（仕切り線）の位置を基準にしてください
- 隣の区画に食み出さないでください

JSONのみで返してください:
{
  "main": {"x1": 数値, "y1": 数値, "x2": 数値, "y2": 数値},
  "sub1": {"x1": 数値, "y1": 数値, "x2": 数値, "y2": 数値},
  "sub2": {"x1": 数値, "y1": 数値, "x2": 数値, "y2": 数値}
}`;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 500,
        messages: [{
          role: "user",
          content: [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: prompt },
          ],
        }],
      }),
    });

    if (!res.ok) return DEFAULT_CROP_DEFS;

    const data = await res.json();
    const text = data.content?.[0]?.text || "{}";
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) return DEFAULT_CROP_DEFS;

    const regions = JSON.parse(match[0]);
    const valid = (r) => r && typeof r.x1 === "number" && typeof r.y1 === "number"
      && typeof r.x2 === "number" && typeof r.y2 === "number"
      && r.x1 >= 0 && r.x2 <= 1 && r.y1 >= 0 && r.y2 <= 1
      && r.x1 < r.x2 && r.y1 < r.y2;

    if (!valid(regions.main) || !valid(regions.sub1) || !valid(regions.sub2)) {
      return DEFAULT_CROP_DEFS;
    }

    return [
      [regions.main.y1, regions.main.y2, regions.main.x1, regions.main.x2],
      [regions.sub1.y1, regions.sub1.y2, regions.sub1.x1, regions.sub1.x2],
      [regions.sub2.y1, regions.sub2.y2, regions.sub2.x1, regions.sub2.x2],
    ];
  } catch {
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
以下を日本語で具体的に答えてください（数値は出さないこと）:

1. どの食材が見えるか？
2. 食材の配置は？（中央/片寄り/等間隔など）
3. 赤茶色のトレー底面が見える場所は？（上/下/左/右/どの部分にどのくらい）
4. 食材同士の間に隙間はあるか？
5. 全体の印象として「ぎっしり」「やや余裕あり」「明確な空きあり」「スカスカ」のどれか？

【分類の厳しい基準】
- ぎっしり: トレー底面が全く見えない、縁までびっしり
- やや余裕あり: 縁に沿って細い隙間がある、小さな隙間がポツポツ
- 明確な空きあり: 片側に明らかな空きスペース、食材が片寄っている、トレー底面が明確に見える部分がある
- スカスカ: 半分近くが空いている

厳しめに判定してください。少しでも空きが気になる場合は「明確な空きあり」を選んでください。

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

JSONで返してください: {"pct": <${range[0]}〜${range[1]}の整数>, "reason": "観察結果の要約（日本語100文字以内）"}`;

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
  const [cropDefs, setCropDefs] = useState(DEFAULT_CROP_DEFS);
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
      // STEP 0: 容器の位置を自動検出
      setProgressLabel("容器の位置を検出中...");
      const cropDefs = await detectRegions(imgFile);
      setCropDefs(cropDefs);

      const areaResults = [];
      for (let i = 0; i < AREAS.length; i++) {
        setProgressLabel(`${AREAS[i].name} を解析中...`);
        setProgress(Math.round(((i + 1) / (AREAS.length + 1)) * 100));

        const [y1r, y2r, x1r, x2r] = cropDefs[i];
        const b64 = await cropFileToBase64(imgFile, x1r, y1r, x2r, y2r);
        const res = await analyzeArea(b64, AREAS[i].name);
        areaResults.push({ ...AREAS[i], pct: res.pct ?? 0, reason: res.reason ?? "" });
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
        <button
          onClick={() => setViewHistory(!viewHistory)}
          style={{ background: "transparent", border: `1px solid ${C.border}`, color: C.text, borderRadius: 20, padding: "8px 18px", cursor: "pointer", fontSize: 14, fontWeight: 500 }}
        >
          📋 履歴 ({history.length})
        </button>
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
                <div style={{ position: "relative", display: "inline-block", width: "85%" }}>
                  <img
                    src={imgSrc}
                    alt="bento"
                    style={{ width: "100%", borderRadius: 12, display: "block" }}
                  />
                  {/* 3つのエリア枠オーバーレイ */}
                  {cropDefs.map((def, i) => {
                    const [y1r, y2r, x1r, x2r] = def;
                    const colors = ["#ff3b30", "#0071e3", "#34c759"];
                    const labels = ["右上（メイン）", "左上（副菜A）", "右下（副菜B）"];
                    const pct = results?.areas?.[i]?.pct;
                    return (
                      <div
                        key={i}
                        style={{
                          position: "absolute",
                          left: `${x1r * 100}%`,
                          top: `${y1r * 100}%`,
                          width: `${(x2r - x1r) * 100}%`,
                          height: `${(y2r - y1r) * 100}%`,
                          border: `3px solid ${colors[i]}`,
                          borderRadius: 8,
                          boxSizing: "border-box",
                          pointerEvents: "none",
                        }}
                      >
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
                        }}>
                          {labels[i]}{pct !== undefined ? ` ${pct}%` : ""}
                        </div>
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
