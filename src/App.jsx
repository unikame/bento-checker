import { useState, useRef, useCallback } from "react";

const AREAS = [
  { name: "右上（メイン）", key: "main" },
  { name: "左上（副菜A）", key: "sub1" },
  { name: "右下（副菜B）", key: "sub2" },
];

const CROP_DEFS = [
  [0.08, 0.48, 0.35, 0.95],  // 右上メイン: y1, y2, x1, x2
  [0.08, 0.48, 0.05, 0.35],  // 左上副菜A
  [0.5,  0.95, 0.52, 0.95],  // 右下副菜B
];

// OffscreenCanvas で File を直接クロップ → base64（CORSを完全回避）
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

async function analyzeArea(b64, areaName, apiKey) {
  const prompt = `You are inspecting "${areaName}" in a bento tray.

STEP 1: Describe what you see
Look at the image. What food items do you see? Are they in paper cups or placed directly on the red/brown tray?

STEP 2: Check for direct placement
Is there ANY food sitting directly on the tray surface (not inside a decorative paper cup)?
- Hamburg steak, korokke (croquette), tamagoyaki (egg roll), rice = DIRECT on tray
- Food only in colorful paper cups = NOT direct

STEP 3: If DIRECT placement exists:
Look for LARGE EMPTY RED/BROWN TRAY AREAS with no food and no cup.
- If you see a big empty gap on the right side, left side, or anywhere = 15-25%
- Small gap = ~15%
- Medium gap = ~20%
- Large gap = ~25%

STEP 4: If ONLY cups (no direct food):
Return: {"pct": 0, "reason": "すべておかずカップ内に入っており問題なし"}

IMPORTANT:
- Paper cup gaps are normal, don't count them
- Only count empty tray where food SHOULD be placed directly

Return ONLY JSON: {"pct": number, "reason": "what you see in Japanese"}`;

  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [
        {
          role: "user",
          content: [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: prompt },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message || `API error ${res.status}`);
  }

  const data = await res.json();
  const text = data.content?.[0]?.text || "{}";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("JSON not found in response");
  return JSON.parse(match[0]);
}

export default function BentoCheckerPro() {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('apiKey') || "");
  const [apiKeySet, setApiKeySet] = useState(true);
  const [imgSrc, setImgSrc] = useState(null);
  const [imgFile, setImgFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [viewHistory, setViewHistory] = useState(false);
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
    if (!imgFile || !apiKey) return;
    setAnalyzing(true);
    setError(null);
    setProgress(0);
    setProgressLabel("");

    try {
      const areaResults = [];
      for (let i = 0; i < AREAS.length; i++) {
        setProgressLabel(`${AREAS[i].name} を解析中...`);
        setProgress(Math.round((i / AREAS.length) * 100));

        const [y1r, y2r, x1r, x2r] = CROP_DEFS[i];
        const b64 = await cropFileToBase64(imgFile, x1r, y1r, x2r, y2r);
        const res = await analyzeArea(b64, AREAS[i].name, apiKey);
        areaResults.push({ ...AREAS[i], pct: res.pct ?? 0, reason: res.reason ?? "" });
        setProgress(Math.round(((i + 1) / AREAS.length) * 100));
      }

      setProgressLabel("完了！");
      const avg = areaResults.reduce((s, r) => s + r.pct, 0) / areaResults.length;
      const isFail = areaResults.some((r) => r.pct >= 15);
      const record = {
        id: Date.now(),
        time: new Date().toLocaleString("ja-JP"),
        imgSrc,
        areas: areaResults,
        avg: Math.round(avg * 10) / 10,
        status: isFail ? "FAIL" : "PASS",
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
  }, [apiKey, imgFile, imgSrc]);

  const reset = () => {
    setImgSrc(null);
    setImgFile(null);
    setResults(null);
    setError(null);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#1a1510", fontFamily: "'Georgia', serif", color: "#f0ede8" }}>
      <div style={{ borderBottom: "1px solid #3a2e1e", padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 28 }}>🍱</span>
          <div>
            <div style={{ fontFamily: "monospace", fontSize: 18, fontWeight: 700, letterSpacing: 2, color: "#e8c97a" }}>BENTO CHECKER PRO</div>
            <div style={{ fontSize: 11, color: "#7a6a50", letterSpacing: 1 }}>AI-POWERED QUALITY INSPECTION</div>
          </div>
        </div>
        <button
          onClick={() => setViewHistory(!viewHistory)}
          style={{ background: "transparent", border: "1px solid #3a2e1e", color: "#a89060", borderRadius: 6, padding: "6px 14px", cursor: "pointer", fontSize: 13, fontFamily: "monospace" }}
        >
          📋 履歴 ({history.length})
        </button>
      </div>

      <div style={{ display: "flex", minHeight: "calc(100vh - 61px)" }}>
        <div style={{ flex: 1, padding: 24, borderRight: "1px solid #3a2e1e" }}>
          {!apiKeySet && (
            <div style={{ background: "#231d12", border: "1px solid #3a2e1e", borderRadius: 10, padding: 20, marginBottom: 24 }}>
              <div style={{ fontSize: 12, color: "#a89060", letterSpacing: 1, marginBottom: 10, fontFamily: "monospace" }}>ANTHROPIC API KEY</div>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type="password"
                  placeholder="sk-ant-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && apiKey.startsWith("sk-")) { localStorage.setItem('apiKey', apiKey); setApiKeySet(true); } }}
                  style={{ flex: 1, background: "#1a1510", border: "1px solid #3a2e1e", color: "#f0ede8", borderRadius: 6, padding: "8px 12px", fontSize: 13, fontFamily: "monospace", outline: "none" }}
                />
                <button
                  onClick={() => { if (apiKey.startsWith("sk-")) { localStorage.setItem('apiKey', apiKey); setApiKeySet(true); } }}
                  style={{ background: "#e8c97a", color: "#1a1510", border: "none", borderRadius: 6, padding: "8px 16px", cursor: "pointer", fontFamily: "monospace", fontWeight: 700, fontSize: 13 }}
                >SET</button>
              </div>
            </div>
          )}

          {apiKeySet && (
            <div style={{ fontSize: 12, color: "#4a8a5a", fontFamily: "monospace", marginBottom: 16, display: "flex", justifyContent: "space-between" }}>
              <span>✓ API KEY SET</span>
              <span onClick={() => { setApiKeySet(false); setApiKey(""); }} style={{ color: "#f0ede8", cursor: "pointer", textDecoration: "underline" }}>変更</span>
            </div>
          )}

          {!imgSrc && (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileRef.current?.click()}
              style={{ border: "2px dashed #3a2e1e", borderRadius: 12, padding: 48, textAlign: "center", cursor: "pointer" }}
            >
              <div style={{ fontSize: 48, marginBottom: 12 }}>📷</div>
              <div style={{ color: "#a89060", fontSize: 14, marginBottom: 6 }}>お弁当の写真をドロップ</div>
              <div style={{ color: "#f0ede8", fontSize: 12 }}>または クリックして選択</div>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
            </div>
          )}

          {imgSrc && (
            <div>
              <img src={imgSrc} alt="bento" style={{ width: "50%", borderRadius: 10, border: "1px solid #3a2e1e", display: "block", margin: "0 auto" }} />

              {analyzing && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12, fontFamily: "monospace", color: "#a89060" }}>
                    <span>{progressLabel}</span>
                    <span>{progress}%</span>
                  </div>
                  <div style={{ background: "#231d12", borderRadius: 4, height: 6, overflow: "hidden" }}>
                    <div style={{ width: `${progress}%`, height: "100%", background: "#e8c97a", transition: "width 0.4s ease" }} />
                  </div>
                </div>
              )}

              {error && (
                <div style={{ marginTop: 16, background: "#2a1010", border: "1px solid #5a2020", borderRadius: 8, padding: 12, color: "#e05050", fontSize: 13, wordBreak: "break-all" }}>
                  ⚠️ {error}
                </div>
              )}

              {!analyzing && !results && (
                <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                  <button
                    onClick={analyze}
                    disabled={!apiKeySet}
                    style={{ flex: 1, background: apiKeySet ? "#e8c97a" : "#3a2e1e", color: apiKeySet ? "#1a1510" : "#5a4a30", border: "none", borderRadius: 8, padding: "12px 0", cursor: apiKeySet ? "pointer" : "not-allowed", fontFamily: "monospace", fontWeight: 700, fontSize: 15, letterSpacing: 1 }}
                  >🔍 解析開始</button>
                  <button onClick={reset} style={{ background: "transparent", border: "1px solid #3a2e1e", color: "#7a6a50", borderRadius: 8, padding: "12px 18px", cursor: "pointer", fontFamily: "monospace" }}>✕</button>
                </div>
              )}

              {results && (
                <button onClick={reset} style={{ width: "50%", marginTop: 14, background: "transparent", border: "1px solid #3a2e1e", color: "#a89060", borderRadius: 8, padding: "10px 0", cursor: "pointer", fontFamily: "monospace", fontSize: 13 }}>
                  ＋ 新規スキャン
                </button>
              )}
            </div>
          )}
        </div>

        <div style={{ width: 320, padding: 24, overflowY: "auto" }}>
          {!results && !viewHistory && (
            <div style={{ color: "#3a2e1e", textAlign: "center", marginTop: 60, fontSize: 13, fontFamily: "monospace" }}>
              画像をアップロードして<br />解析を開始してください
            </div>
          )}

          {results && !viewHistory && (
            <div>
              <div style={{
                background: results.status === "FAIL" ? "#2a1010" : "#0a2010",
                border: `2px solid ${results.status === "FAIL" ? "#c03030" : "#30a060"}`,
                borderRadius: 12, padding: "16px 20px", marginBottom: 20,
                display: "flex", alignItems: "center", justifyContent: "space-between"
              }}>
                <div style={{ fontFamily: "monospace", fontSize: 28, fontWeight: 700, color: results.status === "FAIL" ? "#e05050" : "#40c070" }}>
                  {results.status}
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 11, color: "#7a6a50", fontFamily: "monospace" }}>平均空き率</div>
                  <div style={{ fontFamily: "monospace", fontSize: 24, fontWeight: 700, color: "#e8c97a" }}>{results.avg}%</div>
                </div>
              </div>

              {results.areas.map((area) => {
                const fail = area.pct >= 15;
                return (
                  <div key={area.key} style={{
                    background: "#231d12",
                    border: `1px solid ${fail ? "#5a2020" : "#1a3a1a"}`,
                    borderLeft: `4px solid ${fail ? "#c03030" : "#30a060"}`,
                    borderRadius: 8, padding: "14px 16px", marginBottom: 12,
                  }}>
                    <div style={{ fontSize: 11, color: "#7a6a50", fontFamily: "monospace", marginBottom: 4 }}>{area.name}</div>
                    <div style={{ fontFamily: "monospace", fontSize: 22, fontWeight: 700, color: fail ? "#e05050" : "#40c070", marginBottom: 6 }}>{area.pct}%</div>
                    <div style={{ background: "#1a1510", borderRadius: 4, height: 4, marginBottom: 8 }}>
                      <div style={{ width: `${Math.min(area.pct, 100)}%`, height: "100%", background: fail ? "#c03030" : "#30a060", borderRadius: 4, transition: "width 0.5s" }} />
                    </div>
                    <div style={{ fontSize: 11, color: "#f0ede8", lineHeight: 1.5 }}>{area.reason}</div>
                  </div>
                );
              })}

              <div style={{ fontSize: 11, color: "#3a2e1e", fontFamily: "monospace", marginTop: 8 }}>{results.time}</div>
            </div>
          )}

          {viewHistory && (
            <div>
              <div style={{ fontFamily: "monospace", fontSize: 13, color: "#a89060", marginBottom: 16, letterSpacing: 1 }}>INSPECTION HISTORY</div>
              {history.length === 0 && <div style={{ color: "#3a2e1e", fontSize: 13, fontFamily: "monospace" }}>履歴なし</div>}
              {history.map((rec) => (
                <div key={rec.id} style={{ background: "#231d12", border: "1px solid #3a2e1e", borderRadius: 8, padding: 12, marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={{ fontFamily: "monospace", fontWeight: 700, color: rec.status === "FAIL" ? "#e05050" : "#40c070", fontSize: 14 }}>{rec.status}</span>
                    <span style={{ fontFamily: "monospace", fontSize: 18, color: "#e8c97a" }}>{rec.avg}%</span>
                  </div>
                  <img src={rec.imgSrc} alt="" style={{ width: "50%", borderRadius: 6, marginBottom: 6 }} />
                  <div style={{ fontSize: 11, color: "#f0ede8", fontFamily: "monospace" }}>{rec.time}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
