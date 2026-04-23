import { useState, useRef, useCallback } from "react";

const AREAS = [
  { name: "右上（メイン）", key: "main" },
  { name: "左上（副菜A）", key: "sub1" },
  { name: "右下（副菜B）", key: "sub2" },
];

const CROP_DEFS = [
  [0.35, 0.08, 0.95, 0.48],
  [0.05, 0.08, 0.35, 0.48],
  [0.52, 0.5,  0.95, 0.95],
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
  const prompt = `You are a strict quality inspector analyzing a section of a Japanese bento box tray called "${areaName}".

Your job: Estimate the percentage of EMPTY space where food SHOULD be placed but is NOT.

RULES:
- Count empty space = visible tray bottom, gaps between foods, areas with no food
- Do NOT count the tray border/frame/edge as empty (only the food placement area inside)
- Be strict: even a small gap on one side of the compartment counts
- If food is pushed to one side leaving visible space on the other side, that IS empty space
- A side gap of about 1/4 of the compartment width = approximately 20-25% empty
- A side gap of about 1/5 of the compartment width = approximately 15-20% empty
- NEVER underestimate: when in doubt, round UP

Respond ONLY with valid JSON: {"pct": number, "reason": "string in Japanese"}`;

  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-opus-4-6",
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
  const [apiKey, setApiKey] = useState("");
  const [apiKeySet, setApiKeySet] = useState(false);
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

        const [x1r, y1r, x2r, y2r] = CROP_DEFS[i];
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
                  onKeyDown={(e) => e.key === "Enter" && apiKey.startsWith("sk-") && setApiKeySet(true)}
                  style={{ flex: 1, background: "#1a1510", border: "1px solid #3a2e1e", color: "#f0ede8", borderRadius: 6, padding: "8px 12px", fontSize: 13, fontFamily: "monospace", outline: "none" }}
                />
                <button
                  onClick={() => apiKey.startsWith("sk-") && setApiKeySet(true)}
                  style={{ background: "#e8c97a", color: "#1a1510", border: "none", borderRadius: 6, padding: "8px 16px", cursor: "pointer", fontFamily: "monospace", fontWeight: 700, fontSize: 13 }}
                >SET</button>
              </div>
            </div>
          )}

          {apiKeySet && (
            <div style={{ fontSize: 12, color: "#4a8a5a", fontFamily: "monospace", marginBottom: 16, display: "flex", justifyContent: "space-between" }}>
              <span>✓ API KEY SET</span>
              <span onClick={() => { setApiKeySet(false); setApiKey(""); }} style={{ color: "#5a4a30", cursor: "pointer", textDecoration: "underline" }}>変更</span>
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
              <div style={{ color: "#5a4a30", fontSize: 12 }}>または クリックして選択</div>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
            </div>
          )}

          {imgSrc && (
            <div>
              <img src={imgSrc} alt="bento" style={{ width: "100%", borderRadius: 10, border: "1px solid #3a2e1e", display: "block" }} />

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
                <button onClick={reset} style={{ width: "100%", marginTop: 14, background: "transparent", border: "1px solid #3a2e1e", color: "#a89060", borderRadius: 8, padding: "10px 0", cursor: "pointer", fontFamily: "monospace", fontSize: 13 }}>
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
                    <div style={{ fontSize: 11, color: "#5a4a30", lineHeight: 1.5 }}>{area.reason}</div>
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
                  <img src={rec.imgSrc} alt="" style={{ width: "100%", borderRadius: 6, marginBottom: 6 }} />
                  <div style={{ fontSize: 11, color: "#5a4a30", fontFamily: "monospace" }}>{rec.time}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
