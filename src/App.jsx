const [y1r, y2r, x1r, x2r] = cropDefs[i];
        const b64 = await cropFileToBase64(targetFile, x1r, y1r, x2r, y2r);

        // 空白率はピクセル検出で機械的に算出（AIに任せない＝ブレない）
        const pixel = await calcEmptyRateByPixel(targetFile, x1r, y1r, x2r, y2r);

        // 食材名のみAIに取得させる（1回でOK・数値に影響しないため）
        const ai = await analyzeArea(b64, AREAS[i].name);

        areaResults.push({
          ...AREAS[i],
          pct: pixel.rate,
          reason: ai.reason,
          empty_boxes: []
        });
