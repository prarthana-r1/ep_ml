import { Router } from "express";
import { fetchForecast, summarizeNext24h } from "../services/weatherService.js";
// import { requireAuth } from "../middleware/auth.js"; // uncomment if you want auth

const router = Router();

router.get("/forecast", async (req, res) => {
  const { lat, lon } = req.query;

  if (!lat || !lon) {
    return res.status(400).json({ error: "lat and lon required" });
  }

  try {
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);
    if (isNaN(latNum) || isNaN(lonNum)) {
      return res.status(400).json({ error: "Invalid lat/lon values" });
    }

    // 1️⃣ Get forecast
    const forecast = await fetchForecast(latNum, lonNum);
    if (!forecast || !forecast.hourly) {
      return res.status(502).json({ error: "Forecast service unavailable" });
    }

    // 2️⃣ Summarize forecast
    const summary = summarizeNext24h(forecast);

    // 3️⃣ Send response
    res.json({
      location: { lat: latNum, lon: lonNum },
      summary,
      raw: forecast,
    });
  } catch (err) {
    console.error("Weather route error:", err.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;
