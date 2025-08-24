import { Router } from "express";
import { fetchForecast, summarizeNext24h } from "../services/weatherService.js";
import { query } from "../services/db.js";
import { requireAuth } from "../middleware/auth.js";
import axios from "axios";

const router = Router();

router.get("/", requireAuth, async (req, res) => {
  const { locationId } = req.query;
  if (!locationId) return res.status(400).json({ error: "locationId required" });

  const loc = (
    await query("SELECT * FROM locations WHERE id=$1 AND user_id=$2", [locationId, req.user.id])
  ).rows[0];
  if (!loc) return res.status(404).json({ error: "Location not found" });

  try {
    // 1️⃣ Weather forecast
    const forecast = await fetchForecast(loc.lat, loc.lon);
    const summary = summarizeNext24h(forecast);

    // 2️⃣ Fire data from NASA FIRMS (⚠️ you need your FIRMS API key)
    let brightness = 0, bright_t31 = 0, frp = 0;
    try {
      const fireResp = await axios.get(
        `https://firms.modaps.eosdis.nasa.gov/api/area/csv/YOUR_API_KEY/VIIRS_SNPP_NRT/${loc.lat},${loc.lon},0.5` // lat,lon,buffer
      );
      const fireLines = fireResp.data.split("\n").filter(l => l && !l.startsWith("latitude"));
      if (fireLines.length > 0) {
        const headers = fireResp.data.split("\n")[0].split(",");
        const first = fireLines[0].split(",");
        brightness = parseFloat(first[headers.indexOf("brightness")]) || 0;
        bright_t31 = parseFloat(first[headers.indexOf("bright_t31")]) || 0;
        frp = parseFloat(first[headers.indexOf("frp")]) || 0;
      }
    } catch (fireErr) {
      console.warn("⚠️ FIRMS fetch failed:", fireErr.message);
    }

    // 3️⃣ Call Flask ML API
    const flaskResp = await axios.post("http://127.0.0.1:5000/api/risk", {
      rain24: summary.rain24,
      humidity: summary.humidity,
      temp: summary.temp,
      wind: summary.wind,
      pop: summary.pop,
      brightness,
      bright_t31,
      frp,
    });

    const { flood_risk, wildfire_risk, wildfire_level } = flaskResp.data;

    // 4️⃣ Save snapshot into DB (store both weather + fire info in details JSONB)
    await query(
      "INSERT INTO risk_scores(location_id, ts, flood_risk, wildfire_risk, details) VALUES ($1, now(), $2, $3, $4)",
      [
        loc.id,
        flood_risk,
        wildfire_risk,
        {
          ...summary,
          brightness,
          bright_t31,
          frp,
        },
      ]
    );

    // 5️⃣ Respond to frontend
    res.json({
      location: loc,
      risks: { flood_risk, wildfire_risk, wildfire_level },
      details: { ...summary, brightness, bright_t31, frp },
    });
  } catch (err) {
    console.error("Risk route error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;
