import { Router } from "express";
import { query } from "../services/db.js";
import { requireAuth } from "../middleware/auth.js";

const router = Router();

// -------------------- GET LOCATIONS --------------------
router.get("/", requireAuth, async (req, res) => {
  try {
    const r = await query(
      "SELECT * FROM locations WHERE user_id=$1 ORDER BY id DESC",
      [req.user.id]
    );
    res.json(r.rows);
  } catch (e) {
    console.error("Fetch locations error:", e.message);
    res.status(500).json({ error: "Failed to fetch locations" });
  }
});

// -------------------- ADD LOCATION --------------------
router.post("/", requireAuth, async (req, res) => {
  try {
    const { name, subdivision, lat, lon } = req.body;

    // Validate required fields
    if (!name || !subdivision || lat == null || lon == null) {
      return res.status(400).json({ error: "Name, subdivision, lat, and lon are required" });
    }

    // Validate lat/lon as numbers within correct ranges
    const latitude = parseFloat(lat);
    const longitude = parseFloat(lon);

    if (isNaN(latitude) || isNaN(longitude)) {
      return res.status(400).json({ error: "Latitude and longitude must be valid numbers" });
    }
    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
      return res.status(400).json({ error: "Latitude must be between -90 and 90, longitude between -180 and 180" });
    }

    // Optional: limit string sizes
    if (name.length > 100 || subdivision.length > 100) {
      return res.status(400).json({ error: "Name and subdivision must be under 100 characters" });
    }

    const r = await query(
      "INSERT INTO locations(user_id, name, subdivision, lat, lon) VALUES ($1, $2, $3, $4, $5) RETURNING *",
      [req.user.id, name, subdivision, latitude, longitude]
    );

    res.status(201).json(r.rows[0]);
  } catch (e) {
    console.error("Add location error:", e.message);
    res.status(500).json({ error: "Failed to add location" });
  }
});

export default router;
