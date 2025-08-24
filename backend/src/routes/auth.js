import { Router } from "express";
import { query } from "../services/db.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

const router = Router();
const SALT_ROUNDS = parseInt(process.env.SALT_ROUNDS || "10", 10);

// -------------------- REGISTER --------------------
router.post("/register", async (req, res) => {
  try {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ error: "Name, email, and password required" });
    }

    // Basic email format check
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
      return res.status(400).json({ error: "Invalid email format" });
    }

    // Check duplicate email
    const exists = await query("SELECT id FROM users WHERE email=$1", [email]);
    if (exists.rows.length > 0) {
      return res.status(400).json({ error: "Email already registered" });
    }

    const hash = await bcrypt.hash(password, SALT_ROUNDS);

    await query(
      "INSERT INTO users(name, email, password_hash) VALUES($1, $2, $3)",
      [name, email, hash]
    );

    return res.json({ ok: true, message: "User registered successfully" });
  } catch (e) {
    console.error("Register error:", e.message);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// -------------------- LOGIN --------------------
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password required" });
    }

    const r = await query("SELECT * FROM users WHERE email=$1", [email]);
    const u = r.rows[0];
    if (!u) {
      return res.status(400).json({ error: "User not found" });
    }

    const ok = await bcrypt.compare(password, u.password_hash);
    if (!ok) {
      return res.status(400).json({ error: "Invalid credentials" });
    }

    const token = jwt.sign(
      { id: u.id, email: u.email },
      process.env.JWT_SECRET,
      { expiresIn: "7d" }
    );

    return res.json({ token });
  } catch (e) {
    console.error("Login error:", e.message);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export default router;
