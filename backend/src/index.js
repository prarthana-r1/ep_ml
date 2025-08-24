import dotenv from 'dotenv';
dotenv.config();   // 👈 load first!

import express from 'express';
import cors from 'cors';
import authRoutes from './routes/auth.js';
import locationRoutes from './routes/locations.js';
import weatherRoutes from './routes/weather.js';
import riskRoutes from './routes/risk.js';
import metaRoutes from './routes/meta.js';
import './jobs/riskScheduler.js'; // starts the periodic risk computation

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/auth', authRoutes);
app.use('/api/locations', locationRoutes);
app.use('/api/weather', weatherRoutes);
app.use('/api/risk', riskRoutes);
app.use('/api/meta', metaRoutes);

const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`EarthPulse API running on :${port}`));


// import { fileURLToPath } from "url";
// import { dirname, join } from "path";

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = dirname(__filename);

// dotenv.config({ path: join(__dirname, "../.env") }); // force load

// console.log("PORT:", process.env.PORT);
// console.log("DATABASE_URL:", process.env.DATABASE_URL);
// console.log("JWT_SECRET:", process.env.JWT_SECRET);

