 import cron from 'node-cron';
 import { query } from '../services/db.js';
 import { fetchForecast, summarizeNext24h } from '../services/weatherService.js';
 import { scoreRisks } from '../services/riskService.js';
 import { sendAlert } from '../services/alertService.js';
 // Every hour at minute 5
 cron.schedule('5 * * * *', async () => {
 try {
 const locs = (await query('SELECT l.*, u.email as user_email FROM locations l JOIN users u ON u.id=l.user_id')).rows;
 for (const loc of locs) {
 const forecast = await fetchForecast(loc.lat, loc.lon);
 const summary = summarizeNext24h(forecast);
 const scored = scoreRisks({ subdivision: loc.subdivision, summary });
 await query(
 'INSERT INTO risk_scores(location_id, ts, flood_risk, wildfire_risk, details) VALUES ($1, now(), $2, $3, $4)',
 [loc.id, scored.flood_risk, scored.wildfire_risk, scored.details]
 );
 // Alert thresholds (tune as you like)
 const triggers = [];
 if (scored.flood_risk >= 0.7) triggers.push('FLOOD');
 if (scored.wildfire_risk >= 0.7) triggers.push('WILDFIRE');
 
for (const type of triggers) {
 const severity = scored[type.toLowerCase() + '_risk'] >= 0.85 ?
 'HIGH' : 'MEDIUM';
 const msg = `${type} risk ${severity} at ${loc.name}. Flood=$
 {scored.flood_risk}, Fire=${scored.wildfire_risk}`;
 await query(
 'INSERT INTO alerts(location_id, ts, type, severity, message, delivered_via) VALUES ($1, now(), $2, $3, $4, $5)',
 [loc.id, type, severity, msg, JSON.stringify(['email'])]
 );
 await sendAlert({ to: process.env.ALERT_EMAIL_TO || loc.user_email,
 subject: `[EarthPulse] ${type} ${severity}`, text: msg });
 }
 }
 } catch (e) {
 console.error('Scheduler error', e.message);
 }
 });