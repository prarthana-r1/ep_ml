-- PostgreSQL schema for EarthPulse
-- Users
CREATE TABLE IF NOT EXISTS users (
id SERIAL PRIMARY KEY,
name TEXT NOT NULL,
email TEXT UNIQUE NOT NULL,
password_hash TEXT NOT NULL,
phone TEXT,
alert_preferences JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now()
);


-- Locations saved by users
CREATE TABLE IF NOT EXISTS locations (
id SERIAL PRIMARY KEY,
user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
name TEXT NOT NULL,
subdivision TEXT NOT NULL, -- must match an item from subdivisions.json
lat DOUBLE PRECISION NOT NULL,
lon DOUBLE PRECISION NOT NULL,
created_at TIMESTAMP DEFAULT now()
);


-- Weather observations (from OWM fetch)
CREATE TABLE IF NOT EXISTS weather_observations (
id BIGSERIAL PRIMARY KEY,
location_id INTEGER REFERENCES locations(id) ON DELETE CASCADE,
ts TIMESTAMP NOT NULL,
temp_c DOUBLE PRECISION,
humidity DOUBLE PRECISION,
wind_speed DOUBLE PRECISION,
rain_3h DOUBLE PRECISION,
raw JSONB NOT NULL
);


-- Risk scores (computed from OWM + historical baselines)
CREATE TABLE IF NOT EXISTS risk_scores (
id BIGSERIAL PRIMARY KEY,
location_id INTEGER REFERENCES locations(id) ON DELETE CASCADE,
ts TIMESTAMP NOT NULL,
flood_risk DOUBLE PRECISION NOT NULL,
wildfire_risk DOUBLE PRECISION NOT NULL,
details JSONB NOT NULL
);


-- Alerts issued by engine
CREATE TABLE IF NOT EXISTS alerts (
id BIGSERIAL PRIMARY KEY,
location_id INTEGER REFERENCES locations(id) ON DELETE CASCADE,
ts TIMESTAMP NOT NULL,
type TEXT NOT NULL, -- 'FLOOD' | 'WILDFIRE'
severity TEXT NOT NULL, -- 'LOW' | 'MEDIUM' | 'HIGH'
message TEXT NOT NULL,
delivered_via JSONB DEFAULT '[]'::jsonb,
acknowledged BOOLEAN DEFAULT FALSE
);


CREATE INDEX IF NOT EXISTS idx_weather_loc_ts ON weather_observations(location_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_risk_loc_ts ON risk_scores(location_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_loc_ts ON alerts(location_id, ts DESC);