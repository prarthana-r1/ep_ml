 import axios from 'axios';
 const OWM_BASE = 'https://api.openweathermap.org/data/2.5';
 9
export async function fetchForecast(lat, lon) {
 const url = `${OWM_BASE}/forecast`;
 const params = { lat, lon, appid: process.env.OWM_API_KEY, units: 'metric' };
 const { data } = await axios.get(url, { params });
 return data; // has .list of 3h steps
 }
 export function summarizeNext24h(forecast) {
 const now = Date.now();
 const cutoff = now + 24 * 3600 * 1000;
 const next24 = forecast.list.filter(x => new Date(x.dt * 1000).getTime() <=
 cutoff);
 const sum = (arr, sel) => arr.reduce((a, x) => a + (sel(x) || 0), 0);
 const avg = (arr, sel) => arr.length ? sum(arr, sel) / arr.length : 0;
 const rain24 = sum(next24, x => x.rain?.['3h']);
 const humidity = avg(next24, x => x.main?.humidity);
 const temp = avg(next24, x => x.main?.temp);
 const wind = avg(next24, x => x.wind?.speed);
 const pop = avg(next24, x => x.pop);
 return { rain24: rain24 || 0, humidity: humidity || 0, temp: temp || 0, wind:
 wind || 0, pop: pop || 0 };
 }