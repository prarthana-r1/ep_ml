import { useEffect, useState } from 'react';
 import { api } from '../api/client';
 export default function AddLocationForm({ onAdd }) {
 const [subs, setSubs] = useState([]);
 const [f, setF] = useState({ name: '', subdivision: '', lat: 12.9716, lon:
 77.5946 });
 useEffect(() => { api.get('/meta/subdivisions').then(r =>
 setSubs(r.data)); }, []);
 const change = e => setF(s => ({ ...s, [e.target.name]: e.target.value }));
 const submit = e => { e.preventDefault(); onAdd({ ...f, lat: Number(f.lat),
 lon: Number(f.lon) }); };
 return (
 <form onSubmit={submit} style={{ display: 'grid', gap: 8 }}>
 <h3>Add Location</h3>
 <input name="name" placeholder="Label (e.g., Home)" value={f.name}
 onChange={change} required />
 <select name="subdivision" value={f.subdivision} onChange={change}
 required>
 <option value="">Choose IMD Subdivision</option>
 {subs.map(s => <option key={s} value={s}>{s}</option>)}
 </select>
 <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
 <input name="lat" type="number" step="0.0001" value={f.lat}
 onChange={change} />
 <input name="lon" type="number" step="0.0001" value={f.lon}
 onChange={change} />
 </div>
 <button type="submit">Save Location</button>
 </form>
 );
 }