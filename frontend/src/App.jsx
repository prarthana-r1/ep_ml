import { useEffect, useMemo, useState } from 'react';
import { api, setAuth } from './api/client';
import MapView from './components/MapView';
import RiskPanel from './components/RiskPanel';
import AddLocationForm from './components/AddLocationForm';

export default function App() {
  const [token, setToken] = useState(null);
  const [locations, setLocations] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [risks, setRisks] = useState(null); // new state for risk scores

  const selected = useMemo(() => locations.find(l => l.id === selectedId),
    [locations, selectedId]
  );

  useEffect(() => {
    setAuth(token);
    if (token) refresh();
  }, [token]);

  async function registerAndLogin() {
    // quick demo user (use proper auth UI in production)
    const email = `demo-${Math.random().toString(36).slice(2)}@local`;
    const pw = 'password';
    await api.post('/auth/register', { name: 'Demo', email, password: pw }).catch(() => {});
    const { data } = await api.post('/auth/login', { email, password: pw });
    setToken(data.token);
  }

  async function refresh() {
    const { data } = await api.get('/locations');
    setLocations(data);
    if (!selectedId && data[0]) setSelectedId(data[0].id);
  }

  async function addLocation(loc) {
    await api.post('/locations', loc);
    await refresh();
  }

 async function computeRisk() {
  if (!selected) return;
  setRisks("loading"); // temporary marker

  try {
    const { data } = await api.get("/risk", {
      params: { lat: selected.lat, lon: selected.lon },
    });
    setRisks(data); // backend returns full object with flood/wildfire risks
  } catch (err) {
    console.error("Error computing risk:", err);
    alert("Failed to compute risk. Check backend/ML service.");
    setRisks(null);
  }
}


  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', height: '100vh' }}>
      <div>
        {!token && <button onClick={registerAndLogin}>Start Demo Session</button>}
        <MapView locations={locations} onSelect={setSelectedId} />
      </div>
      <div style={{ padding: 16, overflow: 'auto' }}>
        <h2>EarthPulse</h2>
        <AddLocationForm onAdd={addLocation} />
        <hr/>
        <RiskPanel selected={selected} onCompute={computeRisk} risks={risks} />
      </div>
    </div>
  );
}
