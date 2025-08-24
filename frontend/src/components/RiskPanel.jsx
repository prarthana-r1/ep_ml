export default function RiskPanel({ selected, onCompute, risks }) {
  if (!selected) return <p>Select a location to see risks</p>;

  return (
    <div>
      <h3>{selected.name}</h3>
      <button onClick={onCompute}>Compute Risk</button>

      {risks ? (
        <div style={{ marginTop: 8 }}>
          <p>🌊 Flood Risk: <strong>{risks.flood_risk}</strong></p>
          <p>🔥 Wildfire Risk: <strong>{risks.wildfire_risk}</strong></p>
        </div>
      ) : (
        <p style={{ marginTop: 8 }}>Click "Compute Risk" to see risk scores.</p>
      )}
    </div>
  );
}
