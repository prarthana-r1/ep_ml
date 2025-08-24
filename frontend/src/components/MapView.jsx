import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet';

export default function MapView({ locations, selectedId, onSelect }) {
  return (
    <MapContainer center={[20.5, 78.9]} zoom={4.5} style={{ height: '100%', width: '100%' }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {locations.map(loc => (
        loc.id === selectedId ? (
          <CircleMarker
            key={loc.id}
            center={[loc.lat, loc.lon]}
            radius={15}
            color="red"
            eventHandlers={{ click: () => onSelect(loc.id) }}
          >
          </CircleMarker>
        ) : (
          <Marker
            key={loc.id}
            position={[loc.lat, loc.lon]}
            eventHandlers={{ click: () => onSelect(loc.id) }}
          />
        )
      ))}
    </MapContainer>
  );
}
