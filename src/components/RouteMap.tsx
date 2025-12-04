import { useEffect, useRef, useState, useMemo } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { AgentRoute, generateArcPoints } from "@/utils/tspSolver";

interface RouteMapProps {
  routes: AgentRoute[];
  selectedAgent: number | null;
  onSelectAgent: (agentId: number | null) => void;
}

const RouteMap = ({ routes, selectedAgent, onSelectAgent }: RouteMapProps) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layersRef = useRef<L.LayerGroup[]>([]);

  // Calculate map center
  const center = useMemo(() => {
    const allPoints = routes.flatMap((r) => r.route);
    if (allPoints.length === 0) return { lat: 6.5, lng: 3.35 };

    const lat = allPoints.reduce((sum, p) => sum + p.lat, 0) / allPoints.length;
    const lng = allPoints.reduce((sum, p) => sum + p.lng, 0) / allPoints.length;
    return { lat, lng };
  }, [routes]);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    // Initialize map
    const map = L.map(mapRef.current, {
      center: [center.lat, center.lng],
      zoom: 12,
      zoomControl: true,
    });

    // Dark tile layer
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [center]);

  // Update routes when data changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Clear existing layers
    layersRef.current.forEach((layer) => {
      map.removeLayer(layer);
    });
    layersRef.current = [];

    // Add routes
    routes.forEach((route) => {
      const layerGroup = L.layerGroup();
      const isSelected = selectedAgent === null || selectedAgent === route.agentId;
      const opacity = isSelected ? 1 : 0.15;

      // Draw route lines with arc interpolation
      if (route.route.length > 1) {
        const allPoints: [number, number][] = [];

        for (let i = 0; i < route.route.length; i++) {
          const current = route.route[i];
          const next = route.route[(i + 1) % route.route.length];
          const arcPoints = generateArcPoints(current.lat, current.lng, next.lat, next.lng, 10);
          allPoints.push(...arcPoints);
        }

        const polyline = L.polyline(allPoints, {
          color: route.color,
          weight: isSelected ? 3 : 2,
          opacity,
          smoothFactor: 1,
        });

        polyline.on("click", () => {
          onSelectAgent(selectedAgent === route.agentId ? null : route.agentId);
        });

        layerGroup.addLayer(polyline);
      }

      // Add markers for locations
      route.route.forEach((loc, idx) => {
        const isStart = idx === 0;
        const markerHtml = `
          <div style="
            width: ${isStart ? "16px" : "10px"};
            height: ${isStart ? "16px" : "10px"};
            background: ${route.color};
            border: 2px solid white;
            border-radius: 50%;
            opacity: ${opacity};
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
          "></div>
        `;

        const icon = L.divIcon({
          html: markerHtml,
          className: "custom-marker",
          iconSize: [isStart ? 16 : 10, isStart ? 16 : 10],
          iconAnchor: [isStart ? 8 : 5, isStart ? 8 : 5],
        });

        const marker = L.marker([loc.lat, loc.lng], { icon });

        marker.bindPopup(`
          <div class="p-1">
            <div class="font-semibold text-sm">${loc.name}</div>
            <div class="text-xs text-muted-foreground mt-1">${loc.phone}</div>
            <div class="text-xs mt-2" style="color: ${route.color}">
              ${route.agentName} • Stop #${idx + 1}
            </div>
          </div>
        `);

        layerGroup.addLayer(marker);
      });

      layerGroup.addTo(map);
      layersRef.current.push(layerGroup);
    });
  }, [routes, selectedAgent, onSelectAgent]);

  return (
    <div className="relative w-full h-full">
      <div ref={mapRef} className="w-full h-full rounded-xl" />
      {selectedAgent !== null && (
        <button
          onClick={() => onSelectAgent(null)}
          className="absolute top-4 right-4 z-[1000] px-3 py-1.5 bg-card/90 backdrop-blur-sm border border-border rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
        >
          Show All Routes
        </button>
      )}
    </div>
  );
};

export default RouteMap;
