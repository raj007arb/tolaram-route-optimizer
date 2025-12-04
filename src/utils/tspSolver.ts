import { Location } from "@/data/locations";

// Haversine distance in kilometers
export function haversineDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371; // Earth's radius in km
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function toRad(deg: number): number {
  return deg * (Math.PI / 180);
}

// K-means clustering to divide locations among agents
export function kMeansClustering(locations: Location[], k: number, maxIterations = 50): Location[][] {
  if (locations.length === 0 || k === 0) return [];

  // Initialize centroids using k-means++ initialization
  const centroids: { lat: number; lng: number }[] = [];
  const usedIndices = new Set<number>();

  // First centroid: random
  const firstIdx = Math.floor(Math.random() * locations.length);
  centroids.push({ lat: locations[firstIdx].lat, lng: locations[firstIdx].lng });
  usedIndices.add(firstIdx);

  // Subsequent centroids: weighted by distance
  while (centroids.length < k) {
    const distances: number[] = locations.map((loc) => {
      const minDist = Math.min(...centroids.map((c) => haversineDistance(loc.lat, loc.lng, c.lat, c.lng)));
      return minDist * minDist;
    });
    const totalDist = distances.reduce((a, b) => a + b, 0);
    let random = Math.random() * totalDist;

    for (let i = 0; i < locations.length; i++) {
      random -= distances[i];
      if (random <= 0 && !usedIndices.has(i)) {
        centroids.push({ lat: locations[i].lat, lng: locations[i].lng });
        usedIndices.add(i);
        break;
      }
    }
  }

  let clusters: Location[][] = Array.from({ length: k }, () => []);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Assign points to nearest centroid
    clusters = Array.from({ length: k }, () => []);

    for (const loc of locations) {
      let minDist = Infinity;
      let closestCluster = 0;

      for (let i = 0; i < centroids.length; i++) {
        const dist = haversineDistance(loc.lat, loc.lng, centroids[i].lat, centroids[i].lng);
        if (dist < minDist) {
          minDist = dist;
          closestCluster = i;
        }
      }

      clusters[closestCluster].push(loc);
    }

    // Update centroids
    let converged = true;
    for (let i = 0; i < k; i++) {
      if (clusters[i].length === 0) continue;

      const newLat = clusters[i].reduce((sum, loc) => sum + loc.lat, 0) / clusters[i].length;
      const newLng = clusters[i].reduce((sum, loc) => sum + loc.lng, 0) / clusters[i].length;

      if (Math.abs(newLat - centroids[i].lat) > 0.0001 || Math.abs(newLng - centroids[i].lng) > 0.0001) {
        converged = false;
      }

      centroids[i] = { lat: newLat, lng: newLng };
    }

    if (converged) break;
  }

  // Balance clusters - redistribute from larger to smaller
  const avgSize = Math.ceil(locations.length / k);
  for (let i = 0; i < clusters.length; i++) {
    while (clusters[i].length > avgSize + 2) {
      // Find smallest cluster
      let smallestIdx = -1;
      let smallestSize = Infinity;
      for (let j = 0; j < clusters.length; j++) {
        if (j !== i && clusters[j].length < smallestSize) {
          smallestSize = clusters[j].length;
          smallestIdx = j;
        }
      }

      if (smallestIdx === -1 || smallestSize >= avgSize) break;

      // Move the point closest to the smaller cluster's centroid
      const targetCentroid = centroids[smallestIdx];
      let bestIdx = 0;
      let bestDist = Infinity;

      for (let p = 0; p < clusters[i].length; p++) {
        const dist = haversineDistance(
          clusters[i][p].lat,
          clusters[i][p].lng,
          targetCentroid.lat,
          targetCentroid.lng
        );
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = p;
        }
      }

      clusters[smallestIdx].push(clusters[i].splice(bestIdx, 1)[0]);
    }
  }

  return clusters.filter((c) => c.length > 0);
}

// Nearest neighbor heuristic for TSP
export function nearestNeighborTSP(locations: Location[]): Location[] {
  if (locations.length <= 1) return locations;

  const route: Location[] = [];
  const unvisited = new Set(locations.map((l) => l.id));

  // Start from the location closest to the centroid
  const centroidLat = locations.reduce((sum, l) => sum + l.lat, 0) / locations.length;
  const centroidLng = locations.reduce((sum, l) => sum + l.lng, 0) / locations.length;

  let current = locations.reduce((closest, loc) => {
    const closestDist = haversineDistance(closest.lat, closest.lng, centroidLat, centroidLng);
    const locDist = haversineDistance(loc.lat, loc.lng, centroidLat, centroidLng);
    return locDist < closestDist ? loc : closest;
  });

  route.push(current);
  unvisited.delete(current.id);

  while (unvisited.size > 0) {
    let nearest: Location | null = null;
    let minDist = Infinity;

    for (const loc of locations) {
      if (!unvisited.has(loc.id)) continue;

      const dist = haversineDistance(current.lat, current.lng, loc.lat, loc.lng);
      if (dist < minDist) {
        minDist = dist;
        nearest = loc;
      }
    }

    if (nearest) {
      route.push(nearest);
      unvisited.delete(nearest.id);
      current = nearest;
    }
  }

  return route;
}

// 2-opt improvement
export function twoOptImprove(route: Location[], maxIterations = 100): Location[] {
  if (route.length <= 3) return route;

  let improved = true;
  let iterations = 0;
  let bestRoute = [...route];

  while (improved && iterations < maxIterations) {
    improved = false;
    iterations++;

    for (let i = 0; i < bestRoute.length - 2; i++) {
      for (let j = i + 2; j < bestRoute.length; j++) {
        const d1 =
          haversineDistance(bestRoute[i].lat, bestRoute[i].lng, bestRoute[i + 1].lat, bestRoute[i + 1].lng) +
          haversineDistance(
            bestRoute[j].lat,
            bestRoute[j].lng,
            bestRoute[(j + 1) % bestRoute.length].lat,
            bestRoute[(j + 1) % bestRoute.length].lng
          );

        const d2 =
          haversineDistance(bestRoute[i].lat, bestRoute[i].lng, bestRoute[j].lat, bestRoute[j].lng) +
          haversineDistance(
            bestRoute[i + 1].lat,
            bestRoute[i + 1].lng,
            bestRoute[(j + 1) % bestRoute.length].lat,
            bestRoute[(j + 1) % bestRoute.length].lng
          );

        if (d2 < d1 - 0.0001) {
          // Reverse the segment between i+1 and j
          const newRoute = [...bestRoute.slice(0, i + 1), ...bestRoute.slice(i + 1, j + 1).reverse(), ...bestRoute.slice(j + 1)];
          bestRoute = newRoute;
          improved = true;
        }
      }
    }
  }

  return bestRoute;
}

// Calculate total route distance
export function calculateRouteDistance(route: Location[]): number {
  if (route.length <= 1) return 0;

  let total = 0;
  for (let i = 0; i < route.length - 1; i++) {
    total += haversineDistance(route[i].lat, route[i].lng, route[i + 1].lat, route[i + 1].lng);
  }
  // Add return to start
  total += haversineDistance(route[route.length - 1].lat, route[route.length - 1].lng, route[0].lat, route[0].lng);
  return total;
}

export interface AgentRoute {
  agentId: number;
  agentName: string;
  route: Location[];
  distance: number;
  color: string;
}

// Generate distinct colors for agents
export function generateAgentColors(count: number): string[] {
  const colors: string[] = [];
  for (let i = 0; i < count; i++) {
    const hue = (i * 360) / count;
    colors.push(`hsl(${hue}, 75%, 50%)`);
  }
  return colors;
}

// Main solver function
export function solveVRPTSP(
  locations: Location[],
  agents: { id: number; name: string }[]
): AgentRoute[] {
  const numAgents = agents.length;
  const colors = generateAgentColors(numAgents);

  // Cluster locations
  const clusters = kMeansClustering(locations, numAgents);

  // Solve TSP for each cluster
  const agentRoutes: AgentRoute[] = [];

  for (let i = 0; i < clusters.length; i++) {
    const agent = agents[i] || agents[agents.length - 1];
    let route = nearestNeighborTSP(clusters[i]);
    route = twoOptImprove(route);

    agentRoutes.push({
      agentId: agent.id,
      agentName: agent.name,
      route,
      distance: calculateRouteDistance(route),
      color: colors[i],
    });
  }

  // Sort by agent ID
  return agentRoutes.sort((a, b) => a.agentId - b.agentId);
}

// Generate great circle arc points between two coordinates
export function generateArcPoints(
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number,
  numPoints = 20
): [number, number][] {
  const points: [number, number][] = [];

  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const lambda1 = toRad(lng1);
  const lambda2 = toRad(lng2);

  const d = 2 * Math.asin(
    Math.sqrt(
      Math.sin((phi2 - phi1) / 2) ** 2 +
        Math.cos(phi1) * Math.cos(phi2) * Math.sin((lambda2 - lambda1) / 2) ** 2
    )
  );

  if (d === 0) {
    return [[lat1, lng1], [lat2, lng2]];
  }

  for (let i = 0; i <= numPoints; i++) {
    const f = i / numPoints;
    const A = Math.sin((1 - f) * d) / Math.sin(d);
    const B = Math.sin(f * d) / Math.sin(d);

    const x = A * Math.cos(phi1) * Math.cos(lambda1) + B * Math.cos(phi2) * Math.cos(lambda2);
    const y = A * Math.cos(phi1) * Math.sin(lambda1) + B * Math.cos(phi2) * Math.sin(lambda2);
    const z = A * Math.sin(phi1) + B * Math.sin(phi2);

    const lat = Math.atan2(z, Math.sqrt(x * x + y * y)) * (180 / Math.PI);
    const lng = Math.atan2(y, x) * (180 / Math.PI);

    points.push([lat, lng]);
  }

  return points;
}
