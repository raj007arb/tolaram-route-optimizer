"""
TSP Route Optimizer for Tolaram Assignment
Solves Vehicle Routing Problem for 41 agents covering ~400 locations in Lagos, Nigeria

Requirements:
    pip install pandas openpyxl numpy matplotlib folium scikit-learn

Usage:
    python tsp_solver.py

Output:
    - route_assignments.csv: CSV file with all route assignments
    - routes_map.html: Interactive map with all routes
    - routes_plot.png: Static plot of all routes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2, pi
from sklearn.cluster import KMeans
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

LOCATIONS_FILE = "TSP1.xlsm"  # or TSP2.xlsx - update path as needed
AGENTS_FILE = "TSP2.xlsx"      # File containing agents list
NUM_AGENTS = 41

# =============================================================================
# DISTANCE CALCULATION (Haversine Formula)
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def calculate_distance_matrix(locations):
    """Calculate pairwise distance matrix for all locations."""
    n = len(locations)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(
                locations[i]['lat'], locations[i]['lng'],
                locations[j]['lat'], locations[j]['lng']
            )
            matrix[i][j] = dist
            matrix[j][i] = dist
    
    return matrix

# =============================================================================
# CLUSTERING (K-Means)
# =============================================================================

def cluster_locations(locations, num_clusters):
    """
    Cluster locations using K-Means algorithm.
    Returns cluster assignments for each location.
    """
    coords = np.array([[loc['lat'], loc['lng']] for loc in locations])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)
    
    return clusters

# =============================================================================
# TSP SOLVERS
# =============================================================================

def nearest_neighbor_tsp(locations, distance_matrix):
    """
    Solve TSP using Nearest Neighbor heuristic.
    Returns ordered list of location indices.
    """
    if len(locations) == 0:
        return []
    if len(locations) == 1:
        return [0]
    
    n = len(locations)
    visited = [False] * n
    route = [0]  # Start from first location
    visited[0] = True
    
    for _ in range(n - 1):
        current = route[-1]
        nearest = -1
        nearest_dist = float('inf')
        
        for j in range(n):
            if not visited[j] and distance_matrix[current][j] < nearest_dist:
                nearest = j
                nearest_dist = distance_matrix[current][j]
        
        if nearest != -1:
            route.append(nearest)
            visited[nearest] = True
    
    return route

def two_opt_improve(route, distance_matrix, max_iterations=1000):
    """
    Improve route using 2-opt optimization.
    Swaps edges to reduce total distance.
    """
    if len(route) < 4:
        return route
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(route) - 2):
            for j in range(i + 2, len(route)):
                # Calculate current distance
                d1 = distance_matrix[route[i]][route[i+1]]
                d2 = distance_matrix[route[j]][route[(j+1) % len(route)]]
                
                # Calculate new distance after swap
                d3 = distance_matrix[route[i]][route[j]]
                d4 = distance_matrix[route[i+1]][route[(j+1) % len(route)]]
                
                # If swap improves distance, apply it
                if d3 + d4 < d1 + d2:
                    route[i+1:j+1] = reversed(route[i+1:j+1])
                    improved = True
    
    return route

def calculate_route_distance(route, distance_matrix):
    """Calculate total distance of a route."""
    if len(route) < 2:
        return 0
    
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i+1]]
    
    # Add return to start (circular route)
    total += distance_matrix[route[-1]][route[0]]
    
    return total

# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_vrp_tsp(locations, num_agents):
    """
    Solve Vehicle Routing Problem using:
    1. K-Means clustering to assign locations to agents
    2. Nearest Neighbor heuristic for initial route
    3. 2-opt optimization to improve routes
    
    Returns list of agent routes with statistics.
    """
    print(f"\n{'='*60}")
    print(f"SOLVING TSP FOR {len(locations)} LOCATIONS WITH {num_agents} AGENTS")
    print(f"{'='*60}\n")
    
    # Step 1: Cluster locations
    print("Step 1: Clustering locations using K-Means...")
    clusters = cluster_locations(locations, num_agents)
    
    # Group locations by cluster
    clustered_locations = [[] for _ in range(num_agents)]
    for idx, cluster_id in enumerate(clusters):
        clustered_locations[cluster_id].append({
            'index': idx,
            'location': locations[idx]
        })
    
    print(f"   - Created {num_agents} clusters")
    print(f"   - Locations per cluster: min={min(len(c) for c in clustered_locations)}, "
          f"max={max(len(c) for c in clustered_locations)}, "
          f"avg={sum(len(c) for c in clustered_locations)/num_agents:.1f}")
    
    # Step 2: Solve TSP for each cluster
    print("\nStep 2: Solving TSP for each agent's cluster...")
    agent_routes = []
    
    for agent_id in range(num_agents):
        cluster_locs = clustered_locations[agent_id]
        
        if len(cluster_locs) == 0:
            agent_routes.append({
                'agent_id': agent_id + 1,
                'locations': [],
                'route_order': [],
                'distance': 0
            })
            continue
        
        # Extract locations for this cluster
        locs = [item['location'] for item in cluster_locs]
        original_indices = [item['index'] for item in cluster_locs]
        
        # Calculate distance matrix for this cluster
        dist_matrix = calculate_distance_matrix(locs)
        
        # Solve TSP using Nearest Neighbor
        route = nearest_neighbor_tsp(locs, dist_matrix)
        
        # Improve with 2-opt
        route = two_opt_improve(route, dist_matrix)
        
        # Calculate final distance
        distance = calculate_route_distance(route, dist_matrix)
        
        # Map back to original indices and locations
        ordered_locations = [locs[i] for i in route]
        ordered_indices = [original_indices[i] for i in route]
        
        agent_routes.append({
            'agent_id': agent_id + 1,
            'locations': ordered_locations,
            'route_order': ordered_indices,
            'distance': distance
        })
        
        print(f"   Agent {agent_id + 1:2d}: {len(locs):3d} locations, {distance:8.2f} km")
    
    # Step 3: Calculate statistics
    distances = [r['distance'] for r in agent_routes if r['distance'] > 0]
    total_distance = sum(distances)
    avg_distance = total_distance / len(distances) if distances else 0
    std_distance = np.std(distances) if distances else 0
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Total locations covered: {sum(len(r['locations']) for r in agent_routes)}")
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Average distance per agent: {avg_distance:.2f} km")
    print(f"Standard deviation: {std_distance:.2f} km")
    print(f"Route balance (CV): {(std_distance/avg_distance)*100:.1f}%" if avg_distance > 0 else "N/A")
    print(f"Min route: {min(distances):.2f} km" if distances else "N/A")
    print(f"Max route: {max(distances):.2f} km" if distances else "N/A")
    
    return agent_routes

# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_colors(n):
    """Generate n distinct colors for routes."""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSL to RGB (simplified)
        if hue < 1/6:
            r, g, b = 1, hue*6, 0
        elif hue < 2/6:
            r, g, b = 1-(hue-1/6)*6, 1, 0
        elif hue < 3/6:
            r, g, b = 0, 1, (hue-2/6)*6
        elif hue < 4/6:
            r, g, b = 0, 1-(hue-3/6)*6, 1
        elif hue < 5/6:
            r, g, b = (hue-4/6)*6, 0, 1
        else:
            r, g, b = 1, 0, 1-(hue-5/6)*6
        
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    
    return colors

def generate_arc_points(lat1, lon1, lat2, lon2, num_points=20):
    """
    Generate points along a great circle arc between two coordinates.
    This creates the curved path effect on the map.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    d = 2 * atan2(
        sqrt(sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2),
        sqrt(1 - sin((lat2-lat1)/2)**2 - cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2)
    )
    
    if d == 0:
        return [[lat1 * 180/pi, lon1 * 180/pi]]
    
    points = []
    for i in range(num_points + 1):
        f = i / num_points
        A = sin((1-f)*d) / sin(d)
        B = sin(f*d) / sin(d)
        
        x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
        y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
        z = A * sin(lat1) + B * sin(lat2)
        
        lat = atan2(z, sqrt(x**2 + y**2))
        lon = atan2(y, x)
        
        points.append([lat * 180/pi, lon * 180/pi])
    
    return points

def create_folium_map(agent_routes, output_file='routes_map.html'):
    """Create interactive Folium map with all routes."""
    print(f"\nCreating interactive map: {output_file}")
    
    # Calculate map center
    all_locs = []
    for route in agent_routes:
        all_locs.extend(route['locations'])
    
    if not all_locs:
        print("   No locations to display!")
        return
    
    center_lat = sum(loc['lat'] for loc in all_locs) / len(all_locs)
    center_lng = sum(loc['lng'] for loc in all_locs) / len(all_locs)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=11,
        tiles='cartodbdark_matter'
    )
    
    colors = generate_colors(len(agent_routes))
    
    # Add routes
    for route, color in zip(agent_routes, colors):
        if len(route['locations']) < 2:
            continue
        
        locs = route['locations']
        
        # Create curved path
        all_points = []
        for i in range(len(locs)):
            next_i = (i + 1) % len(locs)
            arc_points = generate_arc_points(
                locs[i]['lat'], locs[i]['lng'],
                locs[next_i]['lat'], locs[next_i]['lng'],
                num_points=10
            )
            all_points.extend(arc_points)
        
        # Add polyline
        folium.PolyLine(
            all_points,
            color=color,
            weight=2,
            opacity=0.8,
            popup=f"Agent {route['agent_id']}: {route['distance']:.2f} km"
        ).add_to(m)
        
        # Add markers for each location
        for idx, loc in enumerate(locs):
            folium.CircleMarker(
                location=[loc['lat'], loc['lng']],
                radius=4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"Agent {route['agent_id']}, Stop {idx+1}: {loc.get('name', 'Location')}"
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;
                color: white; font-family: Arial; font-size: 12px;">
        <b>TSP Route Optimizer</b><br>
        41 Agents | ''' + str(len(all_locs)) + ''' Locations<br>
        Total Distance: ''' + f"{sum(r['distance'] for r in agent_routes):.2f}" + ''' km
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_file)
    print(f"   Map saved to {output_file}")

def create_matplotlib_plot(agent_routes, output_file='routes_plot.png'):
    """Create static matplotlib plot of all routes."""
    print(f"\nCreating static plot: {output_file}")
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    colors = generate_colors(len(agent_routes))
    
    for route, color in zip(agent_routes, colors):
        if len(route['locations']) < 2:
            continue
        
        locs = route['locations']
        
        # Plot curved paths
        for i in range(len(locs)):
            next_i = (i + 1) % len(locs)
            arc_points = generate_arc_points(
                locs[i]['lat'], locs[i]['lng'],
                locs[next_i]['lat'], locs[next_i]['lng'],
                num_points=20
            )
            lats = [p[0] for p in arc_points]
            lngs = [p[1] for p in arc_points]
            ax.plot(lngs, lats, color=color, linewidth=1, alpha=0.7)
        
        # Plot points
        point_lats = [loc['lat'] for loc in locs]
        point_lngs = [loc['lng'] for loc in locs]
        ax.scatter(point_lngs, point_lats, c=color, s=15, alpha=0.8, zorder=5)
    
    ax.set_xlabel('Longitude', color='white')
    ax.set_ylabel('Latitude', color='white')
    ax.set_title('TSP Route Optimization - 41 Agents, Lagos Nigeria', 
                 color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Add stats text
    total_dist = sum(r['distance'] for r in agent_routes)
    total_locs = sum(len(r['locations']) for r in agent_routes)
    stats_text = f"Total: {total_locs} locations | {total_dist:.2f} km"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            color='white', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"   Plot saved to {output_file}")

def export_to_csv(agent_routes, output_file='route_assignments.csv'):
    """Export route assignments to CSV file."""
    print(f"\nExporting to CSV: {output_file}")
    
    rows = []
    for route in agent_routes:
        for stop_num, loc in enumerate(route['locations'], 1):
            rows.append({
                'Agent_ID': route['agent_id'],
                'Stop_Number': stop_num,
                'Location_Name': loc.get('name', f"Location_{stop_num}"),
                'Latitude': loc['lat'],
                'Longitude': loc['lng'],
                'Route_Total_Distance_km': round(route['distance'], 2)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"   Exported {len(rows)} records to {output_file}")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_locations_from_excel(filepath):
    """Load locations from Excel file."""
    print(f"Loading locations from: {filepath}")
    
    try:
        # Try reading the file
        df = pd.read_excel(filepath)
        print(f"   Columns found: {list(df.columns)}")
        
        # Try to identify lat/lng columns
        lat_col = None
        lng_col = None
        name_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'lat' in col_lower:
                lat_col = col
            elif 'lon' in col_lower or 'lng' in col_lower:
                lng_col = col
            elif 'name' in col_lower or 'location' in col_lower or 'outlet' in col_lower:
                name_col = col
        
        if lat_col is None or lng_col is None:
            # Try numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                # Assume first two numeric columns are lat/lng
                lat_col = numeric_cols[0]
                lng_col = numeric_cols[1]
                print(f"   Using numeric columns: {lat_col}, {lng_col}")
        
        if lat_col is None or lng_col is None:
            raise ValueError("Could not identify latitude/longitude columns")
        
        print(f"   Lat column: {lat_col}, Lng column: {lng_col}")
        
        locations = []
        for idx, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lng = float(row[lng_col])
                
                # Validate coordinates (Lagos area)
                if 3 < lat < 8 and 2 < lng < 5:
                    loc = {'lat': lat, 'lng': lng}
                    if name_col and pd.notna(row[name_col]):
                        loc['name'] = str(row[name_col])
                    else:
                        loc['name'] = f"Location_{idx+1}"
                    locations.append(loc)
            except (ValueError, TypeError):
                continue
        
        print(f"   Loaded {len(locations)} valid locations")
        return locations
        
    except Exception as e:
        print(f"   Error loading file: {e}")
        return None

# =============================================================================
# SAMPLE DATA (Fallback)
# =============================================================================

def get_sample_locations():
    """Return sample Lagos locations if Excel loading fails."""
    # This is sample data - replace with your actual Excel data
    return [
        {"lat": 6.5244, "lng": 3.3792, "name": "Lagos Island"},
        {"lat": 6.4541, "lng": 3.3947, "name": "Victoria Island"},
        {"lat": 6.4698, "lng": 3.5852, "name": "Lekki"},
        {"lat": 6.5959, "lng": 3.3491, "name": "Ikeja"},
        {"lat": 6.6018, "lng": 3.3515, "name": "Maryland"},
        {"lat": 6.5833, "lng": 3.3500, "name": "Yaba"},
        {"lat": 6.4483, "lng": 3.4725, "name": "Ikoyi"},
        {"lat": 6.6194, "lng": 3.5105, "name": "Gbagada"},
        {"lat": 6.5355, "lng": 3.3087, "name": "Surulere"},
        {"lat": 6.4400, "lng": 3.4200, "name": "Oniru"},
        # Add more sample locations as needed
    ]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "="*60)
    print("TSP ROUTE OPTIMIZER - TOLARAM ASSIGNMENT")
    print("="*60)
    
    # Try to load locations from Excel
    locations = load_locations_from_excel(LOCATIONS_FILE)
    
    if locations is None or len(locations) < 10:
        print("\nFalling back to sample data...")
        print("Please update LOCATIONS_FILE path at the top of this script")
        locations = get_sample_locations()
    
    # Solve the VRP-TSP problem
    agent_routes = solve_vrp_tsp(locations, NUM_AGENTS)
    
    # Generate outputs
    export_to_csv(agent_routes)
    create_matplotlib_plot(agent_routes)
    create_folium_map(agent_routes)
    
    print("\n" + "="*60)
    print("COMPLETE! Generated files:")
    print("  - route_assignments.csv  (Route data)")
    print("  - routes_plot.png        (Static visualization)")
    print("  - routes_map.html        (Interactive map)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
