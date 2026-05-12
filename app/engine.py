import geopandas as gpd

import uuid
from shapely.geometry import shape, Point, MultiPolygon
from app.schemas import GenerationRequest



import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from typing import Dict, Tuple

# Assuming schemas.py is in the same directory
from .schemas import GenerationRequest

def _edges_intersect(p1: np.ndarray, p2: np.ndarray, G: nx.Graph) -> bool:
    """
    Phase 2 Helper: Ensures planarity by preventing new edges from crossing existing ones.
    """
    new_line = LineString([p1, p2])
    for u, v in G.edges:
        existing_line = LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])
        # We only care about interior intersections, not sharing a node (touches)
        if new_line.crosses(existing_line):
            return True
    return False

def _attempt_cross_link(G: nx.Graph, new_node_id: int, request: GenerationRequest):
    """
    Phase 2: Topological Weaving. Attempts to form 3, 4, or 5 edge loops.
    """
    if np.random.rand() > request.cross_link_probability:
        return

    new_pos = G.nodes[new_node_id]['pos']
    
    # Find all potential neighbors within a certain search radius
    search_radius = request.arterial_length * 1.5
    candidates = []
    for node in G.nodes:
        if node == new_node_id: continue
        # Valency Check: Don't connect to overloaded intersections
        if G.degree[node] >= request.max_valency: continue
        
        dist = np.linalg.norm(G.nodes[node]['pos'] - new_pos)
        if dist < search_radius:
            candidates.append((dist, node))
            
    # Sort by closest
    candidates.sort()

    for _, target_node in candidates:
        # Prevent duplicate edges
        if G.has_edge(new_node_id, target_node): continue
        
        # 1. Cycle Check: The 3-5 Loop Rule
        try:
            # Shortest path between them right now (before adding the edge)
            path_length = nx.shortest_path_length(G, source=new_node_id, target=target_node)
            # If path is 2, adding edge makes a 3-loop (triangle block)
            # If path is 3, adding edge makes a 4-loop (square block)
            # If path is 4, adding edge makes a 5-loop (pentagon block)
            if path_length not in [2, 3, 4]:
                continue
        except nx.NetworkXNoPath:
            # Should not happen since graph is connected, but safety first
            continue

        # 2. Planarity Check: Don't cross existing streets
        if _edges_intersect(new_pos, G.nodes[target_node]['pos'], G):
            continue

        # 3. Add the Secondary Bond
        weight = request.arterial_length if (G.nodes[new_node_id]['type'] == 'major' and G.nodes[target_node]['type'] == 'major') else request.local_length
        G.add_edge(new_node_id, target_node, weight=weight)
        
        # Only form one cross-link per spawn to keep it organic
        break

def generate_base_graph(request: GenerationRequest) -> nx.Graph:
    """
    Phase 1 & 2: Main Kinematic Growth Loop.
    Generates the topological skeleton of the city.
    """
    G = nx.Graph()
    
    # Spawn Seed Node (The Ancient City Center)
    G.add_node(0, pos=np.array([0.0, 0.0]), type='major', age=0)
    
    for i in range(1, request.target_node_count):
        # Determine Node Type
        node_type = 'major' if np.random.rand() < request.major_node_ratio else 'minor'
        
        # Find valid parents (nodes with open valency)
        valid_parents = [n for n in G.nodes if G.degree[n] < request.max_valency]
        
        # If the city is totally maxed out (rare), stop growing
        if not valid_parents: break
            
        # Pick a parent (bias towards newer nodes to encourage outward sprawl)
        parent = valid_parents[-1] if np.random.rand() > 0.3 else np.random.choice(valid_parents)
        parent_pos = G.nodes[parent]['pos']
        
        # Calculate spawn distance based on Hooke's Law resting lengths
        spawn_dist = request.arterial_length if (node_type == 'major' and G.nodes[parent]['type'] == 'major') else request.local_length
        
        # Spawn at a random angle around the parent
        angle = np.random.uniform(0, 2 * np.pi)
        new_pos = parent_pos + np.array([np.cos(angle), np.sin(angle)]) * spawn_dist
        
        # Add the Node & Primary Bond
        G.add_node(i, pos=new_pos, type=node_type, age=0)
        G.add_edge(parent, i, weight=spawn_dist)
        
        # Attempt Phase 2: Topological Weaving (Secondary Bonds)
        _attempt_cross_link(G, i, request)
        
        # --- THE PHYSICS RELAXATION & HISTORICAL FREEZING ---
        # Update ages
        for n in G.nodes:
            G.nodes[n]['age'] += 1
            
        # Identify Frozen Nodes (Ancient historical core)
        frozen_nodes = [n for n in G.nodes if G.nodes[n]['age'] >= request.freezing_age_threshold]
        
        # If everyone is frozen (rare, but possible if growth is slow), unfreeze the newest to keep physics alive
        if len(frozen_nodes) == len(G.nodes) and len(G.nodes) > 1:
            frozen_nodes.remove(i)
            
        # Run Physics Simulation (Spring Layout)
        # NetworkX's Fruchterman-Reingold acts as our force-directed annealer
        current_positions = nx.get_node_attributes(G, 'pos')
        
        # Execute 3 iterations of physics per spawn tick
        new_positions = nx.spring_layout(
            G, 
            pos=current_positions, 
            fixed=frozen_nodes if frozen_nodes else None, 
            k=spawn_dist, # Optimal distance between nodes
            weight='weight',
            iterations=3,
            seed=42
        )
        
        # Update graph with relaxed positions
        nx.set_node_attributes(G, new_positions, 'pos')

    return G

def generate_district(request: GenerationRequest) -> gpd.GeoDataFrame:
    """
    Vectorized district generator using GeoPandas and NumPy.
    No classes used, strictly functional approach.
    """
    
    # 1. Parse site_polygon into Shapely geometry
    site_geom = shape(request.site_polygon)
    
    # 2. Use NumPy to generate random (x, y) coordinate arrays within bounding box
    minx, miny, maxx, maxy = site_geom.bounds
    # Oversample to ensure we get enough points after site boundary filtering
    sample_size = request.num_buildings * 5 
    x_arr = np.random.uniform(minx, maxx, sample_size)
    y_arr = np.random.uniform(miny, maxy, sample_size)
    
    # 3. Convert valid points (inside site) into a GeoSeries
    pts_gs = gpd.GeoSeries(gpd.points_from_xy(x_arr, y_arr))
    valid_pts = pts_gs[pts_gs.within(site_geom)].iloc[:request.num_buildings]
    
    if valid_pts.empty:
        return gpd.GeoDataFrame(columns=['geometry', 'b_ID', 'height'])

    # 4. Vectorized Growth: Expand points into initial buffers
    # Estimate radius based on target density: Area = density * total_area
    # Area_per_building = (density * total_area) / num_buildings
    # pi * r^2 = target_area => r = sqrt(target_area / pi)
    site_area = site_geom.area
    target_total_building_area = site_area * request.target_density
    radius = np.sqrt((target_total_building_area / len(valid_pts)) / np.pi)
    
    # Apply buffer (vectorized)
    buildings_gs = valid_pts.buffer(radius)
    
    # 5. Merge Overlaps: Dissolve overlapping buffers
    dissolved_union = buildings_gs.unary_union
    
    # 6. Explode: Break back down into individual polygons
    if isinstance(dissolved_union, MultiPolygon):
        exploded_gs = gpd.GeoSeries([dissolved_union]).explode(index_parts=False)
    else:
        exploded_gs = gpd.GeoSeries([dissolved_union])
        
    gdf = gpd.GeoDataFrame(geometry=exploded_gs)
    
    # 7. Boundaries: Ensure no building exceeds site_polygon
    gdf.geometry = gdf.geometry.intersection(site_geom)
    # Clean up empty or invalid results from intersection
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    # 8. Subtract Obstacles
    if request.obstacles and 'features' in request.obstacles:
        obstacles_gdf = gpd.GeoDataFrame.from_features(request.obstacles)
        if not obstacles_gdf.empty:
            # Set CRS to avoid warnings if applicable (using a dummy one here)
            gdf = gpd.overlay(gdf, obstacles_gdf, how='difference')
    
    # 9. Add Metadata
    gdf['b_ID'] = [str(uuid.uuid4()) for _ in range(len(gdf))]
    gdf['height'] = np.random.uniform(10.0, 50.0, len(gdf))
    
    return gdf