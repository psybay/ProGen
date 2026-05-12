import uuid
from typing import Dict, List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import polygonize, unary_union

from .schemas import GenerationRequest, GenerationResponse, GeoJSONFeature, GeoJSONGeometry, GeoJSONFeatureCollection

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
        is_major = (G.nodes[new_node_id]['type'] == 'major' and G.nodes[target_node]['type'] == 'major')
        weight = request.arterial_length if is_major else request.local_length
        
        G.add_edge(new_node_id, target_node, weight=weight, is_arterial=is_major)
        
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
        is_major = (G.nodes[parent]['type'] == 'major' and node_type == 'major')
        G.add_edge(parent, i, weight=spawn_dist, is_arterial=is_major)
        
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


def planarize_graph(G: nx.Graph) -> nx.Graph:
    """
    Phase 3: Mesh Healing (Planarization).
    Breaks intersecting edges while inheriting the 'age' and 'is_arterial' states.
    """
    lines = []
    # Store a mapping of LineStrings back to their original edge data to preserve is_arterial
    line_to_data = {}
    
    for u, v, data in G.edges(data=True):
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        line = LineString([p1, p2])
        lines.append(line)
        line_to_data[line.wkt] = data.get('is_arterial', False)
        
    merged = unary_union(lines)
    
    segments = []
    if merged.geom_type == 'MultiLineString':
        segments = list(merged.geoms)
    elif merged.geom_type == 'LineString':
        segments = [merged]
    elif merged.geom_type == 'GeometryCollection':
        segments = [geom for geom in merged.geoms if geom.geom_type == 'LineString']
        
    G_planar = nx.Graph()
    coord_to_id = {}
    next_id = 0
    
    # Pre-map the original nodes so we can inherit their age/type
    original_coords = { (round(data['pos'][0], 4), round(data['pos'][1], 4)): data for n, data in G.nodes(data=True) }
    
    def get_or_create_node(coord):
        nonlocal next_id
        coord_tup = (round(coord[0], 4), round(coord[1], 4))
        
        if coord_tup not in coord_to_id:
            coord_to_id[coord_tup] = next_id
            
            # INHERITANCE: If this coordinate matches an original node, keep its age/type!
            if coord_tup in original_coords:
                orig_data = original_coords[coord_tup]
                G_planar.add_node(next_id, pos=np.array(coord), type=orig_data['type'], age=orig_data['age'])
            else:
                # It's a newly sliced intersection node. It is young and minor.
                G_planar.add_node(next_id, pos=np.array(coord), type='minor', age=0)
            next_id += 1
            
        return coord_to_id[coord_tup]

    for line in segments:
        coords = list(line.coords)
        
        # INHERITANCE: Figure out if this broken segment was originally an arterial
        # (A simple midpoint check against the original lines)
        midpoint = line.interpolate(0.5, normalized=True)
        is_art = False
        for orig_line_wkt, was_arterial in line_to_data.items():
            import shapely.wkt
            orig_line = shapely.wkt.loads(orig_line_wkt)
            if orig_line.distance(midpoint) < 1e-6:
                is_art = was_arterial
                break

        for i in range(len(coords) - 1):
            u_id = get_or_create_node(coords[i])
            v_id = get_or_create_node(coords[i+1])
            
            dist = np.linalg.norm(G_planar.nodes[u_id]['pos'] - G_planar.nodes[v_id]['pos'])
            G_planar.add_edge(u_id, v_id, weight=dist, is_arterial=is_art)
            
    return G_planar


def resolve_voids_and_zone(G: nx.Graph, request: GenerationRequest) -> Tuple[nx.Graph, List[Polygon]]:
    """
    Phase 4: Zoning & Void Resolution.
    Finds >5 edge blocks and converts them to parks or subdivides via Delaunay.
    """
    # 1. Convert edges back to lines to find faces
    lines = [LineString([G.nodes[u]['pos'], G.nodes[v]['pos']]) for u, v in G.edges]
        
    # 2. Polygonize to find all enclosed blocks
    faces = list(polygonize(lines))
    
    parks = []
    
    # Helper to map coordinates back to graph nodes for Delaunay edge insertion
    coord_to_id = { (round(data['pos'][0], 4), round(data['pos'][1], 4)): n for n, data in G.nodes(data=True) }
    
    for face in faces:
        coords = list(face.exterior.coords)
        # A 3-edge triangle has 4 coordinates (the start and end overlap).
        # Therefore, >6 coordinates means the face has >5 edges (A Void!)
        if len(coords) > 6:
            make_park = False
            if request.void_resolution_strategy == 'parks':
                make_park = True
            elif request.void_resolution_strategy == 'mixed':
                make_park = np.random.rand() < request.park_ratio
                
            if make_park:
                # Save as a Green Space
                parks.append(face)
            else:
                # Delaunay Subdivision
                unique_coords = coords[:-1] # Remove overlapping end coordinate
                pts = np.array(unique_coords)
                
                # Safety check: Delaunay needs at least 3 points
                if len(pts) >= 3:
                    tri = Delaunay(pts)
                    for simplex in tri.simplices:
                        # simplex contains 3 indices pointing to pts array
                        for i in range(3):
                            p1 = pts[simplex[i]]
                            p2 = pts[simplex[(i+1)%3]]
                            
                            id1 = coord_to_id[(round(p1[0], 4), round(p1[1], 4))]
                            id2 = coord_to_id[(round(p2[0], 4), round(p2[1], 4))]
                            
                            # Add the new internal street connecting the perimeter nodes
                            if not G.has_edge(id1, id2):
                                dist = np.linalg.norm(p1 - p2)
                                G.add_edge(id1, id2, weight=dist)
                                
    return G, parks


def finalize_city(G: nx.Graph, parks: List[Polygon], request: GenerationRequest) -> GenerationResponse:
    """
    Phase 5: Final Annealing & Translation.
    Relaxes the final graph, buffers the streets, cuts the blocks, and formats the API response.
    """
    # ---------------------------------------------------------
    # 1. Final Physics Annealing
    # ---------------------------------------------------------
    # Identify which nodes were historically frozen
    frozen_nodes = [n for n in G.nodes if G.nodes[n]['age'] >= request.freezing_age_threshold]
    
    # Run a quick relaxation to smooth out the new Delaunay void-streets
    new_positions = nx.spring_layout(
        G, 
        pos=nx.get_node_attributes(G, 'pos'), 
        fixed=frozen_nodes if frozen_nodes else None,
        k=request.local_length, 
        weight='weight',
        iterations=5, # Just enough to smooth the jagged edges
        seed=42
    )
    nx.set_node_attributes(G, new_positions, 'pos')

    # ---------------------------------------------------------
    # 2. Street Buffering (Geometry Translation)
    # ---------------------------------------------------------
    street_polygons = []
    for u, v, data in G.edges(data=True):
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        line = LineString([p1, p2])
        
        # Determine physical street width
        is_arterial = data.get('is_arterial', False)
        width = request.arterial_width if is_arterial else request.local_width
        
        # Buffer the line (cap_style=2 is flat, join_style=2 is mitre)
        street_polygons.append(line.buffer(width / 2.0, cap_style=2, join_style=2))
        
    # Combine all individual street buffers into one giant asphalt polygon
    master_street_network = unary_union(street_polygons)

    # ---------------------------------------------------------
    # 3. Block Generation (Boolean Subtraction)
    # ---------------------------------------------------------
    # Parse the input GeoJSON site boundary
    # FIX: Convert the Pydantic model to a raw dictionary so Shapely can read it
    geom_dict = request.site_polygon.geometry.model_dump() 
    # Note: if you are using an older version of Pydantic (v1), use .dict() instead of .model_dump()
    
    site_boundary = shape(geom_dict)
    
    # Subtract the streets from the site to get the raw urban blocks
    raw_blocks = site_boundary.difference(master_street_network)
    
    # Explode MultiPolygons into individual block Polygons
    if raw_blocks.geom_type == 'MultiPolygon':
        individual_blocks = list(raw_blocks.geoms)
    elif raw_blocks.geom_type == 'Polygon':
        individual_blocks = [raw_blocks]
    else:
        individual_blocks = []

    # ---------------------------------------------------------
    # 4. Filter Parks & Format API Response
    # ---------------------------------------------------------
    from shapely.geometry import mapping  # Import the native GeoJSON mapper
    
    block_features = []
    park_union = unary_union(parks) if parks else Polygon()
    
    for i, block in enumerate(individual_blocks):
        # Skip tiny slivers created by mathematical rounding
        if block.area < 10.0: continue
            
        is_park = False
        if not park_union.is_empty and block.intersection(park_union).area > (block.area * 0.5):
            is_park = True
            
        # FIX: Use native mapping to guarantee perfect GeoJSON translation
        geom_mapped = mapping(block)
        
        block_features.append(GeoJSONFeature(
            geometry=GeoJSONGeometry(type=geom_mapped["type"], coordinates=geom_mapped["coordinates"]),
            properties={
                "block_id": f"blk_{i}",
                "landuse": "park" if is_park else "development",
                "area": round(block.area, 2)
            }
        ))

    # Format the Street Network for export
    street_features = []
    
    # FIX: Map the entire master network at once. 
    # This automatically handles Polygons, MultiPolygons, and all nested holes flawlessly.
    street_mapped = mapping(master_street_network)
    street_features.append(GeoJSONFeature(
        geometry=GeoJSONGeometry(type=street_mapped["type"], coordinates=street_mapped["coordinates"]),
        properties={"type": "asphalt"}
    ))

    return GenerationResponse(
        blocks=GeoJSONFeatureCollection(features=block_features),
        streets=GeoJSONFeatureCollection(features=street_features),
        metadata={
            "total_nodes": len(G.nodes),
            "total_blocks": len(block_features),
            "parks_generated": len(parks)
        }
    )