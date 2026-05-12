import pytest
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString, MultiPolygon, shape

from app.schemas import GenerationRequest, GeoJSONFeature, GeoJSONGeometry
from app.engine import (
    generate_base_graph, 
    planarize_graph, 
    resolve_voids_and_zone, 
    finalize_city
)

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------
@pytest.fixture
def valid_request():
    """Provides a standard generation request with a 1km x 1km site boundary."""
    return GenerationRequest(
        site_polygon=GeoJSONFeature(
            geometry=GeoJSONGeometry(
                type="Polygon",
                coordinates=[[[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0], [0.0, 0.0]]]
            )
        ),
        target_node_count=60,
        max_valency=4,
        major_node_ratio=0.2,
        arterial_length=150.0,
        local_length=60.0,
        cross_link_probability=0.8,
        freezing_age_threshold=20,
        void_resolution_strategy="mixed",
        park_ratio=0.3,
        arterial_width=12.0,
        local_width=6.0
    )

@pytest.fixture
def base_graph(valid_request):
    """Provides a pre-generated Phase 2 graph for downstream tests."""
    return generate_base_graph(valid_request)

@pytest.fixture
def planar_graph(base_graph):
    """Provides a pre-planarized Phase 3 graph for downstream tests."""
    return planarize_graph(base_graph)

# ---------------------------------------------------------
# Phase 1 & 2: Kinematic Growth Tests
# ---------------------------------------------------------
def test_generate_base_graph_invariants(base_graph, valid_request):
    """Test that the initial growth obeys topological rules and schemas."""
    assert isinstance(base_graph, nx.Graph)
    assert len(base_graph.nodes) > 0
    assert len(base_graph.nodes) <= valid_request.target_node_count
    
    # Check node DNA
    for node, data in base_graph.nodes(data=True):
        assert 'pos' in data
        assert 'type' in data
        assert 'age' in data
        assert data['type'] in ['major', 'minor']
        assert base_graph.degree[node] <= valid_request.max_valency

    # Check edge DNA
    for u, v, data in base_graph.edges(data=True):
        assert 'weight' in data
        assert 'is_arterial' in data
        assert isinstance(data['is_arterial'], bool)

# ---------------------------------------------------------
# Phase 3: Planarization Tests
# ---------------------------------------------------------
def test_planarize_graph_state_inheritance(base_graph):
    """Test that Planarization does not wipe out node ages or edge arterial flags."""
    # Count original hot vs frozen nodes
    orig_ages = [data['age'] for _, data in base_graph.nodes(data=True)]
    max_orig_age = max(orig_ages) if orig_ages else 0
    
    planar_G = planarize_graph(base_graph)
    
    # Ensure ages survived
    new_ages = [data['age'] for _, data in planar_G.nodes(data=True)]
    assert max(new_ages) == max_orig_age, "Planarization wiped out historical node ages!"
    
    # Ensure arterial flags survived
    arterials_exist = any(data.get('is_arterial', False) for _, _, data in planar_G.edges(data=True))
    orig_arterials_exist = any(data.get('is_arterial', False) for _, _, data in base_graph.edges(data=True))
    
    if orig_arterials_exist:
        assert arterials_exist, "Planarization wiped out arterial edge flags!"

def test_planarize_graph_is_planar(planar_graph):
    """Ensure no edges cross each other (excluding touching endpoints)."""
    edges = list(planar_graph.edges(data=True))
    for i, (u1, v1, _) in enumerate(edges):
        line1 = LineString([planar_graph.nodes[u1]['pos'], planar_graph.nodes[v1]['pos']])
        for j in range(i + 1, len(edges)):
            u2, v2, _ = edges[j]
            line2 = LineString([planar_graph.nodes[u2]['pos'], planar_graph.nodes[v2]['pos']])
            # If they cross in the interior, planarization failed
            assert not line1.crosses(line2), f"Graph is not planar! Edge {u1}-{v1} crosses {u2}-{v2}"

# ---------------------------------------------------------
# Phase 4: Zoning & Void Resolution Tests
# ---------------------------------------------------------
def test_resolve_voids_and_zone(planar_graph, valid_request):
    """Test that voids >5 edges are detected and converted to parks or subdivided."""
    zoned_G, parks = resolve_voids_and_zone(planar_graph, valid_request)
    
    # Parks must be Shapely Polygons
    assert isinstance(parks, list)
    for p in parks:
        assert isinstance(p, Polygon)
        assert p.is_valid
    
    # Ensure zoned graph remains a valid networkx graph
    assert isinstance(zoned_G, nx.Graph)
    assert len(zoned_G.nodes) >= len(planar_graph.nodes), "Zoning should only add nodes/edges, never delete them"

# ---------------------------------------------------------
# Phase 5: Finalization Tests
# ---------------------------------------------------------
def test_finalize_city_output_schema(valid_request, planar_graph):
    """Test that the final output strictly matches the Pydantic schema structure."""
    zoned_G, parks = resolve_voids_and_zone(planar_graph, valid_request)
    
    response = finalize_city(zoned_G, parks, valid_request)
    
    # Validate the top-level response
    assert response.blocks.type == "FeatureCollection"
    assert response.streets.type == "FeatureCollection"
    assert "total_nodes" in response.metadata
    
    # Validate Blocks
    for block in response.blocks.features:
        assert block.geometry.type == "Polygon"
        assert "block_id" in block.properties
        assert "landuse" in block.properties
        assert block.properties["landuse"] in ["development", "park"]
        assert block.properties["area"] > 0
        
    # Validate Streets
    for street in response.streets.features:
        assert street.geometry.type == "Polygon"
        assert street.properties["type"] == "asphalt"

def test_finalize_city_boolean_subtraction(valid_request, planar_graph):
    """Ensure blocks do not overlap with streets (validating boolean difference)."""
    zoned_G, parks = resolve_voids_and_zone(planar_graph, valid_request)
    response = finalize_city(zoned_G, parks, valid_request)
    
    # If the boolean difference worked, the intersection of any block with any street should be roughly 0
    if len(response.blocks.features) > 0 and len(response.streets.features) > 0:
        
        # FIX: Use shape() to read the full GeoJSON (including holes) instead of just coords[0]
        block_geom = response.blocks.features[0].geometry.model_dump()
        street_geom = response.streets.features[0].geometry.model_dump()
        
        block_poly = shape(block_geom)
        street_poly = shape(street_geom)

        intersection_area = block_poly.intersection(street_poly).area
        assert intersection_area < 1.0, f"Overlap detected! Area: {intersection_area}"