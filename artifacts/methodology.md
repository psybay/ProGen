# Procedural Generation Methodology

The graph generation flow inside `engine.py` implements a multi-phase, physics-based kinematic growth algorithm to generate urban street networks and blocks. The process is divided into five distinct methodological phases, transitioning from a topological graph skeleton to concrete, planar geometric configurations.

## Phase 1 & 2: Main Kinematic Growth Loop
The base graph is constructed in `generate_base_graph` through an organic, node-by-node growth process.

- **Seed Initialization**: The algorithm parses the input site boundary (GeoJSON) and places the initial seed node at the exact center of the site.
- **Node Spawning**: Nodes are iteratively spawned outwards. They are randomly designated as `major` (arterial) or `minor` (local). The distance of spawning depends on these classifications.
- **Boundary Restrictions**: New nodes are strictly bounded; if a spawn falls outside the `site_polygon`, it is discarded.
- **Topological Weaving (Cross-Linking)**: Periodically, `_attempt_cross_link` attempts to form closed loops of 3, 4, or 5 edges. It enforces planarity by preventing new secondary bonds from crossing existing ones.
- **Physics Relaxation & Historical Freezing**: As nodes age, they are "frozen" once they pass a predefined `freezing_age_threshold`. A spring layout relaxation (`nx.spring_layout`) runs in the background, allowing younger nodes to naturally adjust their positions (repelling/attracting) while anchored by the frozen older nodes.

## Phase 3: Mesh Healing (Planarization)
In `planarize_graph`, the topological network is forcefully planarized. 

- **Intersection Resolution**: Any remaining edge crossings are sliced into separate segments, effectively inserting a new intersection node at the crossing point.
- **Attribute Inheritance**: Newly created segments inherit their parent's properties, specifically checking if the original line was an arterial route (`is_arterial`), and preserving node `age` and `type` for nodes that existed previously.

## Phase 4: Zoning & Void Resolution
The algorithm identifies internal spaces ("faces") bounded by the street network using polygonization in `resolve_voids_and_zone`.

- **Void Detection**: Any closed loop consisting of more than 5 edges (indicating an abnormally large, empty block) is flagged as a "Void".
- **Resolution Strategy**: Depending on the request strategy, voids are either preserved entirely as green space (parks) or dynamically subdivided.
- **Delaunay Subdivision**: If subdivision is selected, the algorithm calculates a Delaunay triangulation of the void's perimeter nodes, creating new internal streets that organically break up the large block.

## Phase 5: Final Annealing & Translation
The last phase (`finalize_city`) transitions the abstract mathematical graph into solid geospatial data suitable for rendering and export.

- **Final Annealing**: A quick spring layout relaxation smooths out the jagged internal streets created by the Delaunay subdivision, anchoring the historically frozen nodes once more.
- **Street Buffering**: Mathematical edges are expanded into 2D polygons using Shapely's `buffer`. Arterial lines use a wider buffer than local streets, creating an accurate physical footprint of the road network.
- **Block Subtraction (Boolean Ops)**: The urban blocks are derived by taking the master site boundary and mathematically subtracting the entire master street network.
- **GeoJSON Serialization**: Finally, the individual block polygons and the unified street network are mapped to standard GeoJSON FeatureCollections (`Pydantic` schemas) with metadata regarding land use and area.
