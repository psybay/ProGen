import json
from fastapi import FastAPI
from .schemas import GenerationRequest, GenerationResponse
from .engine import generate_base_graph, planarize_graph, resolve_voids_and_zone, finalize_city

app = FastAPI(title="Urban Morphology Microservice")

@app.post("/generate-district", response_model=GenerationResponse)
async def generate_district_endpoint(request: GenerationRequest):
    
    # Phase 1 & 2: Kinematic Growth & Topological Weaving
    G_raw = generate_base_graph(request)
    
    # Phase 3: Mesh Healing
    G_planar = planarize_graph(G_raw)
    
    # Phase 4: Zoning & Void Resolution
    G_zoned, parks = resolve_voids_and_zone(G_planar, request)
    
    # Phase 5: Final Annealing & GeoJSON Translation
    final_payload = finalize_city(G_zoned, parks, request)
    
    return final_payload

@app.get("/")
async def root():
    return {"message": "ProGen Procedural Generation Engine is active"}

