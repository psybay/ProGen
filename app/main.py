import json
from fastapi import FastAPI
from app.schemas import GenerationRequest, GenerationResponse
from app.engine import generate_district

app = FastAPI(
    title="ProGen API",
    description="Vectorized Procedural Generation Engine",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "ProGen Procedural Generation Engine is active"}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Main endpoint for triggering a procedural generation run.
    """
    # Run vectorized generation
    gdf = generate_district(request)
    
    # Convert GeoDataFrame to GeoJSON dictionary
    geojson_str = gdf.to_json()
    geojson_dict = json.loads(geojson_str)
    
    return GenerationResponse(
        status="success",
        geojson=geojson_dict
    )
