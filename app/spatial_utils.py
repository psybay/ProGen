import numpy as np
from shapely.geometry import Point, Polygon

def generate_random_points(polygon: Polygon, count: int):
    """
    Vectorized random point generation within a polygon boundary.
    """
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    
    # Simple rejection sampling (can be further vectorized)
    while len(points) < count:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            points.append(p)
            
    return points

def numpy_to_shapely(coords: np.ndarray):
    """
    Converts NumPy arrays to Shapely geometries efficiently.
    """
    return [Point(xy) for xy in coords]
