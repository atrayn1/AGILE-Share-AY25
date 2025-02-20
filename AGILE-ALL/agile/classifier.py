from .prediction import haversine

def get_location_type(coords):
    """
    Prompts the user to classify a location based on its coordinates.
    
    Args:
        coords (tuple): A tuple representing (longitude, latitude).
    
    Returns:
        str: The relationship category provided by the user.
    """
    print(f"Coordinates: {coords}")
    category = input("Enter relationship category for these coordinates (work, friends, family): ").strip().lower()
    if category in ['work', 'friends', 'family']:
        return category
    else:
        print("Invalid input. Defaulting to 'unknown'.")
        return "unknown"

# Dictionary to store classified locations
classified_locations = {}

def classify_edge_by_location(edge, radius):
    """
    Classifies an edge based on its overlap periods, utilizing cached location classifications.
    
    Args:
        edge (Edge): The edge to classify.
        radius (float): The radius in meters to check for previously classified locations.
    
    Returns:
        dict: A dictionary with relationship categories as keys and the percentage
              of time spent in each category as values.
    """
    category_time = {"work": 0.0, "friends": 0.0, "family": 0.0, "unknown": 0.0}
    
    for period in edge.overlap_periods:
        time_tuple, coords = period
        start_time, end_time = time_tuple
        duration = (end_time - start_time).total_seconds()
        
        # Check if the location or a nearby one is already classified
        loc_category = None
        for (stored_coords, stored_category) in classified_locations.items():
            if haversine(coords[1], coords[0], stored_coords[1], stored_coords[0]) <= radius / 1000:
                loc_category = stored_category
                break
        
        # If not found, classify and store it
        if loc_category is None:
            loc_category = get_location_type(coords)
            classified_locations[coords] = loc_category
        
        category_time[loc_category] += duration
    
    total_duration = sum(category_time.values())
    if total_duration == 0:
        return {key: 0 for key in category_time}
    
    return {key: (duration / total_duration) * 100 for key, duration in category_time.items()}

def classifyEdges(graph, radius):
    """
    Iterates over all edges in the graph, classifies each one based on its location-based
    overlap periods, and assigns a dictionary as the relationship type that shows the percentage
    breakdown for each relationship category.
    
    Args:
        graph (Graph): The graph containing nodes and edges.
    """
    for edge in graph.edges:
        relationship_breakdown = classify_edge_by_location(edge, radius)
        # Store the classification dictionary on the edge
        edge.relationship_type = relationship_breakdown
        print(f"Classified edge between {edge.node1.adid} and {edge.node2.adid} as:")
        for category, percent in relationship_breakdown.items():
            print(f"  {category}: {percent:.2f}%")