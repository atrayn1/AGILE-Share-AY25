import pandas as pd
import proximitypyhash as pph
import folium
from streamlit_folium import folium_static

def data_map(container=None, data=None, lois=None):
    """
    Display a map in a Streamlit container.

    Parameters:
        container: Streamlit container to display map
        data: Pandas DataFrame of data points to display on map
        lois: Pandas DataFrame of locations of interest to display on map
    """
    if data is None and lois is None:
        return

    # Determine size of data to be displayed
    data_size = len(lois) if data is None else len(data)

    if data_size > 0:
        # Get coordinates of first data point
        first_point = lois.iloc[0] if data is None else data.iloc[0]
        lat, long = first_point.latitude, first_point.longitude
        
        # Create the map
        m = folium.Map(location=[lat, long], zoom_start=10)

        # Add data points to map
        if data is not None:
            data.apply(lambda row: folium.Marker(
                location=[row.latitude, row.longitude],
                popup=f"Location: {row.latitude}, {row.longitude}"
                        + (f" Node Name: {row.name}" if 'name' in row else "")
                        + (f" Timestamp: {row.datetime}" if 'datetime' in row else "")
                        + (f" Advertiser ID: {row.advertiser_id}" if 'advertiser_id' in row else "")
            ).add_to(m), axis=1)

        # Add locations of interest to map
        if lois is not None:
            lois.apply(lambda row: folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=30,
                popup="LOI"
            ).add_to(m), axis=1)

        if container != None:
            with container:
                # Display the map in a container, if there is one
                folium_static(m, width=725)
        else:
            return m
        
    else:
        if container != None: 
            container.write("No Data Points Available")
    
    return None
        
        
        
        
        
        
        
        
        
def return_map(container, data=None, lois=None):
    """
    Returns a map, rather than displays it in a Streamlit container.

    Parameters:
        container: Streamlit container to display map
        data: Pandas DataFrame of data points to display on map
        lois: Pandas DataFrame of locations of interest to display on map
    """
    if data is None and lois is None:
        return

    # Determine size of data to be displayed
    data_size = len(lois) if data is None else len(data)

    if data_size > 0:
        # Get coordinates of first data point
        first_point = lois.iloc[0] if data is None else data.iloc[0]
        lat, long = first_point.latitude, first_point.longitude

        # Create the map
        m = folium.Map(location=[lat, long], zoom_start=10)

        # Add data points to map
        if data is not None:
            data.apply(lambda row: folium.Marker(
                location=[row.latitude, row.longitude],
                popup=f"Location: {row.latitude}, {row.longitude}"
                        + (f" Node Name: {row.name}" if 'name' in row else "")
                        + (f" Timestamp: {row.datetime}" if 'datetime' in row else "")
                        + (f" Advertiser ID: {row.advertiser_id}" if 'advertiser_id' in row else "")
            ).add_to(m), axis=1)

        # Add locations of interest to map
        if lois is not None:
            lois.apply(lambda row: folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=30,
                popup="LOI"
            ).add_to(m), axis=1)

        # Display the map
        folium_static(m, width=725)
        
    
        

