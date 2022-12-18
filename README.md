# AGILE

## Advertising and Geolocation Information Logical Extractor

### Team

- Samuel Chanow
- Ernest Son
- Katie Dessauer

### File Structure

- app/ contains all of the web application code and data
- app/resources/ contains functions for the web application

### Dependencies
- streamlit
- proximitypyhash
- pygeohash
- pandas
- matplotlib
- Bokeh
- streamlit-folium
- overpy
- geopy

### How do I run the demo?

```console
foo@bar:../AGILE/app$ streamlit run agile.py
```

## DONE

- Resolve addresses from lat/long coordinates

## TODO

- Get areas of interest for a specific data point
  - Effectively and efficiently identify clusters of data points
- Name areas of interest via reverse geocoding
  - centroid?
- Integrate an "areas of interest" query into webapp demo

## FUTURE

- Distinguish frequently visited areas of interest
- Distinguish dwell addresses from areas of interest
- Distinguish workplace addresses from areas of interest
- Identify journeys and end-of-journeys

