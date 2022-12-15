# AGILE

## Advertising and Geolocation Information Logical Extractor

### Team

Samuel Chanow
Ernest Son
Katie Dessauer

### File Structure

app/ contains all of the web application code and data
app/resources/ contains functions for the web application

### Dependencies
- streamlit
- proximitypyhash
- pygeohash
- pandas
- matplotlib
- Bokeh
- streamlit-folium
- overpy

### How do I run the demo?

```console
foo@bar:../AGILE/app$ streamlit run agile.py
```

## TODO

- Resolve addresses from lat/long coordinates
- Get points of interest for a specific data point
  -  to effectively and efficiently identify clusters
- Distinguish frequently visited points of interest
- Distinguish dwell addresses from points of interest
- Distinguish workplace addresses from points of interest


