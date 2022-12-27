# AGILE

## Advertising and Geolocation Information Logical Extractor

### Team

- Samuel Chanow
- Ernest Son
- Katie Dessauer

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
- scikit-learn
- fpdf2

### How do I run the demo?

```console
foo@bar:../AGILE/agile$ streamlit run app.py
```

## DONE

- Resolve addresses from lat/long coordinates
- Get areas of interest for a specific data point
- Integrate an "areas of interest" query into webapp demo

## TODO

- Name areas of interest via reverse geocoding
  - median data point?
- Generate a PDF report of areas of interest for a specific adID

## FUTURE

- Distinguish frequently visited areas of interest
- Distinguish dwell addresses from areas of interest
- Distinguish workplace addresses from areas of interest
- Identify journeys and end-of-journeys

