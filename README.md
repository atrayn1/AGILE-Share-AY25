# AGILE
## Advertising and Geolocation Information Logical Extractor

## Table of Contents

- [Background](#background)
- [Dependencies](#dependencies)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)

## Background

The working repository for our 2023 Capstone Project. ADD MORE DESCRIPTION LATER.

## Dependencies

```
- streamlit
- proximitypyhash
- pygeohash
- pandas
- matplotlib
- Bokeh
- streamlit-folium
- overpy
- geopy
- scikit-learn==1.0.2
- fpdf2
```
The dependencies are also contained within `agile/requirements.txt` for use with the docker image.

## Install

```
foo@bar:../AGILE/agile$ docker build -t "DOCKER IMAGE NAME" --build-arg SSH_PRIVATE_KEY="PRIVATE KEY HERE" .
foo@bar:../AGILE/agile$ docker run -p 8501:8501 "DOCKER IMAGE NAME"
```

## Usage

```
foo@bar:../AGILE/agile$ streamlit run app.py
```

## Contributing

The coding team for AGILE:

- SAMUEL CHANOW
- ERNEST SON
- KATIE DESSAUER