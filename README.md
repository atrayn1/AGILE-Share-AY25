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
- streamlit==1.20.0
- proximitypyhash==0.2.1
- pygeohash==1.2.0
- pandas==1.5.2
- matplotlib==3.6.2
- Bokeh==3.0.2
- streamlit-folium==0.7.0
- geopy==2.3.0
- scikit-learn==1.2.1
- fpdf2==2.6.1
- networkx==3.0
- streamlit-option-menu==0.3.2
```
The dependencies are also contained within `AGILE/requirements.txt` for use with the docker image.

## Install

Default Installation (Recommended):

```
foo@bar:../AGILE$ source /bin/build.sh
foo@bar:../AGILE$ build # To build the docker image
foo@bar:../AGILE$ run # To run the docker image
```

Manual Installation:

```
foo@bar:../AGILE$ docker build -t docker_image_name .
foo@bar:../AGILE$ docker run -p 8501:8501 docker_image_name
```

NOTE: You may have to run either set of install commands with root permissions if using a docker version <19.03 or running from WSL.

## Usage (Without Docker)

```
foo@bar:../AGILE$ streamlit run app.py
```

## Contributing

The coding team for AGILE:

- RACHEL SUMMERS
- JOE REIMANN
- ALEX TONER
