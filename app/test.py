import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import proximitypyhash as pph
import pygeohash as gh
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium

# Our function imports
import resources.location as loc 
import resources.adid as adid
import resources.date as date
import resources.overpassQuery as opq
import resources.loi as loi
import user.profile as prof
from user.report import Report

#Test action for report generation
df = pd.read_csv("data/_54aa7153-1546-ce0d-5dc9-aa9e8e371f00_weeklong.csv")
data = adid.query_adid("54aa7153-1546-ce0d-5dc9-aa9e8e371f00", df)
ubl = prof.Profile("54aa7153-1546-ce0d-5dc9-aa9e8e371f00")
ubl.lois = loi.LOI(data, 10, 7, 24)
Report('report.pdf', ubl)