import pandas as pd
from fpdf import FPDF
from agile.mapping import data_map
import os
import io
from PIL import Image
from agile.filtering import query_location, query_date, query_adid, query_node
from agile.utils.tag import find_all_nearby_nodes

class PDF(FPDF):

    def __init__(self):
        super().__init__()

    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'A.G.I.L.E. Device Activity Report', 0, 1, 'C')
        # The (copyright-free!!) logo
        #self.pdf.image("../img/new_logo.png", w=75, h=100, x=70, y=150)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

class Report:

    def __init__(self, profile):
        self.pdf = PDF()
        self.profile = profile
        self.full_report()
        self.file_name = self.save_pdf()

    def full_report(self):
        # cell height
        ch = 8
        
        self.pdf.add_page()

        # full title
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.cell(w=0, h=ch, txt="Full Report on Device Activity:", align="C")
        self.pdf.ln(ch)

        # advertiser ID and codename
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Device Details:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.cell(w=30, h=ch, txt="Codename: " + self.profile.name, ln=1)
        self.pdf.cell(w=30, h=ch, txt="AdID: " + self.profile.ad_id, ln=1)

        #disclaimer if there isn't enough data
        if self.profile.sd == 0:
            self.pdf.set_font('Arial', '', 14)
            self.pdf.cell(w=0, h=ch, txt="Warning: Limited information available for this device.", align="C")
            self.pdf.ln(ch)
            self.pdf.cell(w=0, h=ch, txt="This may affect the accuracy of the report.", align="C")

        
        # Co-locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        try:
            relevant_features = ['Colocated ADIDs','Alias','lat/long', 'datetime']
            self.display_dataframe(self.profile.coloc[relevant_features], [70, 35, 50, 40])
        except:
            self.display_dataframe(pd.DataFrame())
        
        self.pdf.add_page()
        
        # Locations of interest
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.multi_cell(w=0, h=ch, txt="All Locations of Interest were flagged for either repeated visits separated by more than " + str(self.profile.rep_duration) +
            " hours, or extended stays at the location for over "+ str(self.profile.ext_duration) + " hours. Locations were determined with a geohash precision of 10.")
        self.pdf.ln(ch)
        relevant_features = ['geohash', 'datetime', 'latitude', 'longitude']
        #try:
        self.display_dataframe(self.profile.lois[relevant_features], [48, 48, 48, 48])
        #except:
        #    self.display_dataframe(pd.DataFrame())
        self.pdf.ln(ch)

        self.pdf.add_page()
        
        # Named Locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Named Locations:", ln=1)
        # Overpass API polyline
        adid = self.profile.ad_id
        radius = 20
        query_data = query_adid(adid, self.profile.data) # Filter the data
        res = find_all_nearby_nodes(query_data, radius)
        res = res.drop_duplicates(subset = 'name', keep=False)
        try:
            self.display_dataframe(res, [110, 40, 40])
        except:
            self.display_dataframe(pd.DataFrame())

    # TODO
    # fix this so we can save where we want to
    def save_pdf(self):
        output_path = self.profile.name + '.pdf'
        self.pdf.output(output_path, 'F')
        return output_path

    def display_dataframe(self, df, column_widths = None):
        self.pdf.set_font('Arial', 'B', 12)
        
        if column_widths is None:
            column_widths = [40] * len(df.columns)  # default width for all columns
        
        i = 0
        for col in df.columns:
            self.pdf.cell(column_widths[i], 8, str(col), border=1, align='C')#, new_x=index*40)
            i += 1
        self.pdf.ln()

        self.pdf.set_font('Arial', '', 10)

        for index, row in df.iterrows():
            for i, value in enumerate(row):
                self.pdf.cell(column_widths[i], 10, str(value), border=1)
            self.pdf.ln()



        # feature names
        #self.pdf.set_font('Arial', 'B', 12)
        #i = 0
        #for col in df.columns:
        #        self.pdf.cell(w[i], 8, str(col), border=1, align='C')#, new_x=index*40)
        #        i += 1
        #self.pdf.ln(8)

        # the actual data
        #self.pdf.set_font('Arial', '', 10)
        #for j,row in df.iterrows():
        #    for datum in row.values:
        #        self.pdf.cell(w[j], 8, str(datum), border=1,align='L')
        #    self.pdf.ln(8)


    def display_lois(self, df, column_widths=None):
        if column_widths is None:
            column_widths = [40] * len(df.columns)  # default width for all columns

        # feature names
        self.pdf.set_font('Arial', 'B', 12)
        for i, col in ennumerate(df.columns):
                self.pdf.cell(column_widths[i], 8, str(col), border=1, align='C')#, new_x=index*40)
        self.pdf.ln(8)

        # the actual data
        self.pdf.set_font('Arial', '', 10)
        for j,row in df.iterrows():
            for datum in row.values:
                self.pdf.cell(column_widths[j], 8, str(datum), border=1,align='L')
            self.pdf.ln(8)


