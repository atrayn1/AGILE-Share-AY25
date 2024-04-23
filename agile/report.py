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
        self.tldr_report()
        self.full_report()
        self.file_name = self.save_pdf()

    def tldr_report(self):

        # cell height
        ch = 8
        self.pdf.add_page()

        # tldr title
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.cell(w=0, h=ch, txt="tl;dr Report on Device Activity:", align="C")
        self.pdf.ln(ch)

        #disclaimer if there isn't enough data
        if self.profile.sd == 0:
            self.pdf.set_font('Arial', '', 14)
            self.pdf.cell(w=0, h=ch, txt="Warning: Limited information available for this device.", align="C")
            self.pdf.ln(ch)
            self.pdf.cell(w=0, h=ch, txt="This may affect the accuracy of the report.", align="C")

        # advertiser ID and codename
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Device Details:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.cell(w=30, h=ch, txt="Codename: " + self.profile.name, ln=1)
        self.pdf.cell(w=30, h=ch, txt="AdID: " + self.profile.ad_id, ln=1)

        # Locations of interest
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        # Overpass API polyline
        adid = self.profile.ad_id
        radius = 20
        query_data = query_adid(adid, self.profile.data) # Filter the data
        res = find_all_nearby_nodes(query_data, radius)
        res = res.drop_duplicates(subset = 'name', keep=False)
        self.display_dataframe(res, w=58)

        # Co-locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        try:
            self.display_dataframe(self.profile.coloc.advertiser_id.to_frame(), w=160)
        except:
            self.display_dataframe(pd.DataFrame())

        # Pattern of life
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Pattern of Life:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.set_font('Arial', '', 10)

    def full_report(self):
        # cell height
        ch = 8
        
        """# advertiser ID and codename
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Device Details:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.cell(w=30, h=ch, txt="Codename: " + self.profile.name, ln=1)
        self.pdf.cell(w=30, h=ch, txt="AdID: " + self.profile.ad_id, ln=1)
        
        
        # Add the map of all the data points in
        self.pdf.add_page()
        # Title for the map
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.cell(w=0, h=ch, txt="Map of ADID pins", align="C")
        self.pdf.ln(ch)
        
        
        m = data_map(data=self.profile.data[self.profile.data['advertiser_id'] == self.profile.ad_id])
        map_file_output = f'./saved_data/{self.profile.name}.html'
        map_image_output = f'./saved_data/{self.profile.name}.png'
        
        img_data = m._to_png(3)
        img = Image.open(io.BytesIO(img_data))
        img.save(map_image_output)
        pdf.image(os.path.abspath(map_image_output, x=10, y=50, w=180))
    
        self.pdf.set_font('Arial', '', 16)
        self.pdf.cell(w=30, h=ch, txt="Could not generate the map for this ADID", ln=1)"""
        
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

        # Locations of interest
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.multi_cell(w=0, h=ch, txt="All Locations of Interest were flagged for either repeated visits separated by more than " + str(self.profile.rep_duration) +
            " hours, or extended stays at the location for over "+ str(self.profile.ext_duration) + " hours. Locations were determined with a geohash precision of 10.")
        self.pdf.ln(ch)
        # Everything except the adresses
        relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'address']
        try:
            self.display_dataframe(self.profile.lois[relevant_features], w=32)


        except:
            self.display_dataframe(pd.DataFrame())
        self.pdf.ln(ch)

        # Now we display the resolved addresses (This is mostly for spacing issues since addresses are long)
        #self.pdf.multi_cell(w=0, h=ch, txt="The above Latitudes and Longitudes were resolved to the following addresses.")
        #self.pdf.ln(ch)
        #try:
        #    self.display_dataframe(self.profile.lois.address.to_frame(), w=160)
        #except:
        #    self.display_dataframe(pd.DataFrame())

        # Co-locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        try:
            relevant_features = ['Colocated ADIDs','Alias','lat/long']
            self.display_dataframe(self.profile.coloc[relevant_features], 70)
        except:
            self.display_dataframe(pd.DataFrame())

    # TODO
    # fix this so we can save where we want to
    def save_pdf(self):
        output_path = self.profile.name + '.pdf'
        self.pdf.output(output_path, 'F')
        return output_path

    def display_dataframe(self, df, w=45):

        # feature names
        self.pdf.set_font('Arial', 'B', 12)
        for col in df.columns:
                self.pdf.cell(w, 8, str(col), border=1, align='C')#, new_x=index*40)
        self.pdf.ln(8)

        # the actual data
        self.pdf.set_font('Arial', '', 7)
        for j,row in df.iterrows():
            for datum in row.values:
                self.pdf.cell(w, 8, str(datum), border=1,align='L')
            self.pdf.ln(8)


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


