import pandas as pd
from fpdf import FPDF

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
        self.save_pdf()

    def tldr_report(self):

        # cell height
        ch = 8
        self.pdf.add_page()

        # tldr title
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.cell(w=0, h=ch, txt="tl;dr Report on Device Activity:", align="C")
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

        # We only care about addresses for the summary page
        self.pdf.set_font('Arial', '', 10)
        loi_addresses = pd.DataFrame(self.profile.lois.address.unique(), columns=['address'])
        self.display_dataframe(loi_addresses, w=160)

        # Co-locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.display_dataframe(self.profile.coloc.advertiser_id.to_frame(), w=160)

        # Pattern of life
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Pattern of Life:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.set_font('Arial', '', 10)

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

        # Locations of interest
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.multi_cell(w=0, h=ch, txt="All Locations of Interest were flagged for either repeated visits separated by more than " + str(self.profile.rep_duration) +
            " hours, or extended stays at the location for over "+ str(self.profile.ext_duration) + " hours. Locations were determined with a geohash precision of 10.")
        self.pdf.ln(ch)
        # Everything except the adresses
        relevant_features = ['geohash', 'datetime', 'latitude', 'longitude']
        self.display_dataframe(self.profile.lois[relevant_features])
        self.pdf.ln(ch)

        # Now we display the resolved addresses (This is mostly for spacing issues since addresses are long)
        self.pdf.multi_cell(w=0, h=ch, txt="The above Latitudes and Longitudes were resolved to the following addresses.")
        self.pdf.ln(ch)
        self.display_dataframe(self.profile.lois.address.to_frame(), w=160)

        # Co-locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.display_dataframe(self.profile.coloc.advertiser_id.to_frame(), w=160)

    # TODO
    # fix this so we can save where we want to
    def save_pdf(self):
        output_path = self.profile.name + '.pdf'
        self.pdf.output(output_path, 'F')

    def display_dataframe(self, df, w=45):

        # feature names
        self.pdf.set_font('Arial', 'B', 12)
        for col in df.columns:
                self.pdf.cell(w, 8, str(col), border=1, align='C')#, new_x=index*40)
        self.pdf.ln(8)

        # the actual data
        self.pdf.set_font('Arial', '', 10)
        for j,row in df.iterrows():
            for datum in row.values:
                self.pdf.cell(w, 8, str(datum), border=1,align='L')
            self.pdf.ln(8)

