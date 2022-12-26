import pandas as pd
from fpdf import FPDF

# Ernest Son

class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'A.G.I.L.E. Device Activity Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

# Generate PDF
class Report:
    def __init__(self, profile):
        self.pdf = PDF()
        self.profile = profile
        self.title_page()
        self.content_page()
        self.save_pdf()
    
    def title_page(self):
        # cell height
        ch = 8
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 36)

        # Name / Title of report
        self.pdf.cell(w=0, h=100, txt=self.profile.name, align="C")
        self.pdf.ln(ch)
        self.pdf.cell(w=0, h=120, txt="Report on Device Activity", align="C")

        # The (copyright-free!!) logo
        self.pdf.ln(ch)
        self.pdf.image("../images/new_logo.png", w=75, h=100, x=70, y=150)

    def content_page(self):
        self.pdf.add_page()

        # cell height
        ch = 8

        # advertiser ID and codename
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Device Details:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.cell(w=30, h=ch, txt="Codename: " + self.profile.name, ln=1)
        self.pdf.cell(w=30, h=ch, txt="AdID: " + self.profile.adid, ln=1)

        # Locations of interest
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        #self.pdf.cell(w=30, h=ch, txt=self.profile.lois.to_string(), ln=1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.multi_cell(w=0, h=ch, txt="All Locations of Interest were flagged for either repeated visits separated by more than " + str(self.profile.rep_duration) +
            " hours, or extended stays at the location for over "+ str(self.profile.ext_duration) + " hours. Locations were determined with a geohash precision of " + str(self.profile.prec) + ".")
        self.pdf.ln(ch)
        self.display_dataframe(self.profile.lois)

        # Co-locations
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        #self.display_dataframe(self.profile.coloc)
        self.pdf.multi_cell(w=0, h=ch, txt="TBD")

        # Pattern of life
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Pattern of Life:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        #self.display_dataframe(self.profile.pol)
        self.pdf.multi_cell(w=0, h=ch, txt="TBD")

    # TODO
    # fix this, hacky
    # Report() needs to be run in AGILE/agile/ in order to work...
    def save_pdf(self):
        output_path = '../data/' + self.profile.name + '.pdf'
        self.pdf.output(output_path, 'F')

    def display_dataframe(self, df):
        df = df.drop(columns=['advertiser_id'])

        # feature names
        self.pdf.set_font('Arial', 'B', 12)
        for col in df.columns:
                self.pdf.cell(45, 8, str(col), border=1, align='C')#, new_x=index*40)
        self.pdf.ln(8)

        # the actual data
        self.pdf.set_font('Arial', '', 10)
        for j,row in df.iterrows():
            for datum in row.values:
                self.pdf.cell(45, 8, str(datum), border=1,align='L')
            self.pdf.ln(8)

