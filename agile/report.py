import pandas as pd
from fpdf import FPDF
#PDF Report class for AGILE Device Activity Rreports
# Ernest Son
#Sam Chanow

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
        self.tldr_page()
        #self.title_page()
        self.content_page()
        self.save_pdf()

    def title_page(self):
        # cell height
        ch = 8
        self.pdf.add_page()

        # Name / Title of report
        self.pdf.set_font('Arial', 'B', 36)
        self.pdf.cell(w=0, h=100, txt=self.profile.name, align="C")
        self.pdf.ln(ch)
        self.pdf.cell(w=0, h=120, txt="Report on Device Activity", align="C")

        # The (copyright-free!!) logo
        self.pdf.ln(ch)
        self.pdf.image("../images/new_logo.png", w=75, h=100, x=70, y=150)

    def tldr_page(self):
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
        potential_dwells = self.profile.lois[self.profile.lois.potential_dwell == True]
        potential_dwells = pd.DataFrame(potential_dwells.address.unique(), columns=['address'])
        potential_workplaces = self.profile.lois[self.profile.lois.potential_workplace == True]
        potential_workplaces = pd.DataFrame(potential_workplaces.address.unique(), columns=['address'])
        self.pdf.cell(w=0, h=ch, txt="Potential dwell locations:", ln=1)
        self.display_dataframe(potential_dwells, w=160)
        self.pdf.cell(w=0, h=ch, txt="Potential workplaces:", ln=1)
        self.display_dataframe(potential_workplaces, w=160)

        # Here's an example of a dashed line in fpdf.
        # In practice, we're probably just going to force a new page, but we
        # might play around with pretty-printing a literal "tear-line."

        # Adds a dashed line beginning at point (10,30),
        #  ending at point (110,30) with a
        #  dash length of 1 and a space length of 10.
        #self.pdf.dashed_line(10, 30, 110, 30, 1, 10)

    def content_page(self):
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
            " hours, or extended stays at the location for over "+ str(self.profile.ext_duration) + " hours. Locations were determined with a geohash precision of " + str(self.profile.prec) + ".")
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

        # Pattern of life
        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Pattern of Life:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        named_locations = ['latitude', 'longitude', 'potential_dwell', 'potential_workplace']
        self.display_dataframe(self.profile.lois[named_locations])

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

