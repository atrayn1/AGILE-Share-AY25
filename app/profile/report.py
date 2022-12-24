# fpdf is way easier to use than reportlab... we don't need fancy formatting for
# an output report, presentation clarity is priority #1
import pandas as pd
from fpdf import FPDF

# Ernest Son

class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'AGILE', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

# Generate PDF
class Report:
    def __init__(self, path, data):
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(w=0, h=20, txt="adID goes here", ln=1)

        # cell height
        ch = 8

        pdf.set_font('Arial', '', 16)
        pdf.cell(w=30, h=ch, txt="Date: ", ln=0)
        pdf.cell(w=30, h=ch, txt="12/23/2022", ln=1)
        pdf.cell(w=30, h=ch, txt="Author: ", ln=0)
        pdf.cell(w=30, h=ch, txt="Ernest Son", ln=1)
        pdf.cell(w=30, h=ch, ln=0)
        pdf.cell(w=30, h=ch, txt="Sam Chanow", ln=1)

        pdf.ln(ch)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(w=0, h=ch, txt="Device Details:", ln=1)
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(w=0, h=ch, txt="Lorem ipsum dolor sit amet...")

        #pdf.image('./example_image.png', x = 10, y = None, w = 100, h = 0, type = 'PNG', link = '')

        pdf.ln(ch)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(w=0, h=ch, txt="...consectetur adipiscing elit...")

        pdf.ln(ch)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(w=0, h=ch, txt="...sed do eiusmod tempor incididunt ut labore et dolore magna aliqua...")

        pdf.ln(ch)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(w=0, h=ch, txt="Pattern of Life:", ln=1)
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(w=0, h=ch, txt="...Ut enim ad minim veniam...")

        pdf.ln(ch)

        # Table Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(40, ch, 'Latitude', 1, 0, 'C')
        pdf.cell(40, ch, 'Longitude', 1, 1, 'C')

        # Table contents
        pdf.set_font('Arial', '', 16)
        for i in range(0, len(data)):
            pdf.cell(40, ch, data['latitude'].iloc[i].astype(str), 1, 0, 'C')   
            pdf.cell(40, ch, data['longitude'].iloc[i].astype(str), 1, 1, 'C')

        pdf.output(path, 'F')

# TESTING
#df = pd.DataFrame({'geohash':['asdf','asdf','asdf'], 'datetime':['mon','tue','wed'], 'latitude':[69, 70, 71], 'longitude':[420, 421, 422], 'advertiser_id':['ubl', 'ubl', 'ubl']})
#Report('test.pdf', df)

