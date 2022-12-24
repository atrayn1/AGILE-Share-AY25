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
        self.cell(0, 8, 'A.G.I.L.E. User Activity Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

# Generate PDF
class Report:
    def __init__(self, path, profile):
        self.pdf = PDF()
        self.path = path
        self.profile = profile
        self.titlePage()
        self.contentPage()
        self.savePDF()
    
    def titlePage(self):
        # cell height
        ch = 8
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 36)

        #Name / Title of report

        self.pdf.cell(w=0, h=100, txt=self.profile.name, align="C")
        self.pdf.ln(ch)
        self.pdf.cell(w=0, h=120, txt="Report on User Activity", align="C")

        #The logo
        self.pdf.ln(ch)
        self.pdf.image("../images/new_logo.png", w=75, h=100, x=70, y=150)

    def contentPage(self):
        self.pdf.add_page()
        # cell height
        ch = 8

        #self.pdf.set_font('Arial', '', 16)
        #self.pdf.cell(w=30, h=ch, txt="Date: ", ln=0)
        #self.pdf.cell(w=30, h=ch, txt="12/23/2022", ln=1)
        #self.pdf.cell(w=30, h=ch, txt="Author: ", ln=0)
        #self.pdf.cell(w=30, h=ch, txt="Ernest Son", ln=1)
        #self.pdf.cell(w=30, h=ch, ln=0)
        #self.pdf.cell(w=30, h=ch, txt="Sam Chanow", ln=1)

        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Device Details:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        #self.pdf.multi_cell(w=0, h=ch, txt="Lorem ipsum dolor sit amet...")
        self.pdf.cell(w=30, h=ch, txt="Codename: " + self.profile.name, ln=1)
        self.pdf.cell(w=30, h=ch, txt="AdID: " + self.profile.adid, ln=1)

        #self.pdf.image('./example_image.png', x = 10, y = None, w = 100, h = 0, type = 'PNG', link = '')

        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Locations of Interest:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        #self.pdf.cell(w=30, h=ch, txt=self.profile.lois.to_string(), ln=1)
        self.displayDataframe(self.profile.lois)

        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Co-located Devices:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.displayDataframe(self.profile.coloc)

        self.pdf.ln(ch)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(w=0, h=ch, txt="Pattern of Life:", ln=1)
        self.pdf.set_font('Arial', '', 16)
        self.pdf.multi_cell(w=0, h=ch, txt="TBD")



    def savePDF(self):
        self.pdf.output(self.path, 'F')

    def displayDataframe(self, df):

        self.pdf.set_font('Arial', 'B', 12)
        for col in df.columns:
            self.pdf.cell(30, 8, str(col), border=1, align='C')#, new_x=index*40)

        self.pdf.set_font('Arial', '', 10)
        for j,row in df.iterrows():
            for datum in row.values:
                self.pdf.cell(30, 8, str(datum), border=1,align='L')
            self.pdf.ln(8)

        #first print columns
        #for index, col in enumerate(df.columns):
        #    print(index, col)
        #    self.pdf.cell(30, 8, str(col), 1, index)
        
        #self.pdf.ln()

        #We print out each cell row by row
        ##for index, row in df.iterrows():
        #    for data in row.values:
        #        pass


# TESTING
#df = pd.DataFrame({'geohash':['asdf','asdf','asdf'], 'datetime':['mon','tue','wed'], 'latitude':[69, 70, 71], 'longitude':[420, 421, 422], 'advertiser_id':['ubl', 'ubl', 'ubl']})
#Report('test.pdf', df)

