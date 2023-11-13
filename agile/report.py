import streamlit as stm

def generate_report(adid, st):


    if adid == "1":
        # Header
        st.title("T")
    
        # Section 1: Colocation
        st.markdown("## **Colocation**")
        st.write("121 Eckham St, New London, NC 39416 | 1730 - 0600")
        clinks = st.write("[CW1](#) | [C1](#) | [C2](#)")
        #if clinks:
         #   st.empty()    
          #  generate_report("2", st)

        # Section 2: Work
        st.markdown("## **Work**")
        st.write("4811 Anner Ave, Albemarle, NC 39416 | 0900 - 1700")
        st.write("[CW1](#) | [W1](#) | [W2](#) | [W3](#)")

        # Section 3: Hang-out
        st.markdown("## **Other**")
        st.write("41 Arthur Ave, Albemarle, NC 39416 | 0630 - 0830")
        hlinks = st.write("[H1](#)")
        #if hlinks:
         #   st.empty()
          #  generate_report("2", st)



    if adid == "2":
        # Header
        st.title("CW1")
        # Section 1: Colocation
        st.markdown("## **Colocation**")
        st.write("121 Eckham St, New London, NC 39416 | 2200 - 0830")
        st.write("[T](#) | [T-C1](#) | [T-C2](#)")

        # Section 2: Work
        st.markdown("## **Work**")
        st.write("4811 Anner Ave, Albemarle, NC 39416 | 0900 - 1700")
        st.write("[T](#) | [T-W1](#) | [T-W2](#) | [T-W3](#)")

        # Section 3: Hang-out
        st.markdown("## **Other**")
        st.write("21 Main St, Albemarle, NC 39416 | 1730 - 2130")
        st.write("[T-C2](#)")
        
    if adid == "3":
        # Header
        st.title("H1")

        # Section 1: Colocation
        st.markdown("## **Colocation**")
        st.write("14 Maple Ln | 1730 - 0600")
        st.write("N/A")

        # Section 2: Work
        st.markdown("## **Work**")
        st.write("21 3rd St, Albemarle, NC, 39416 | 0900 - 1700")
        st.write("[W1](#) | [W2](#) | [W3](#)")

        # Section 3: Hang-out
        st.markdown("## **Others**")
        st.write("41 Arthur Ave, Albemarle, NC 39416 | 0630 - 0830")
        st.write("[T](#)")


class Report:

    def __init__(self, adid, cont):
        generate_report(adid, cont)


"""

Generative AI was used to generate parts of this code





"""
