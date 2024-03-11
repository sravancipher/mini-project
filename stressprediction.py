import streamlit as st
def app():    
    st.markdown("""
                <h1 style='text-align:center;color:brown;'><b>We are here to predict your stress</b></h1>
        """,unsafe_allow_html=True)
    form=st.form("form1")
    sr=form.number_input("Snoring Rate",key="1")
    rr=form.number_input("Respiration Rate",key="2")
    bt=form.number_input("Body Temperature",key="3")
    lm=form.number_input("Limb Movement",key="4")
    bo=form.number_input("Blood Oxygen",key="6")
    em=form.number_input("Eye Movement",key="7")
    sh=form.number_input("Sleeping Hours",key="8")
    hr=form.number_input("Heart Rate",key="9")
    submit=form.form_submit_button("Submit Data")
    st.subheader("Let us predict your stress....")
    if submit:
        st.markdown("""<h2>Stress Level:</h2>""",unsafe_allow_html=True)
    # st.markdown("""<style>
    #             .dvn-scroller.glideDataEditor{
    #             visibility:hidden;
    #             }
    #             </style>""",unsafe_allow_html=True)