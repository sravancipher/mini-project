import streamlit as st
import stresslevels as stress
def app():
    st.header("Heart Statistics")
    res=b'stress.app()'
    st.download_button(label="Download File",data=res,file_name="file.csv")