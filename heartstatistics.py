import streamlit as st
def app():
    st.header("Heart Statistics")
    res="HELLO"
    st.download_button(label="Download File",data=res,file_name="file.csv")