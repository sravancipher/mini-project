import streamlit as st
def app():
    team_member1,n,team_member2=st.columns([5,2,5])
    team_member3,n,team_member4=st.columns([5,2,5])
    team_member1.markdown("""
         <div style='width:100%;text-align:center;box-shadow:4px 5px 9px 20px gray;border-radius:25px'>
            <h2 style='color:brown'>Sowjanya</h2>
            <p>Machine Learning</p>
            <h3><a style='text-decoration:none',href="linkedin.com/in/sravan-potnuru-a431a0215/">Linkedin</a></h3>
                             </div><br><br><br>
                             """,unsafe_allow_html=True)
    team_member2.markdown("""
         <div style='width:100%;text-align:center;box-shadow:4px 5px 9px 20px gray;border-radius:25px'>
            <h2 style='color:brown'>Jagadam Divya</h2>
            <p>Machine Learning</p>
            <h3><a style='text-decoration:none',href="linkedin.com/in/sravan-potnuru-a431a0215/">Linkedin</a></h3>
                             </div><br><br><br>
                             """,unsafe_allow_html=True)
    team_member3.markdown("""
         <div style='width:100%;text-align:center;box-shadow:4px 5px 9px 20px gray;border-radius:25px'>
            <h2 style='color:brown'>Vanapalli Sarada</h2>
            <p>Machine Learning</p>
            <h3><a style='text-decoration:none',href="linkedin.com/in/sravan-potnuru-a431a0215/">Linkedin</a></h3>
                             </div><br><br><br>
                             """,unsafe_allow_html=True)
    team_member4.markdown("""
         <div style='width:100%;text-align:center;box-shadow:4px 5px 9px 20px gray;border-radius:25px'>
            <h2 style='color:brown'>Potnuru Sravan</h2>
            <p>Machine Learning</p>
            <h3><a style='text-decoration:none',href="linkedin.com/in/sravan-potnuru-a431a0215/">Linkedin</a></h3>
                             </div><br><br><br>
                             """,unsafe_allow_html=True)
    st.markdown("""
         <div style='width:100%;text-align:center;box-shadow:4px 5px 9px 20px gray;border-radius:25px'>
            <h2 style='color:brown'>Manikanta</h2>
            <p>Machine Learning</p>
            <h3><a style='text-decoration:none',href="linkedin.com/in/sravan-potnuru-a431a0215/">Linkedin</a></h3>
                             </div>
                             """,unsafe_allow_html=True)
app()