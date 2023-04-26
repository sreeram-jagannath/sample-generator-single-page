import streamlit as st
import base64


def add_logo():
    img = "./pepsico-logo.png"
    st.markdown(
        f"""
            <div style="margin-right: 20px;">
                <center><img src="data:image/png;base64,{base64.b64encode(open(img, "rb").read()).decode()}"></center>
            </div>
        """,
        unsafe_allow_html=True,
    )


def header_ui(title="S2U Route Optimization"):
    st.markdown(
        f"""
        <style>
        .pageheader {{
            padding: 0px;
            width: 100%;
            margin-left: 0px;
            margin-top: 10px;
            margin-bottom: 50px;
        }}
        .pagetitle {{
            text-align: center; 
            #position: absolute; 
            width: 100%;  
            margin-bottom: 10px; 
            border: 2px solid black;
            font-size: 30px;
            font-weight: bold;
            padding: 10px;
            border-radius: 10px;
            background-color: #004B93;
            #color: black; 
            color: white;
        }}
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='pageheader'>
            <p class='pagetitle'>
                {title}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )