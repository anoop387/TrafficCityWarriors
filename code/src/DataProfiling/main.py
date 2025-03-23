from nbformat import write
from bardapi import Bard
import os
import streamlit as st
import pandas as pd
import numpy as np
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from transformers import pipeline

print("Hello, PyCharm!")
st.title("Data Profiling App")
st.subheader("This app will help you to do Data Exploration")
st.sidebar.header("User Input Features")
uploaded_file = st. sidebar.file_uploader("upload your input Excel file", type=["csv"])
if uploaded_file is not None:
    st .markdown("----")
    my_data = pd.read_csv(uploaded_file, index_col=None)

    st.subheader("Uploaded CSV File- Sample Dataset")
    st.write(my_data)

    st.subheader("OpenAI Analysis")
    api_key = os.environ.get("_BARD_API_KEY")

    default_multiline_text = """Transaction Amount should always match Reported Amount, except when the transaction involves cross-currency conversions, in which case a permissible deviation of up to 1% is allowed.
Account Balance should never be negative, except in cases of overdraft accounts explicitly marked with an "OD" flag.
Currency should be a valid ISO 4217 currency code, and the transaction must adhere to cross-border transaction limits as per regulatory guidelines.
Country should be an accepted jurisdiction based on bank regulations, and cross-border transactions should include mandatory transaction remarks if the amount exceeds $10,000.Display Row Level Transaction Details"""

    input_text = st.text_area("Regulatory Reporting Instruction",  height=250,value=default_multiline_text)
    answer = Bard(token=api_key).get_answer(f'{input_text} this data {my_data}')['content']
    st.write(f'{answer}')

    input_text2 = st.text_area("ChatBot - Raise Your Question")
    answer2 = Bard(token=api_key).get_answer(f'{input_text2} this data {my_data}')['content']
    st.write(f'{answer2}')

    st.subheader("Pandas Review")
    profile = ProfileReport(my_data, title="Summary of the Data")
    st_profile_report(profile)

else:
    st. markdown('----')
    st.write("You did not upload a new file")
