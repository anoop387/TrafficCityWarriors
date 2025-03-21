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
    input_df = pd.read_csv(uploaded_file, index_col=None)
    profile = ProfileReport(input_df, title="Summary of the Data")
    st.subheader("Uploaded CSV File- Sample Dataset")
    st.write(input_df)
    st.subheader("Pandas Review")
    st_profile_report(profile)

    st.subheader("OpenAI Analysis")
    os.environ['_BARD_API_KEY'] = "Your Key"
    input_text = st.text_area("Regulatory Reporting Instruction")
    my_data = input_df
    answer = Bard().get_answer(f'{input_text} this data {my_data}')['content']
    st.write(f'{answer}')
 # Create a sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")

    # Apply sentiment analysis to a text column
    input_df["sentiment"] = input_df["text_column"].apply(lambda text: sentiment_analyzer(text)[0]["label"])

    # Save the modified DataFrame
    input_df.to_csv("data_with_sentiment.csv", index=False)

else:
    st. markdown('----')
    st.write("You did not upload a new file")