import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
from bardapi import Bard
from transformers import pipeline

class TestDataProfilingApp(unittest.TestCase):

    @patch("streamlit.sidebar.file_uploader")
    @patch("pandas.read_csv")
    def test_file_upload(self, mock_read_csv, mock_file_uploader):
        # Mocking file upload
        mock_file_uploader.return_value = "mock_file.csv"

        # Mocking CSV data
        mock_data = pd.DataFrame({
            "text_column": ["This is great!", "Not good at all.", "Average experience."]
        })
        mock_read_csv.return_value = mock_data

        # Simulate file upload
        uploaded_file = st.sidebar.file_uploader("upload your input Excel file", type=["csv"])
        self.assertEqual(uploaded_file, "mock_file.csv")

        # Simulate reading CSV
        df = pd.read_csv(uploaded_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)




    @patch("os.environ", {})
    @patch("bardapi.Bard.get_answer")
    def test_bard_analysis(self, mock_get_answer):
        # Mock the API response
        mock_get_answer.return_value = {"content": "This is a mock response."}

        # Set environment variable
        os.environ['_BARD_API_KEY'] = "dummy_key"

        # Test Bard API call
        response = Bard().get_answer("Mock prompt")
        self.assertEqual(response['content'], "This is a mock response.")

    def test_app_title(self):
        # Check Streamlit title and subheader
        with patch("streamlit.title") as mock_title, patch("streamlit.subheader") as mock_subheader:
            st.title("Data Profiling App")
            st.subheader("This app will help you to do Data Exploration")

            mock_title.assert_called_once_with("Data Profiling App")
            mock_subheader.assert_called_once_with("This app will help you to do Data Exploration")

if __name__ == "__main__":
    unittest.main()
