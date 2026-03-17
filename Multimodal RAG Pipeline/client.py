import streamlit as st
from qdrant_client import QdrantClient

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["https://ed6d60c4-9057-488b-81ab-15233a1fe1d3.us-west-1-0.aws.cloud.qdrant.io"],
        api_key=st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.211fQQR87FoCfedkDmtYatrfnyGYqYSf436JZwJ7LN0"],
    )