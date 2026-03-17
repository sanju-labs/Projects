# client.py
import streamlit as st
from qdrant_client import QdrantClient
import config

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=str(config.QDRANT_PATH))