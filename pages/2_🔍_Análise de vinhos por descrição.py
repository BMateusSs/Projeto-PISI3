import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise por Descrição de Vinhos",
    page_icon="🍷",
    layout="wide"
)

# Cores padronizadas
CORES_VINHO = {
    'tinto': '#8B0000',
    'branco': '#F5DEB3',
    'rose': '#FFB6C1',
    'espumante': '#FFD700'
}

@st.cache_data
def load_data():
    """Carrega o dataset de descrições de vinhos"""
    try:
        df = pd.read_parquet('data/processed/winemag.parquet')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Pré-processa os dados para análise"""
    if df is None:
        return None
    
    # Criar cópia para não modificar o original
    df_processed = df.copy()
    
    # Limpar dados - manter apenas dados com descrição, país e variedade
    df_processed = df_processed.dropna(subset=['description', 'country', 'variety'])
    
    # Limpar descrições
    df_processed['description_clean'] = df_processed['description'].str.lower()
    
    # Extrair ano do título (se disponível)
    df_processed['ano'] = df_processed['title'].str.extract(r'(\d{4})').astype(float)
    
    return df_processed

def main():
    st.title("🍷 Análise Exploratória por Descrição de Vinhos")
    st.markdown("---")
    
    # Carregar dados
    df = load_data()
    if df is None:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo existe.")
        return
    
    # Pré-processar dados
    df_processed = preprocess_data(df)
    if df_processed is None:
        st.error("Erro no pré-processamento dos dados.")
        return

if __name__ == "__main__":
    main()