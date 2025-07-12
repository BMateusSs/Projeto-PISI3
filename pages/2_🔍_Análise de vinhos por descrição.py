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

def create_filters(df):
    """Cria filtros interativos"""
    st.sidebar.header("🔍 Filtros")
    
    # Filtro por país
    paises = ['Todos'] + sorted(df['country'].unique().tolist())
    pais_selecionado = st.sidebar.selectbox("País:", paises)
    
    # Filtro por variedade
    variedades = ['Todas'] + sorted(df['variety'].unique().tolist())
    variedade_selecionada = st.sidebar.selectbox("Variedade:", variedades)
    
    # Filtro por região
    if 'region_1' in df.columns:
        regioes = ['Todas'] + sorted(df['region_1'].dropna().unique().tolist())
        regiao_selecionada = st.sidebar.selectbox("Região:", regioes)
    else:
        regiao_selecionada = 'Todas'
    
    # Filtro por província
    if 'province' in df.columns:
        provincias = ['Todas'] + sorted(df['province'].dropna().unique().tolist())
        provincia_selecionada = st.sidebar.selectbox("Província:", provincias)
    else:
        provincia_selecionada = 'Todas'
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if pais_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['country'] == pais_selecionado]
    
    if variedade_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['variety'] == variedade_selecionada]
    
    if regiao_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['region_1'] == regiao_selecionada]
    
    if provincia_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['province'] == provincia_selecionada]
    
    return df_filtrado


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
    
    # Criar filtros
    df_filtrado = create_filters(df_processed)
    
    # Mostrar informações gerais
    st.header("📋 Visão Geral dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Vinhos", f"{len(df_filtrado):,}")
    with col2:
        st.metric("Países Únicos", f"{df_filtrado['country'].nunique()}")
    with col3:
        st.metric("Variedades Únicas", f"{df_filtrado['variety'].nunique()}")
    with col4:
        st.metric("Descrições Únicas", f"{df_filtrado['description'].nunique()}")
    
    # Mostrar dados filtrados
    st.write(f"**Dados filtrados:** {len(df_filtrado):,} vinhos")
    
    if len(df_filtrado) == 0:
        st.warning("Nenhum vinho encontrado com os filtros aplicados. Tente ajustar os filtros.")
        return

if __name__ == "__main__":
    main()