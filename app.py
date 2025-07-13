import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Visão Geral", page_icon="🍷", layout="wide")

st.title('🍷 Visão Geral')

# Carregar dados principais
try:
    df_desc = pd.read_parquet('data/processed/winemag.parquet')
except Exception as e:
    st.error(f'Erro ao carregar winemag.parquet: {e}')
    df_desc = None
try:
    df_qual = pd.read_parquet('data/processed/wine-quality-combined.parquet')
except Exception as e:
    st.error(f'Erro ao carregar wine-quality-combined.parquet: {e}')
    df_qual = None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total de Vinhos (Descrição)', f"{len(df_desc) if df_desc is not None else '-'}")
with col2:
    st.metric('Total de Vinhos (Qualidade)', f"{len(df_qual) if df_qual is not None else '-'}")
with col3:
    st.metric('Países Únicos', f"{df_desc['country'].nunique() if df_desc is not None else '-'}")
with col4:
    st.metric('Variedades Únicas', f"{df_desc['variety'].nunique() if df_desc is not None else '-'}")

st.markdown('---')

if df_qual is not None:
    st.subheader('Distribuição de Qualidade dos Vinhos')
    fig1 = px.histogram(df_qual, x='quality', color='type' if 'type' in df_qual.columns else None, barmode='group',
                        labels={'quality': 'Qualidade', 'type': 'Tipo de Vinho'},
                        title='Distribuição de Qualidade por Tipo')
    st.plotly_chart(fig1, use_container_width=True)
    st.metric('Proporção de Vinhos Bons', f"{100 * (df_qual['quality'] >= 6).mean():.1f}%")

if df_desc is not None:
    st.subheader('Top 10 Países por Número de Vinhos (Descrição)')
    top_paises = df_desc['country'].value_counts().head(10)
    fig2 = px.bar(x=top_paises.index, y=top_paises.values,
                  labels={'x': 'País', 'y': 'Quantidade'},
                  title='Top 10 Países')
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader('Top 10 Variedades (Descrição)')
    top_variedades = df_desc['variety'].value_counts().head(10)
    fig3 = px.bar(x=top_variedades.index, y=top_variedades.values,
                  labels={'x': 'Variedade', 'y': 'Quantidade'},
                  title='Top 10 Variedades')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('---')
st.info('Acesse as páginas do menu lateral para análises detalhadas por descrição, qualidade e predição.')