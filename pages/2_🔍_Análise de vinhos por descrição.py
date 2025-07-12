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

def plot_analise_descricoes(df):
    """Plota análise das descrições"""
    st.subheader("📝 Análise das Descrições")
    
    # Preparar texto para análise
    descricoes = ' '.join(df['description'].dropna().astype(str))
    
    # Limpar texto
    descricoes_limpas = re.sub(r'[^\w\s]', '', descricoes.lower())
    
    # Criar wordcloud
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='Reds',
            max_words=100
        ).generate(descricoes_limpas)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Palavras Mais Frequentes nas Descrições', fontsize=16, pad=20)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Não foi possível gerar o wordcloud: {e}")
    
    # Análise de palavras por país
    st.subheader("🔍 Palavras por País")
    
    # Top 5 países
    top_paises = df['country'].value_counts().head(5).index
    
    pais_palavras = {}
    for pais in top_paises:
        descricoes_pais = ' '.join(
            df[df['country'] == pais]['description'].dropna().astype(str)
        )
        palavras = re.findall(r'\b\w+\b', descricoes_pais.lower())
        # Remover palavras comuns
        stop_words = {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'that', 'with', 'for', 'as', 'on', 'be', 'at', 'this', 'by', 'i', 'you', 'have', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'}
        palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 2]
        pais_palavras[pais] = Counter(palavras_filtradas).most_common(10)
    
    # Mostrar palavras por país
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Palavras mais frequentes por país:**")
        for pais, palavras in pais_palavras.items():
            st.write(f"**{pais}:**")
            for palavra, freq in palavras[:5]:
                st.write(f"  - {palavra}: {freq}")
            st.write("")
    
    with col2:
        # Gráfico de barras das palavras mais comuns do país líder
        if pais_palavras:
            pais_lider = list(pais_palavras.keys())[0]
            palavras, freqs = zip(*pais_palavras[pais_lider][:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(palavras)), freqs, color=CORES_VINHO['tinto'], alpha=0.7)
            ax.set_xlabel('Palavras')
            ax.set_ylabel('Frequência')
            ax.set_title(f'Top 10 Palavras - {pais_lider}')
            ax.set_xticks(range(len(palavras)))
            ax.set_xticklabels(palavras, rotation=45, ha='right')
            
            # Adicionar valores nas barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)

def plot_analise_variedades(df):
    """Plota análise das variedades de vinho"""
    st.subheader("🍇 Análise das Variedades de Vinho")
    
    # Top variedades
    top_variedades = df['variety'].value_counts().head(20)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras das top variedades
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_variedades)))
    bars = ax1.bar(range(len(top_variedades)), top_variedades.values, color=colors)
    ax1.set_xlabel('Variedade')
    ax1.set_ylabel('Quantidade de Vinhos')
    ax1.set_title('Top 20 Variedades Mais Comuns')
    ax1.set_xticks(range(len(top_variedades)))
    ax1.set_xticklabels(top_variedades.index, rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # Distribuição geográfica das top variedades
    top_5_variedades = top_variedades.head(5).index
    df_top_variedades = df[df['variety'].isin(top_5_variedades)]
    
    # Contar países por variedade
    pais_por_variedade = df_top_variedades.groupby(['variety', 'country']).size().reset_index(name='count')
    pais_por_variedade = pais_por_variedade.sort_values('count', ascending=False)
    
    # Mostrar top países para cada variedade
    for variedade in top_5_variedades:
        top_paises = pais_por_variedade[pais_por_variedade['variety'] == variedade].head(5)
        if not top_paises.empty:
            ax2.bar(range(len(top_paises)), top_paises['count'], 
                   label=variedade, alpha=0.7)
    
    ax2.set_xlabel('Países')
    ax2.set_ylabel('Quantidade')
    ax2.set_title('Distribuição Geográfica das Top 5 Variedades')
    ax2.legend()
    ax2.set_xticks(range(len(top_paises)))
    ax2.set_xticklabels(top_paises['country'], rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Estatísticas das variedades
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Variedades", f"{df['variety'].nunique()}")
    with col2:
        st.metric("Variedade Mais Comum", f"{top_variedades.index[0]}")
    with col3:
        st.metric("Quantidade da Mais Comum", f"{top_variedades.iloc[0]:,}")

def plot_analise_variedades_descricoes(df):
    """Plota análise de descrições por variedade"""
    st.subheader("🍷 Descrições por Variedade")
    
    # Top 5 variedades
    top_variedades = df['variety'].value_counts().head(5).index
    
    variedade_palavras = {}
    for variedade in top_variedades:
        descricoes_variedade = ' '.join(
            df[df['variety'] == variedade]['description'].dropna().astype(str)
        )
        palavras = re.findall(r'\b\w+\b', descricoes_variedade.lower())
        # Remover palavras comuns
        stop_words = {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'that', 'with', 'for', 'as', 'on', 'be', 'at', 'this', 'by', 'i', 'you', 'have', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'}
        palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 2]
        variedade_palavras[variedade] = Counter(palavras_filtradas).most_common(10)
    
    # Mostrar palavras por variedade
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Palavras mais frequentes por variedade:**")
        for variedade, palavras in variedade_palavras.items():
            st.write(f"**{variedade}:**")
            for palavra, freq in palavras[:5]:
                st.write(f"  - {palavra}: {freq}")
            st.write("")
    
    with col2:
        # Gráfico de barras das palavras mais comuns da variedade líder
        if variedade_palavras:
            variedade_lider = list(variedade_palavras.keys())[0]
            palavras, freqs = zip(*variedade_palavras[variedade_lider][:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(palavras)), freqs, color=CORES_VINHO['branco'], alpha=0.7)
            ax.set_xlabel('Palavras')
            ax.set_ylabel('Frequência')
            ax.set_title(f'Top 10 Palavras - {variedade_lider}')
            ax.set_xticks(range(len(palavras)))
            ax.set_xticklabels(palavras, rotation=45, ha='right')
            
            # Adicionar valores nas barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)

def plot_analise_paises(df):
    """Plota análise por países"""
    st.subheader("🌍 Análise por Países")
    
    # Top países
    top_paises = df['country'].value_counts().head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras dos top países
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_paises)))
    bars = ax1.bar(range(len(top_paises)), top_paises.values, color=colors)
    ax1.set_xlabel('País')
    ax1.set_ylabel('Quantidade de Vinhos')
    ax1.set_title('Top 15 Países com Mais Vinhos')
    ax1.set_xticks(range(len(top_paises)))
    ax1.set_xticklabels(top_paises.index, rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # Variedades mais comuns por país (top 5 países)
    top_5_paises = top_paises.head(5).index
    df_top_paises = df[df['country'].isin(top_5_paises)]
    
    # Contar variedades por país
    variedade_por_pais = df_top_paises.groupby(['country', 'variety']).size().reset_index(name='count')
    variedade_por_pais = variedade_por_pais.sort_values('count', ascending=False)
    
    # Mostrar top variedades para cada país
    for pais in top_5_paises:
        top_variedades_pais = variedade_por_pais[variedade_por_pais['country'] == pais].head(5)
        if not top_variedades_pais.empty:
            ax2.bar(range(len(top_variedades_pais)), top_variedades_pais['count'], 
                   label=pais, alpha=0.7)
    
    ax2.set_xlabel('Variedades')
    ax2.set_ylabel('Quantidade')
    ax2.set_title('Variedades Mais Comuns nos Top 5 Países')
    ax2.legend()
    ax2.set_xticks(range(len(top_variedades_pais)))
    ax2.set_xticklabels(top_variedades_pais['variety'], rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Estatísticas dos países
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Países", f"{df['country'].nunique()}")
    with col2:
        st.metric("País com Mais Vinhos", f"{top_paises.index[0]}")
    with col3:
        st.metric("Quantidade do País Líder", f"{top_paises.iloc[0]:,}")

def plot_analise_regioes(df):
    """Plota análise por regiões"""
    if 'region_1' in df.columns:
        st.subheader("🗺️ Análise por Regiões")
        
        # Top regiões
        top_regioes = df['region_1'].value_counts().head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras das top regiões
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_regioes)))
        bars = ax1.bar(range(len(top_regioes)), top_regioes.values, color=colors)
        ax1.set_xlabel('Região')
        ax1.set_ylabel('Quantidade de Vinhos')
        ax1.set_title('Top 15 Regiões com Mais Vinhos')
        ax1.set_xticks(range(len(top_regioes)))
        ax1.set_xticklabels(top_regioes.index, rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        # Variedades por região (top 5 regiões)
        top_5_regioes = top_regioes.head(5).index
        df_top_regioes = df[df['region_1'].isin(top_5_regioes)]
        
        # Contar variedades por região
        variedade_por_regiao = df_top_regioes.groupby(['region_1', 'variety']).size().reset_index(name='count')
        variedade_por_regiao = variedade_por_regiao.sort_values('count', ascending=False)
        
        # Mostrar top variedades para cada região
        for regiao in top_5_regioes:
            top_variedades_regiao = variedade_por_regiao[variedade_por_regiao['region_1'] == regiao].head(3)
            if not top_variedades_regiao.empty:
                ax2.bar(range(len(top_variedades_regiao)), top_variedades_regiao['count'], 
                       label=regiao, alpha=0.7)
        
        ax2.set_xlabel('Variedades')
        ax2.set_ylabel('Quantidade')
        ax2.set_title('Variedades Mais Comuns nas Top 5 Regiões')
        ax2.legend()
        ax2.set_xticks(range(len(top_variedades_regiao)))
        ax2.set_xticklabels(top_variedades_regiao['variety'], rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estatísticas das regiões
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Regiões", f"{df['region_1'].nunique()}")
        with col2:
            st.metric("Região com Mais Vinhos", f"{top_regioes.index[0]}")
        with col3:
            st.metric("Quantidade da Região Líder", f"{top_regioes.iloc[0]:,}")




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
    
    # Análises
    plot_analise_variedades(df_filtrado)
    plot_analise_descricoes(df_filtrado)
    plot_analise_variedades_descricoes(df_filtrado)
    plot_analise_paises(df_filtrado)
    plot_analise_regioes(df_filtrado)

if __name__ == "__main__":
    main()