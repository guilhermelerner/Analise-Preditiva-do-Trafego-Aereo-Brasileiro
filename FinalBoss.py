import pandas as pd
import requests
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import urllib.parse
import folium
from folium.plugins import MarkerCluster, HeatMap 
from dotenv import load_dotenv
import statsmodels.api as sm
import matplotlib.ticker as ticker # <-- Para formatar os números dos gráficos

from sqlalchemy import create_engine

# Bibliotecas de Machine Learning (Scikit-Learn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression # <-- LogisticRegression adicionada
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix # <-- confusion_matrix adicionada

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not DB_PASSWORD:
    print("❌ ERRO: Senha do banco não encontrada. Verifique se o arquivo .env existe e está preenchido corretamente.")
    exit()

DATASET_ID = "voos-e-operacoes-aereas-dados-estatisticos-do-transporte-aereo"
API_URL = f"https://dados.gov.br/api/3/action/package_show?id={DATASET_ID}"
ARQUIVO_ZIP = "Base_10_anos.zip"
ARQUIVO_CSV_EXTRAIDO = "dados_anac_raw.csv"
ARQUIVO_JSON = "dados_anac_processados.json"
URL_AEROPORTOS = "https://davidmegginson.github.io/ourairports-data/airports.csv"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def buscar_url_via_api():
    print("--- ETAPA 0: CONSULTA À API ---")
    url_seguranca = "https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/dados-estatisticos/arquivos/Base_10_anos.zip"
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=20)
        if response.status_code == 200:
            recursos = response.json().get('result', {}).get('resources', [])
            for res in recursos:
                if "10 anos" in res.get('name', '') or "Base_10_anos" in res.get('url', ''):
                    return res['url']
        return url_seguranca
    except Exception:
        return url_seguranca

def extrair_e_baixar(url):
    print("\n--- ETAPA 1: EXTRAÇÃO (DOWNLOAD E ZIP) ---")
    if not os.path.exists(ARQUIVO_ZIP):
        print("A iniciar o download da base de dados da ANAC... (O servidor é lento, aguarde)")
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(ARQUIVO_ZIP, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
            print("✔ Download concluído com sucesso!")
        except requests.exceptions.RequestException as e:
            print(f"\n O servidor do governo derrubou a conexão: {e}")
            exit()
    else:
        print("✔ O ficheiro ZIP já existe localmente. A saltar o download.")
    
    print("A extrair o ficheiro CSV do ZIP...")
    try:
        with zipfile.ZipFile(ARQUIVO_ZIP, 'r') as zip_ref:
            csv_interno = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')][0]
            zip_ref.extract(csv_interno)
            if os.path.exists(ARQUIVO_CSV_EXTRAIDO): os.remove(ARQUIVO_CSV_EXTRAIDO)
            os.rename(csv_interno, ARQUIVO_CSV_EXTRAIDO)
        print("✔ Extração concluída!")
    except zipfile.BadZipFile:
        print(f"\n Erro: Arquivo corrompido. Exclua o ZIP e tente novamente.")
        os.remove(ARQUIVO_ZIP)
        exit()

def transformar_dados():
    print("\n--- ETAPA 2: TRANSFORMAÇÃO E LIMPEZA ---")
    df = pd.read_csv(ARQUIVO_CSV_EXTRAIDO, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
    
    if df.shape[1] < 2:
        df = pd.read_csv(ARQUIVO_CSV_EXTRAIDO, sep=',', encoding='latin1', on_bad_lines='skip')

    df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()

    mapeamento = {
        'PASSAGEIROS_PAGOS': ['PASSAGEIROS_PAGOS', 'PASSAGEIROS_PAGOS_TOTAL', 'PAX_PAGOS'],
        'CARGA_PAGA_KG': ['CARGA_PAGA_KG', 'CARGA_PAGA_(KG)', 'CARGA_PAGA', 'CARGA_PAGA_TOTAL'],
        'ANO': ['ANO', 'ANO_REFERENCIA'],
        'MES': ['MES', 'MÊS', 'MES_REFERENCIA'],
        'ORIGEM': ['AEROPORTO_DE_ORIGEM_SIGLA', 'AERODROMO_DE_ORIGEM_SIGLA', 'AEROPORTO_ORIGEM_SIGLA', 'ORIGEM_SIGLA', 'AEROPORTO_DE_ORIGEM_(SIGLA)', 'AEROPORTO_DE_ORIGEM']
    }

    def encontrar_coluna(lista_possibilidades):
        for p in lista_possibilidades:
            if p in df.columns: return p
        return None

    col_ano = encontrar_coluna(mapeamento['ANO'])
    col_pax = encontrar_coluna(mapeamento['PASSAGEIROS_PAGOS'])
    col_carga = encontrar_coluna(mapeamento['CARGA_PAGA_KG'])
    col_mes = encontrar_coluna(mapeamento['MES'])
    col_origem = encontrar_coluna(mapeamento['ORIGEM'])

    if not col_origem:
        print("\n ERRO FATAL: Não encontrei a coluna do Aeroporto de Origem.")
        exit()

    df[col_ano] = pd.to_numeric(df[col_ano], errors='coerce')
    df = df[df[col_ano] >= 2016].copy()
    
    df[col_pax] = pd.to_numeric(df[col_pax], errors='coerce').fillna(0)
    df[col_carga] = pd.to_numeric(df[col_carga], errors='coerce').fillna(0)

    df = df.rename(columns={
        col_ano: 'ANO', 
        col_pax: 'PASSAGEIROS_PAGOS', 
        col_carga: 'CARGA_PAGA_KG',
        col_mes: 'MES',
        col_origem: 'AEROPORTO_ORIGEM'
    })
    
    colunas_final = ['ANO', 'MES', 'AEROPORTO_ORIGEM', 'EMPRESA_SIGLA', 'PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG', 'NATUREZA']
    df_final = df[[c for c in colunas_final if c in df.columns]]
    
    df_final.to_json(ARQUIVO_JSON, orient='records', indent=4)
    print(f"✔ Transformação concluída. {len(df_final)} registros processados.")
    return df_final

def salvar_banco_mysql(df):
    print("\n--- ETAPA 3: BANCO DE DADOS MYSQL ---")
    try:
        senha_tratada = urllib.parse.quote_plus(DB_PASSWORD)
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{senha_tratada}@{DB_HOST}/{DB_NAME}")
        df.to_sql('transporte_aereo', engine, if_exists='replace', index=False)
        print("✔ Dados salvos com sucesso no seu MySQL local!")
    except Exception as e:
        print(f"❌ Erro ao conectar no MySQL: {e}")

def exercicio_knn_mapa(df):
    print("\n--- ETAPA 4: EXERCÍCIO PRÁTICO - kNN 4D & MAPA PROFISSIONAL ---")
    
    try:
        df_airports = pd.read_csv(URL_AEROPORTOS)
        df_airports = df_airports[['ident', 'latitude_deg', 'longitude_deg', 'name']]
    except Exception as e:
        print("Erro ao baixar coordenadas.", e)
        return

    df_ml = pd.merge(df, df_airports, left_on='AEROPORTO_ORIGEM', right_on='ident', how='inner')
    df_ml = df_ml.dropna(subset=['latitude_deg', 'longitude_deg', 'NATUREZA']).copy()
    df_aeroportos = df_ml.groupby(['AEROPORTO_ORIGEM', 'name', 'latitude_deg', 'longitude_deg', 'NATUREZA'])[['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG']].sum().reset_index()

    brazil_lat = (-34.0, 5.0)
    brazil_lon = (-74.0, -34.0)
    df_br = df_aeroportos[
        (df_aeroportos['latitude_deg'] >= brazil_lat[0]) & (df_aeroportos['latitude_deg'] <= brazil_lat[1]) &
        (df_aeroportos['longitude_deg'] >= brazil_lon[0]) & (df_aeroportos['longitude_deg'] <= brazil_lon[1])
    ].copy()

    df_br['TARGET'] = df_br['NATUREZA'].astype('category').cat.codes
    X = df_br[['longitude_deg', 'latitude_deg', 'PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG']] 
    y = df_br['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"✔ Modelo kNN treinado! Acurácia: {acuracia:.4f}")

    mapa = folium.Map(location=[-15.7801, -47.9292], zoom_start=4, tiles='CartoDB Positron')
    heat_data = [[row['latitude_deg'], row['longitude_deg'], row['PASSAGEIROS_PAGOS']] for index, row in df_br.iterrows()]
    HeatMap(heat_data, name="Densidade de Passageiros", radius=15, blur=10, max_zoom=1).add_to(mapa)
    marker_cluster = MarkerCluster(name="Classificação kNN (Aeroportos)").add_to(mapa)

    for idx, row in df_br.iterrows():
        cor = '#e74c3c' if row['NATUREZA'] == 'DOMÉSTICA' else '#2980b9'
        html_popup = f"""
        <div style='font-family: Arial; min-width: 220px;'>
            <b>Aeroporto:</b> {row['name']} ({row['AEROPORTO_ORIGEM']})<br>
            <b>Classificação:</b> <span style='color:{cor}; font-weight:bold;'>{row['NATUREZA']}</span><br>
            <b>Passageiros:</b> {int(row['PASSAGEIROS_PAGOS']):,}<br>
            <b>Carga (KG):</b> {float(row['CARGA_PAGA_KG']):,.2f}
        </div>
        """
        folium.CircleMarker(
            location=[row['latitude_deg'], row['longitude_deg']],
            radius=7, popup=folium.Popup(html_popup, max_width=300),
            color=cor, fill=True, fill_color=cor, fill_opacity=0.9
        ).add_to(marker_cluster)
        
    folium.LayerControl().add_to(mapa)
    arquivo_mapa = 'mapa_interativo_knn_V4.html'
    mapa.save(arquivo_mapa)
    print(f"✔ Gráfico 1 (kNN): Arquivo HTML '{arquivo_mapa}' gerado com sucesso!")

def projeto_final_regressao(df):
    print("\n--- ETAPA 5: PROJETO FINAL - REGRESSÃO LINEAR SIMPLES ---")
    df_agrupado = df.groupby(['ANO', 'MES'])[['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG']].sum().reset_index()
    
    X = df_agrupado[['PASSAGEIROS_PAGOS']]
    y = df_agrupado['CARGA_PAGA_KG']

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"✔ R² da Regressão Simples: {r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='steelblue', alpha=0.7)
    plt.plot(X, y_pred, color='crimson', linewidth=2)
    plt.title('Regressão Linear: Fluxo de Passageiros vs Volume de Carga')
    plt.xlabel('Passageiros Pagos')
    plt.ylabel('Carga Paga (KG)')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}".replace(",", ".")))
    plt.tight_layout()
    plt.savefig('grafico_regressao_linear.png')
    print("✔ Gráfico 2 (Linear Simples): 'grafico_regressao_linear.png' gerado com sucesso!")

def projeto_final_regressao_multipla(df):
    print("\n--- ETAPA 6: REGRESSÃO MÚLTIPLA E REGULARIZAÇÃO ---")
    df_reg = df.dropna(subset=['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG', 'NATUREZA']).copy()
    df_reg = df_reg.groupby(['AEROPORTO_ORIGEM', 'ANO', 'MES', 'NATUREZA'])[['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG']].sum().reset_index()
    
    df_reg['DUMMY_INTL'] = (df_reg['NATUREZA'] == 'INTERNACIONAL').astype(int)
    df_reg['INTERACAO_PAX_INTL'] = df_reg['PASSAGEIROS_PAGOS'] * df_reg['DUMMY_INTL']
    
    X = df_reg[['PASSAGEIROS_PAGOS', 'DUMMY_INTL', 'INTERACAO_PAX_INTL']]
    y = df_reg['CARGA_PAGA_KG']
    
    # Modelo OLS para o terminal (Tabelas)
    X_sm = sm.add_constant(X) 
    modelo_ols = sm.OLS(y, X_sm).fit()
    print("\n[Tabelas OLS geradas para análise estatística]")
    # Removi os prints longos das tabelas para o terminal não ficar poluído, mas o modelo rodou perfeitamente.
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Treinando o modelo Ridge (Regularizado) para gerar o gráfico
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    y_pred_multi = ridge.predict(X_test)
    
    # NOVO: Gerando o Gráfico da Regressão Múltipla
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_multi, color='purple', alpha=0.5)
    # Linha de referência perfeita (y = x)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', lw=2, linestyle='--')
    
    plt.title('Regressão Múltipla (Ridge): Carga Real vs. Predição do Modelo')
    plt.xlabel('Volume de Carga Real (KG)')
    plt.ylabel('Previsão do Modelo (KG)')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}".replace(",", ".")))
    plt.tight_layout()
    plt.savefig('grafico_regressao_multipla.png')
    print("✔ Gráfico 3 (Múltipla): 'grafico_regressao_multipla.png' gerado com sucesso!")

def projeto_final_regressao_logistica(df):
    print("\n--- ETAPA 7: REGRESSÃO LOGÍSTICA ---")
    
    # Prepara os dados (remover nulos)
    df_log = df.dropna(subset=['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG', 'NATUREZA']).copy()
    
    # Agrupa por voos e empresas para evitar ruídos
    df_log = df_log.groupby(['AEROPORTO_ORIGEM', 'NATUREZA'])[['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG']].sum().reset_index()

    # Criação da Variável Alvo Binária (0 = Doméstico, 1 = Internacional)
    df_log['TARGET'] = (df_log['NATUREZA'] == 'INTERNACIONAL').astype(int)
    
    X = df_log[['PASSAGEIROS_PAGOS', 'CARGA_PAGA_KG']]
    y = df_log['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Padronização (Obrigatório para Logística)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do Modelo Logístico
    modelo_log = LogisticRegression()
    modelo_log.fit(X_train_scaled, y_train)
    
    y_pred = modelo_log.predict(X_test_scaled)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"✔ R² (Acurácia) da Regressão Logística: {acuracia:.4f}")

    # NOVO: Gerando o Gráfico da Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Doméstico', 'Internacional'], 
                yticklabels=['Doméstico', 'Internacional'])
    plt.title('Regressão Logística: Matriz de Confusão')
    plt.xlabel('Classe Prevista pelo Modelo')
    plt.ylabel('Classe Real da ANAC')
    plt.tight_layout()
    plt.savefig('grafico_regressao_logistica.png')
    print("✔ Gráfico 4 (Logística): 'grafico_regressao_logistica.png' gerado com sucesso!")

def main():
    start = datetime.now()
    print(f"PIPELINE EXECUTÁVEL INICIADO EM {start.strftime('%H:%M:%S')}")
    print("="*60)
    
    url = buscar_url_via_api()
    extrair_e_baixar(url)
    dados = transformar_dados()
    salvar_banco_mysql(dados)
    
    # Chamando as 4 funções que geram os 4 gráficos exigidos:
    exercicio_knn_mapa(dados)
    projeto_final_regressao(dados)
    projeto_final_regressao_multipla(dados)
    projeto_final_regressao_logistica(dados)
    
    print("="*60)
    print(f"TRABALHO FINALIZADO COM SUCESSO! Duração: {datetime.now() - start}")

if __name__ == "__main__":
    main()