# Análise Preditiva do Tráfego Aéreo Brasileiro

Pipeline de ciência de dados que coleta, limpa e analisa dados públicos do
transporte aéreo brasileiro. O projeto combina estatística, machine learning e
visualização geográfica para explorar passageiros, cargas, aeroportos e a
natureza das operações aéreas.

## O que o projeto faz

1. Consulta a API de dados abertos e baixa a base histórica da ANAC.
2. Extrai e normaliza o CSV, tratando diferenças entre versões das colunas.
3. Exporta os registros processados para JSON.
4. Opcionalmente persiste o conjunto em MySQL com SQLAlchemy.
5. Cruza aeroportos com coordenadas geográficas públicas.
6. Treina e avalia modelos de classificação e regressão.
7. Gera mapas e gráficos para apoiar a interpretação dos resultados.

## Modelos e métricas

- k-NN para classificação de operações
- Regressão linear simples e múltipla
- Ridge e Lasso
- Regressão logística
- Acurácia, precisão, recall e F1-score
- MAE, MSE, RMSE e R²

## Tecnologias

- Python
- pandas e NumPy
- scikit-learn e statsmodels
- Matplotlib e Seaborn
- Folium
- SQLAlchemy e MySQL

## Como executar

### Pré-requisitos

- Python 3.10 ou superior
- MySQL, apenas se desejar usar a etapa de persistência

Crie um ambiente virtual e instale as dependências:

```bash
git clone https://github.com/guilhermelerner/Analise-Preditiva-do-Trafego-Aereo-Brasileiro.git
cd Analise-Preditiva-do-Trafego-Aereo-Brasileiro
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy requests matplotlib seaborn folium python-dotenv \
  statsmodels sqlalchemy pymysql scikit-learn
```

No Windows, ative o ambiente com `.venv\\Scripts\\activate`.

Crie um arquivo `.env` para a conexão opcional com o banco:

```dotenv
DB_HOST=localhost
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
DB_NAME=seu_banco
```

Execute:

```bash
python FinalBoss.py
```

## Fontes de dados

- Dados estatísticos do transporte aéreo disponibilizados pela ANAC
- Coordenadas de aeroportos do projeto OurAirports

## Observações

O download pode levar alguns minutos devido ao tamanho da base. Arquivos de
dados já presentes no diretório são reutilizados para evitar downloads
desnecessários. As métricas obtidas dependem da versão e da qualidade da base.
