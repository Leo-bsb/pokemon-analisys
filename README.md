<div align="center">

# 🎮 Pokémon Battle Prediction

### Pipeline de Dados e Machine Learning — do ETL à aplicação interativa

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458?style=flat&logo=pandas&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-1.1+-FFF000?style=flat&logo=duckdb&logoColor=black)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-concluído-brightgreen?style=flat)
![License](https://img.shields.io/badge/licença-portfólio-lightgrey?style=flat)

</div>

---

> **Os atributos estruturais de um Pokémon são suficientes para prever quem vence uma batalha?**
> Esse projeto tenta responder essa pergunta com um pipeline completo — da extração autenticada dos dados até uma aplicação interativa de predição.

---

## Sumário

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Arquitetura do Projeto](#2-arquitetura-do-projeto)
3. [Pipeline de Dados — ETL](#3-pipeline-de-dados--etl)
4. [Qualidade e Consistência dos Dados](#4-qualidade-e-consistência-dos-dados)
5. [Análise Exploratória (EDA)](#5-análise-exploratória-eda)
6. [Modelagem Preditiva](#6-modelagem-preditiva)
7. [Resultados e Conclusões](#7-resultados-e-conclusões)
8. [Tecnologias Utilizadas](#8-tecnologias-utilizadas)
9. [Como Executar](#9-como-executar)

---

## 1. Visão Geral do Projeto

### O problema

Batalhas Pokémon costumam ser analisadas pelo viés da estratégia — tipagem, moveset, itens. Mas olhando os dados de outra perspectiva, surge uma pergunta mais direta: **até que ponto os atributos estruturais de um Pokémon (Speed, Attack, Defense, HP) já carregam informação suficiente para prever quem vence?** E qual desses fatores pesa mais?

### Objetivo analítico

Construir um pipeline de ponta a ponta capaz de:

- Extrair dados brutos de uma API autenticada com JWT
- Limpar, transformar e armazenar os dados em formato adequado para análise
- Investigar hipóteses sobre quais atributos mais influenciam o resultado das batalhas
- Treinar e comparar modelos supervisionados de classificação
- Entregar insights interpretáveis, sustentados pelos dados e pelos modelos

### Pergunta central

> *"Atributos estruturais como Speed, Attack e Defense conseguem prever o resultado de batalhas com boa precisão — e qual atributo tem maior peso nessa previsão?"*

### Resumo executivo

O projeto partiu de **46.125 registros brutos de batalha** e **799 Pokémon únicos**, após limpeza que removeu ~4% de duplicatas e filtrou batalhas com dados incompletos. Em seguida, foi construído um dataset estruturado e consistente para modelagem.

A análise exploratória indicou que **Speed é o fator mais associado ao resultado**: 92,4% das vitórias ocorrem quando o Pokémon é o mais rápido (correlação de 0,677). A vantagem de tipo também aparece nos dados, mas com impacto bem menor — cerca de 2 pontos percentuais acima do baseline neutro.

Dois modelos foram treinados e comparados:

| Modelo | Acurácia | AUC-ROC |
|---|---|---|
| Regressão Logística | ~84% | ~0,88 |
| CatBoost | ~88% | ~0,94 |

---

## 2. Arquitetura do Projeto

### Estrutura de pastas

```
POKEMON/
├── catboost_info/              # Logs e artefatos temporários do CatBoost
│
├── data/
│   ├── database/
│   │   └── pokemon.duckdb      # Banco analítico DuckDB
│   ├── models/
│   │   ├── catboost_model.pkl
│   │   └── logistic_regression_model.pkl
│   ├── processed/
│   │   ├── combats_processed.parquet
│   │   └── pokemon_processed.parquet
│   └── raw/
│       ├── combats.json        # Dados brutos de batalhas (API)
│       └── pokemon.json        # Atributos brutos dos Pokémon (API)
│
├── src/
│   ├── analysis/
│   │   └── analysis.py         # EDA: distribuições, correlações, hipóteses
│   ├── etl/
│   │   ├── extract.py          # Extração autenticada via API
│   │   ├── transform.py        # Limpeza e feature engineering
│   │   └── load.py             # Escrita em Parquet + carga no DuckDB
│   ├── models/
│   │   ├── catboost_model.py
│   │   ├── logistic_regression.py
│   │   ├── model_comparison.py
│   │   ├── model_utils.py
│   │   └── MODELS_README.md
│   ├── utils/
│   │   ├── api.py              # Autenticação JWT + cliente HTTP
│   │   └── config.py           # Gerenciamento de configs e caminhos
│   ├── __init__.py
│   ├── app.py                  # Aplicação Streamlit
│   └── main.py                 # Orquestrador do pipeline
│
├── .env                        # Variáveis de ambiente (não versionado)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### Fluxo do pipeline

```
[API REST com autenticação JWT]
            │
            ▼
      [extract.py]  ──►  data/raw/  (pokemon.json, combats.json)
            │
            ▼
     [transform.py] ──►  Limpeza, feature engineering, deduplicação
            │
            ▼
       [load.py]    ──►  data/processed/  (.parquet)
                    ──►  data/database/   (pokemon.duckdb)
            │
            ▼
     [analysis.py]  ──►  EDA: distribuições, correlações, hipóteses
            │
            ▼
[model_comparison.py] ──►  Treinamento: Regressão Logística + CatBoost
            │
            ▼
      [data/models/] ──►  Modelos serializados (.pkl)
            │
            ▼
        [app.py]     ──►  Aplicação interativa (Streamlit)
```

### Notas sobre as escolhas técnicas

**Parquet** foi adotado como formato intermediário por preservar tipos nativos (booleanos, listas, estruturas aninhadas) e oferecer compressão eficiente — eliminando a necessidade de reprocessar os JSONs brutos a cada execução. A integração com DuckDB e Pandas é direta, sem declaração de schema manual.

**DuckDB** entra como camada analítica, permitindo rodar SQL diretamente sobre Parquet e DataFrames em memória. Isso facilita validações ao longo do pipeline e deixa a arquitetura próxima do que se veria em um data warehouse real (BigQuery, Redshift), sem depender de infraestrutura externa.

**Serialização em `.pkl`** desacopla treino de inferência. A aplicação Streamlit carrega o modelo sob demanda, sem precisar reexecutar o pipeline — o que torna o ciclo de atualização mais simples de gerenciar.

---

## 3. Pipeline de Dados — ETL

### 3.1 Extração — `extract.py`

Os dados vêm de uma API REST autenticada com JWT. O `api.py` gerencia o token e o injeta como `Authorization: Bearer <token>` em todas as requisições. Dois endpoints são consumidos:

- `/pokemon` → array JSON com os Pokémon e seus atributos aninhados
- `/combats` → array JSON com os registros de batalha (`first_pokemon`, `second_pokemon`, `winner`)

Os dados brutos são salvos em `data/raw/` antes de qualquer transformação. Isso garante que o pipeline pode ser reexecutado sem bater na API novamente — e que os dados originais ficam intactos caso algum passo seguinte precise ser corrigido.

### 3.2 Transformação — `transform.py`

**Normalização dos atributos dos Pokémon:**
- Achatamento do objeto `attributes` aninhado em estrutura tabular plana
- Separação do campo `types` (ex: `"Grass/Poison"`) nas colunas `type1` e `type2`
- Derivação das listas de `advantage` e `weakness` com base nas tabelas de interação de tipos
- Conversão de `generation` para inteiro e `legendary` para booleano

**Enriquecimento dos combates:**
- Join de cada batalha com os atributos completos dos dois Pokémon participantes
- Criação de features diferenciais: `stats_diff`, `speed_diff`, `attack_diff`, `defense_diff`
- Cálculo de `total_stats` para cada Pokémon como proxy de força geral
- Derivação de `type_advantage` a partir da tabela de interação de tipos
- Resolução de `winner_name` a partir do ID do vencedor

**Deduplicação:**

Cerca de 4% dos registros apareciam duplicados — batalhas com os mesmos dois Pokémon em posições invertidas, mas sempre com o mesmo vencedor. Esses registros foram removidos por uma chave canônica de par (`min(id1, id2)`, `max(id1, id2)`). Mantê-los inflaria artificialmente o dataset e geraria vetores de features simétricos, o que poderia introduzir viés nos modelos.

**Filtragem de Pokémon sem atributos:**

Cerca de 100 IDs presentes nos combates não tinham entrada correspondente no endpoint de Pokémon. Batalhas envolvendo esses IDs foram excluídas — sem atributos estruturais, não é possível construir as features diferenciais que alimentam os modelos.

### 3.3 Carga — `load.py`

Após a transformação, os DataFrames são persistidos em dois formatos:

- **Parquet** em `data/processed/` para reuso rápido nas etapas seguintes
- **DuckDB** em `data/database/pokemon.duckdb` para validação via SQL e consumo pelos scripts de análise e modelagem

---

## 4. Qualidade e Consistência dos Dados

### Estatísticas iniciais

| Métrica | Valor |
|---|---|
| Registros de batalha (bruto) | 46.125 |
| Pokémon únicos | 799 |
| Tipos primários distintos | 25 |
| Pokémon lendários | 65 |

### Problemas identificados e resoluções

**Duplicatas posicionais (~4% dos combates)**

Registros com os Pokémon em posições trocadas, mas com o mesmo vencedor — descartados por chave canônica de par.

**Pokémon sem atributos registrados (~100 IDs)**

IDs referenciados nos combates sem entrada correspondente no endpoint de Pokémon — batalhas com esses IDs foram filtradas do dataset.

### Dataset final

| Métrica | Valor |
|---|---|
| Batalhas após limpeza | ~42.900 (est.) |
| Colunas no dataset de combates | 34 |
| Valores ausentes | Nenhum |

A limpeza foi conservadora: preferiu-se descartar do que imputar, preservando a integridade das features diferenciais.

---

## 5. Análise Exploratória (EDA)

### Distribuição dos atributos

HP e Defesa apresentam distribuições concentradas em valores baixos com caudas longas à direita. Speed tem uma distribuição mais espalhada e heterogênea — o que já sugere, antes de qualquer modelo, que pode ser uma boa variável discriminativa.

Pokémon lendários se concentram no topo de quase todos os atributos, mas sua raridade (65 em 799) limita o peso que têm no conjunto total.

### Hipótese 1 — Speed influencia quem vence?

| Métrica | Valor |
|---|---|
| Taxa de vitória sendo mais rápido | **92,4%** |
| Taxa de vitória sendo mais lento | 4,3% |
| Correlação (speed × vitória) | **0,677** |
| Lift acima do baseline (50%) | +42,4 pp |

Speed tem a maior correlação individual com o resultado da batalha entre todos os atributos analisados. Uma taxa de 92,4% de vitória para o Pokémon mais rápido é expressiva — sugere que velocidade carrega o principal sinal disponível para predição nesse dataset.

### Hipótese 2 — Vantagem de tipo faz diferença?

| Cenário | Taxa de vitória |
|---|---|
| Vantagem de tipo | 52,4% |
| Neutralidade | 46,9% |
| Desvantagem de tipo | 42,6% |

A vantagem de tipo é estatisticamente detectável, mas o efeito é bem mais modesto que Speed. A diferença entre ter vantagem e estar em desvantagem é de ~10 pp — comparado com a separação de ~88 pp observada para speed, tipo ocupa um papel secundário na maioria dos casos.

---

## 6. Modelagem Preditiva

A tarefa foi tratada como **classificação binária**: dado o par de Pokémon com seus atributos, prever qual dos dois vence.

**Variável alvo:** `winner` — 1 se o `first_pokemon` vence, 0 caso contrário.

A estratégia de features foi baseada em **diferenças e valores absolutos** dos atributos de ambos os Pokémon, sem qualquer informação baseada em ID.

---

### Modelo 1 — Regressão Logística

Objetivo: estabelecer uma baseline linear interpretável que quantifique o peso relativo de cada feature.

**Features (5):** `stats_diff`, `speed_diff`, `attack_diff`, `defense_diff`, `type_advantage`

| Métrica | Valor |
|---|---|
| Acurácia | ~84% |
| AUC-ROC | ~0,88 |

O coeficiente de `speed_diff` é o mais alto entre todas as features, confirmando quantitativamente o que a EDA já indicava. O fato de um modelo linear chegar a ~84% com apenas 5 features já mostra que o espaço de atributos estruturais é razoavelmente separável — e a regressão logística deixa isso explícito de forma transparente.

---

### Modelo 2 — CatBoost

O CatBoost foi escolhido principalmente pelo **suporte nativo a variáveis categóricas**: tipos Pokémon e geração são categóricos, e o CatBoost lida com o encoding internamente, sem precisar de one-hot ou label encoding manual. Além disso, o ensemble de árvores consegue capturar interações entre features que um modelo linear não representa.

**Features (13):** as 5 da regressão logística, mais `type1_1`, `type2_1`, `type1_2`, `type2_2`, `generation_1`, `generation_2`, `legendary_1`, `legendary_2`, `sum_attributes`

| Métrica | Valor |
|---|---|
| Acurácia | ~88% |
| AUC-ROC | ~0,94 |
| Ganho sobre Regressão Logística | +4 pp / +0,06 AUC |

Speed continua como a feature de maior importância. Tipo e geração contribuem de forma incremental — especialmente nos casos-limite onde a diferença de speed entre os Pokémon é pequena.

**Comparativo:**

| Modelo | Acurácia | AUC-ROC | Features | Interpretabilidade |
|---|---|---|---|---|
| Regressão Logística | ~84% | ~0,88 | 5 | Alta |
| CatBoost | ~88% | ~0,94 | 13 | Média |

---

## 7. Resultados e Conclusões

### Speed como fator mais associado à vitória

Speed diferencial foi consistentemente a variável mais preditiva — na correlação da EDA, nos coeficientes da regressão e na importância de features do CatBoost. Com 92,4% de taxa de vitória para o Pokémon mais rápido e correlação de 0,677, velocidade carrega a maior parte do sinal disponível nesse dataset.

### Vantagem de tipo: presente, mas secundária

Tipo aparece com contribuição positiva em todos os modelos, mas bem menor que Speed. A diferença de ~10 pp entre vantagem e desvantagem indica que tipo influencia as probabilidades — sem ser preponderante. Stats sólidos, especialmente velocidade, tendem a compensar desvantagens táticas com frequência.

### Poder preditivo alcançado

| Modelo | AUC-ROC |
|---|---|
| Regressão Logística | 0,88 |
| CatBoost | **0,94** |

Um AUC-ROC de 0,94 significa que o modelo ordena corretamente a probabilidade de vitória entre dois Pokémon aleatórios em ~94% das vezes — usando apenas atributos estruturais, sem nenhuma informação de estratégia ou moveset.

### Principais achados

- Atributos estruturais carregam bastante sinal: não é preciso conhecer moveset ou estratégia para chegar a ~88% de acurácia.
- Speed é a variável de maior alavancagem preditiva — maximizar o diferencial de velocidade provavelmente tem mais retorno do que explorar vantagens de tipo.
- Tipo importa, mas menos do que a intuição costuma sugerir. Stats superiores tendem a compensar desvantagens táticas em boa parte dos casos.

---

## 8. Tecnologias Utilizadas

| Tecnologia | Versão | Uso no projeto |
|---|---|---|
| Python | ≥3.10 | Runtime principal |
| Pandas | ≥2.2.3 | Manipulação e transformação de dados |
| Polars | ≥1.21.0 | Operações de alto desempenho em DataFrames |
| NumPy | ≥2.2.0 | Operações numéricas |
| PyArrow | ≥19.0.0 | Backend de leitura/escrita Parquet |
| SciPy | ≥1.15.0 | Análise estatística |
| DuckDB | ≥1.1.3 | Banco analítico SQL em processo |
| Scikit-learn | ≥1.6.1 | Regressão Logística, métricas, pré-processamento |
| CatBoost | ≥1.2.7 | Gradient boosting com suporte nativo a categóricas |
| Plotly | ≥6.0.0 | Visualizações interativas |
| Streamlit | ≥1.41.0 | Interface da aplicação de predição |
| python-dotenv | ≥1.0.1 | Gerenciamento de variáveis de ambiente |
| Requests | ≥2.32.3 | Cliente HTTP para extração via API |

---

## 9. Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- Credenciais de acesso à API

### Configurando o ambiente

```bash
git clone https://github.com/<seu-usuario>/pokemon-battle-prediction.git
cd pokemon-battle-prediction

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais:

```dotenv
# Autenticação da API
API_BASE_URL=https://<api-host>/api
API_USERNAME=seu_usuario
API_PASSWORD=sua_senha

# Caminhos de dados (opcional — usa os padrões relativos do projeto)
RAW_DATA_PATH=data/raw
PROCESSED_DATA_PATH=data/processed
DATABASE_PATH=data/database
MODELS_PATH=data/models
```

> **Atenção:** o `.env` está no `.gitignore`. Nunca suba credenciais para o repositório.

### Instalando as dependências

```bash
pip install -r requirements.txt
```

### Executando o pipeline completo

```bash
python main.py
```

Etapas executadas em sequência:
1. Extração dos dados brutos da API → `data/raw/`
2. Transformação, limpeza e feature engineering → `data/processed/`
3. Carga no DuckDB → `data/database/pokemon.duckdb`
4. Análise exploratória com geração de visualizações
5. Treinamento e comparação dos modelos → `data/models/`

Saída esperada ao final:

```
================================================================================
PIPELINE FINALIZADO COM SUCESSO - EXECUTE 'STREAMLIT RUN APP.PY'
================================================================================
```

### Iniciando a aplicação

```bash
streamlit run src/app.py
```

---

<div align="center">

Desenvolvido com Python · DuckDB · CatBoost · Streamlit

</div>
