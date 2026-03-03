# Módulo de Machine Learning - Pokémon Battle Predictor

## 📁 Estrutura

```
src/models/
├── __init__.py              # Pacote principal
├── model_utils.py           # Utilitários compartilhados
├── logistic_regression.py   # Modelo de Regressão Logística
├── catboost_model.py        # Modelo CatBoost
└── model_comparison.py      # Comparação entre modelos
```

## 🎯 Modelos Implementados

### 1. Regressão Logística
**Arquivo:** `logistic_regression.py`

**Features utilizadas:**
- `stats_diff`: Diferença de status total
- `speed_diff`: Diferença de velocidade
- `attack_diff`: Diferença de ataque
- `defense_diff`: Diferença de defesa
- `vantagem_p1`: Vantagem de tipo (binária)

**Características:**
- Modelo linear simples e interpretável
- Coeficientes indicam importância relativa
- Rápido para treinar e predizer
- Baseline para comparação

### 2. CatBoost
**Arquivo:** `catboost_model.py`

**Features utilizadas:**
- Todas as features numéricas da Regressão Logística
- Features categóricas adicionais:
  - `generation_1`, `generation_2`
  - `legendary_1`, `legendary_2`
  - `type1_1`, `type1_2`
  - `type2_1`, `type2_2`

**Características:**
- Gradient boosting otimizado para categóricas
- Captura interações complexas
- Early stopping automático
- Melhor performance geral

## 🚀 Uso

### Treinamento Individual

```python
# Regressão Logística
from src.models.logistic_regression import run as train_logistic
model, metrics = train_logistic()

# CatBoost
from src.models.catboost_model import run as train_catboost
model, metrics = train_catboost()
```

### Comparação de Modelos

```python
from src.models.model_comparison import run as compare_models
results = compare_models()
```

### Predição com Modelo Treinado

```python
from src.models import PokemonLogisticRegression, PokemonCatBoost

# Carregar modelo salvo
model = PokemonCatBoost.load('catboost_model.pkl')

# Fazer predição
result = model.predict('Pikachu', 'Charizard')
print(f"Vencedor: {result['vencedor_previsto']}")
print(f"Probabilidade: {result['probabilidade_vitoria_p1']:.2%}")
```

## 📊 Métricas Avaliadas

- **Acurácia**: Proporção de predições corretas
- **AUC-ROC**: Área sob a curva ROC
- **Precisão**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos detectados
- **F1-Score**: Média harmônica entre precisão e recall

## 💾 Modelos Salvos

Os modelos treinados são salvos em `data/models/`:
- `logistic_regression_model.pkl`
- `catboost_model.pkl`

Cada arquivo contém:
- Modelo treinado
- Dicionário de estatísticas dos Pokémon
- Lista de features utilizadas

## 🔧 Configuração

Adicione ao seu `.env`:
```
MODELS_PATH=data/models
```

## 📈 Visualizações

O módulo `model_comparison.py` gera:
1. Comparação de métricas (gráfico de barras)
2. Matrizes de confusão lado a lado
3. Importância das features comparativa

## 🎮 Exemplos de Predições

```python
# Batalhas clássicas
model.predict('Pikachu', 'Bulbasaur')
model.predict('Charizard', 'Blastoise')
model.predict('Mewtwo', 'Mew')

# Batalhas lendárias
model.predict('Arceus', 'Rayquaza')
model.predict('Kyogre', 'Groudon')
```

## ⚙️ Parâmetros Ajustáveis

### Regressão Logística
- `C`: Força da regularização (padrão: 1.0)
- `class_weight`: Balanceamento de classes (padrão: 'balanced')

### CatBoost
- `iterations`: Número de árvores (padrão: 1000)
- `learning_rate`: Taxa de aprendizado (padrão: 0.03)
- `depth`: Profundidade das árvores (padrão: 6)
- `verbose`: Frequência de print (padrão: 100)

## 🧪 Pipeline Completo

Execute via `main.py`:
```bash
python main.py
```

Isso executará:
1. ETL (extração, transformação, load)
2. Análise exploratória
3. Treinamento de ambos os modelos
4. Comparação e visualizações
