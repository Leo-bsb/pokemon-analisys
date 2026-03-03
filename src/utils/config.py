import os
from pathlib import Path
from dotenv import load_dotenv

# Diretório base do projeto (sobe 2 níveis a partir deste arquivo)
BASE_DIR = Path(__file__).resolve().parents[2]

# Carrega variáveis do arquivo .env na raiz do projeto
load_dotenv(BASE_DIR / ".env")

# ==================== Configurações da API ====================
API_BASE_URL = os.getenv("API_BASE_URL")
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

# Validação das variáveis obrigatórias
missing_vars = [var for var in ["API_BASE_URL", "API_USERNAME", "API_PASSWORD"] if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Variáveis de ambiente obrigatórias não definidas: {', '.join(missing_vars)}")

# ==================== Configuração de diretórios ====================
def get_absolute_path(env_var: str, default_rel_path: str) -> Path:
    """
    Retorna um Path absoluto a partir de uma variável de ambiente.
    Se a variável não existir, usa o caminho relativo padrão (resolvido em relação a BASE_DIR).
    Se a variável existir e for um caminho absoluto, mantém; caso contrário, interpreta como relativo a BASE_DIR.
    """
    path_str = os.getenv(env_var, default_rel_path)
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()  # resolve para caminho canônico absoluto

# Definição dos caminhos (agora todos absolutos em relação a BASE_DIR, a menos que a env forneça absoluto)
RAW_DATA_PATH = get_absolute_path("RAW_DATA_PATH", "data/raw")
PROCESSED_DATA_PATH = get_absolute_path("PROCESSED_DATA_PATH", "data/processed")
DATABASE_PATH = get_absolute_path("DATABASE_PATH", "data/database")
MODELS_PATH = get_absolute_path("MODELS_PATH", "data/models")

# Lista de todos os diretórios que devem existir
DATA_DIRS = [RAW_DATA_PATH, PROCESSED_DATA_PATH, DATABASE_PATH, MODELS_PATH]

# Cria os diretórios se não existirem
for dir_path in DATA_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"RAW_DATA_PATH: {RAW_DATA_PATH}")
print(f"PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")
print(f"DATABASE_PATH: {DATABASE_PATH}")
print(f"MODELS_PATH: {MODELS_PATH}")