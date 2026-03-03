import duckdb
from pathlib import Path
from src.utils.config import PROCESSED_DATA_PATH, DATABASE_PATH

# Garante que o diretório do banco de dados exista
DATABASE_PATH.mkdir(parents=True, exist_ok=True)

# Caminho completo para o arquivo do banco de dados
DB_FILE = DATABASE_PATH / "pokemon.duckdb"

def run():
    """
    Carrega os dados processados (parquet) para um banco DuckDB.
    """
    print("Carregando dados para o banco DuckDB...")

    # Conecta ao banco (cria se não existir)
    con = duckdb.connect(str(DB_FILE))

    # Caminhos dos arquivos parquet processados
    pokemon_parquet = PROCESSED_DATA_PATH / "pokemon_processed.parquet"
    combats_parquet = PROCESSED_DATA_PATH / "combats_processed.parquet"

    # Verifica se os arquivos existem
    if not pokemon_parquet.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pokemon_parquet}")
    if not combats_parquet.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {combats_parquet}")

    # Cria as tabelas a partir dos parquet
    con.execute(f"""
        CREATE OR REPLACE TABLE pokemon AS
        SELECT * FROM '{pokemon_parquet}'
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE combats AS
        SELECT * FROM '{combats_parquet}'
    """)

    # Opcional: verificar se as tabelas foram criadas
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"Tabelas criadas: {[t[0] for t in tables]}")

    con.close()
    print(f"Dados carregados com sucesso em: {DB_FILE}")

if __name__ == "__main__":
    run()