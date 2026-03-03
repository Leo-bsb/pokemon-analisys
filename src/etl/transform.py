import polars as pl
from pathlib import Path
from typing import Dict, List
from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Definindo os caminhos como Path
RAW_PATH = Path(RAW_DATA_PATH)
PROCESSED_PATH = Path(PROCESSED_DATA_PATH)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

TYPE_ADVANTAGE_DICT = {
    "Fire": ["Grass", "Ice", "Bug", "Steel"],
    "Water": ["Fire", "Ground", "Rock"],
    "Grass": ["Ground", "Rock", "Water"],
    "Electric": ["Water", "Flying"],
    "Ice": ["Flying", "Ground", "Grass", "Dragon"],
    "Fighting": ["Normal", "Rock", "Steel", "Ice", "Dark"],
    "Poison": ["Grass", "Fairy"],
    "Flying": ["Fighting", "Bug", "Grass"],
    "Psychic": ["Fighting", "Poison"],
    "Bug": ["Grass", "Psychic", "Dark"],
    "Rock": ["Flying", "Bug", "Fire", "Ice"],
    "Ghost": ["Ghost", "Psychic"],
    "Dark": ["Ghost", "Psychic"],
    "Steel": ["Ice", "Rock", "Fairy"],
    "Dragon": ["Dragon"],
    "Fairy": ["Fighting", "Dragon", "Dark"],
    "Ground": ["Poison", "Rock", "Steel", "Fire", "Electric"]
}

# Dicionário de fraquezas (tipos que têm vantagem sobre cada tipo)
WEAKNESS_DICT = {}
for attacker, defenders in TYPE_ADVANTAGE_DICT.items():
    for defender in defenders:
        if defender not in WEAKNESS_DICT:
            WEAKNESS_DICT[defender] = []
        if attacker not in WEAKNESS_DICT[defender]:
            WEAKNESS_DICT[defender].append(attacker)


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def compute_type_advantage(types1: List[str], types2: List[str]) -> Dict:
    """
    Calcula a vantagem de tipo entre dois Pokémon com base em seus tipos.
    
    Args:
        types1: Lista de tipos do primeiro Pokémon
        types2: Lista de tipos do segundo Pokémon
    
    Returns:
        Dicionário com vantagem (0, 1 ou 2) e motivo
    """
    types1 = [t for t in types1 if t != "None"]
    types2 = [t for t in types2 if t != "None"]
    
    score_1 = sum(
        1 for t in types1 
        for t2 in types2 
        if t2 in TYPE_ADVANTAGE_DICT.get(t, [])
    )
    score_2 = sum(
        1 for t in types2 
        for t1 in types1 
        if t1 in TYPE_ADVANTAGE_DICT.get(t, [])
    )
    
    if score_1 > score_2:
        return {"advantage": 1, "reason": f"{types1} > {types2}"}
    elif score_2 > score_1:
        return {"advantage": 2, "reason": f"{types2} > {types1}"}
    else:
        return {"advantage": 0, "reason": "None"}


def compute_advantage_list(types: List[str]) -> List[str]:
    """
    Obtém a lista de tipos contra os quais este Pokémon possui vantagem.
    
    Args:
        types: Lista de tipos do Pokémon
    
    Returns:
        Lista única de tipos com vantagem
    """
    types = [t for t in types if t != "None"]
    
    advantages = []
    for t in types:
        advantages.extend(TYPE_ADVANTAGE_DICT.get(t, []))
    
    # Remove duplicatas preservando a ordem
    return list(dict.fromkeys(advantages))


def compute_weakness_list(types: List[str]) -> List[str]:
    """
    Obtém a lista de tipos contra os quais este Pokémon é fraco (tipos que têm vantagem sobre ele).
    
    Args:
        types: Lista de tipos do Pokémon
    
    Returns:
        Lista única de tipos que são fortes contra este Pokémon
    """
    types = [t for t in types if t != "None"]
    
    weaknesses = []
    for t in types:
        weaknesses.extend(WEAKNESS_DICT.get(t, []))
    
    # Remove duplicatas preservando a ordem
    return list(dict.fromkeys(weaknesses))


# =============================================================================
# DATA LOAD
# =============================================================================

def load_pokemon_data(filepath: str) -> pl.DataFrame:
    """Carrega e prepara os dados de atributos dos Pokémon."""
    df = (
        pl.read_json(filepath)
        .drop(["id", "name"])
        .unnest("attributes")
    )
    
    # Conversão de tipos
    df = df.with_columns(
        pl.col("id").cast(pl.Int64)
    )
    
    # Limpeza dos campos generation e legendary
    df = df.with_columns([
        pl.col("generation")
            .cast(pl.Utf8)
            .str.extract(r"(\d+)", 1)
            .cast(pl.Int64)
            .alias("generation"),
        
        pl.col("legendary")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .eq("true")
            .alias("legendary")
    ])
    
    return df


def load_combat_data(filepath: str) -> pl.DataFrame:
    """Carrega e prepara os dados de resultados de combates."""
    df = pl.read_json(filepath)
    
    # Converte IDs para Int64
    df = df.with_columns([
        pl.col("first_pokemon").cast(pl.Int64),
        pl.col("second_pokemon").cast(pl.Int64),
        pl.col("winner").cast(pl.Int64),
    ])
    
    # -------------------------------------------------------------------------
    # REMOVE DUPLICATAS (independente da ordem)
    # -------------------------------------------------------------------------
    
    # Cria par normalizado (independente da ordem)
    df = df.with_columns(
        pl.concat_list(["first_pokemon", "second_pokemon"])
        .list.sort()
        .alias("pair_id")
    )
    
    # Remove pares duplicados
    df = df.unique(subset=["pair_id"]).drop("pair_id")
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def join_pokemon_attributes(
    df_combats: pl.DataFrame, 
    df_pokemon: pl.DataFrame
) -> pl.DataFrame:
    """Junta os dados de combate com os atributos dos Pokémon para ambos os lutadores."""
    
    # Junta dados do primeiro Pokémon
    df = df_combats.join(
        df_pokemon,
        left_on="first_pokemon",
        right_on="id",
        how="left",
        suffix="_1"
    )
    
    # Junta dados do segundo Pokémon
    df = df.join(
        df_pokemon,
        left_on="second_pokemon",
        right_on="id",
        how="left",
        suffix="_2"
    )
    
    # Renomeia colunas para maior clareza
    df = df.rename({
        "name": "pokemon_1",
        "hp": "hp_1",
        "attack": "attack_1",
        "defense": "defense_1",
        "sp_attack": "sp_attack_1",
        "sp_defense": "sp_defense_1",
        "speed": "speed_1",
        "generation": "generation_1",
        "legendary": "legendary_1",
        "types": "types_1",
        "name_2": "pokemon_2",
        "hp_2": "hp_2",
        "attack_2": "attack_2",
        "defense_2": "defense_2",
        "sp_attack_2": "sp_attack_2",
        "sp_defense_2": "sp_defense_2",
        "speed_2": "speed_2",
        "generation_2": "generation_2",
        "legendary_2": "legendary_2",
        "types_2": "types_2",
    })
    
    df = df.filter(
        pl.col("pokemon_1").is_not_null() &
        pl.col("pokemon_2").is_not_null()
    )
    
    return df


def create_type_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cria atributos relacionados a tipos, incluindo cálculos de vantagem."""
    
    # Divide os tipos em colunas individuais
    df = df.with_columns([
        pl.col("types_1").str.split("/").list.get(0).alias("type1_1"),
        pl.col("types_1").str.split("/").list.get(1, null_on_oob=True).fill_null("None").alias("type2_1"),
        pl.col("types_2").str.split("/").list.get(0).alias("type1_2"),
        pl.col("types_2").str.split("/").list.get(1, null_on_oob=True).fill_null("None").alias("type2_2"),
    ])
    
    # Calcula a vantagem de tipo
    df = df.with_columns(
        pl.struct(["type1_1", "type2_1", "type1_2", "type2_2"])
        .map_elements(
            lambda row: compute_type_advantage(
                [row["type1_1"], row["type2_1"]],
                [row["type1_2"], row["type2_2"]]
            ),
            return_dtype=pl.Struct([
                pl.Field("advantage", pl.Int64),
                pl.Field("reason", pl.Utf8)
            ])
        )
        .alias("type_adv_info")
    )
    
    # Extrai os campos de vantagem
    df = df.with_columns([
        pl.col("type_adv_info").struct.field("advantage").alias("type_advantage"),
        pl.col("type_adv_info").struct.field("reason").alias("advantage_reason")
    ]).drop("type_adv_info")
    
    return df


def create_stat_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cria atributos de comparação estatística entre os combatentes."""
    
    # Calcula os atributos totais para cada Pokémon
    df = df.with_columns([
        (pl.col("hp_1") + pl.col("attack_1") + pl.col("defense_1") +
         pl.col("sp_attack_1") + pl.col("sp_defense_1") + pl.col("speed_1"))
        .alias("total_stats_1"),
        
        (pl.col("hp_2") + pl.col("attack_2") + pl.col("defense_2") +
         pl.col("sp_attack_2") + pl.col("sp_defense_2") + pl.col("speed_2"))
        .alias("total_stats_2"),
    ])
    
    # Calcula as diferenças
    df = df.with_columns([
        (pl.col("total_stats_1") - pl.col("total_stats_2")).alias("stats_diff"),
        (pl.col("speed_1") - pl.col("speed_2")).alias("speed_diff"),
        (pl.col("attack_1") - pl.col("attack_2")).alias("attack_diff"),
        (pl.col("defense_1") - pl.col("defense_2")).alias("defense_diff"),
        (pl.col("total_stats_1") + pl.col("total_stats_2")).alias("sum_attributes"),
    ])
    
    return df


def create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Cria a variável alvo (nome do vencedor)."""
    
    df = df.with_columns(
        pl.when(pl.col("winner") == pl.col("first_pokemon"))
        .then(pl.col("pokemon_1"))
        .otherwise(pl.col("pokemon_2"))
        .alias("winner_name")
    )
    
    return df


def clean_and_reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Remove colunas desnecessárias e reordena para melhor legibilidade."""
    
    # Remove colunas técnicas
    df = df.drop([
        "first_pokemon",
        "second_pokemon",
        "winner",
    ])
    
    # Converte colunas Float64 para Int64
    df = df.with_columns(
        pl.col(pl.Float64).cast(pl.Int64)
    )
    
    # Define a ordem desejada das colunas
    ordered_columns = [
        # Pokémon 1
        "pokemon_1", "generation_1", "legendary_1", "types_1", "type1_1", "type2_1",
        "hp_1", "attack_1", "defense_1", "sp_attack_1", "sp_defense_1", "speed_1",
        "total_stats_1",
        
        # Pokémon 2
        "pokemon_2", "generation_2", "legendary_2", "types_2", "type1_2", "type2_2",
        "hp_2", "attack_2", "defense_2", "sp_attack_2", "sp_defense_2", "speed_2",
        "total_stats_2",
        
        # Comparações
        "type_advantage", "advantage_reason", "stats_diff", "speed_diff",
        "attack_diff", "defense_diff", "sum_attributes",
        
        # Alvo
        "winner_name"
    ]
    
    return df.select(ordered_columns)


# Adiciona todas as features de tipo ao DataFrame de Pokémon
def enrich_pokemon_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adiciona detalhes de tipo ao DataFrame de Pokémon:
    - Divide os tipos em type1 e type2
    - Lista de tipos contra os quais este Pokémon tem vantagem (ofensivo)
    - Lista de tipos contra os quais este Pokémon é fraco (defensivo)
    """
    
    # Divide os tipos em colunas individuais
    df = df.with_columns([
        pl.col("types").str.split("/").list.get(0).alias("type1"),
        pl.col("types").str.split("/").list.get(1, null_on_oob=True).fill_null("None").alias("type2"),
    ])
    
    # Cria uma coluna temporária de lista para cálculo de vantagem/fraqueza
    df = df.with_columns(
        pl.concat_list(["type1", "type2"]).alias("type_list")
    )
    
    # Calcula a lista de vantagens (tipos contra os quais este pokémon é forte)
    df = df.with_columns(
        pl.col("type_list")
        .map_elements(
            compute_advantage_list,
            return_dtype=pl.List(pl.Utf8)
        )
        .alias("advantage")
    )
    
    # Calcula a lista de fraquezas (tipos contra os quais este pokémon é fraco)
    df = df.with_columns(
        pl.col("type_list")
        .map_elements(
            compute_weakness_list,
            return_dtype=pl.List(pl.Utf8)
        )
        .alias("weakness")
    )
    
    # Remove a coluna temporária
    df = df.drop("type_list")
    
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_combat_data(
    pokemon_filepath: str,
    combat_filepath: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Pipeline ETL principal para dados de combate de Pokémon.
    
    Args:
        pokemon_filepath: Caminho para pokemon.json
        combat_filepath: Caminho para combats.json
    
    Returns:
        Tupla (df_pokemon_enriquecido, df_combates_processado)
    """
    
    print("Carregando dados...")
    df_pokemon = load_pokemon_data(pokemon_filepath)
    df_combats = load_combat_data(combat_filepath)
    
    print("Juntando atributos dos Pokémon com dados de combate...")
    df_combats = join_pokemon_attributes(df_combats, df_pokemon)
    
    print("Criando atributos de tipo...")
    df_combats = create_type_features(df_combats)
    
    print("Criando atributos estatísticos...")
    df_combats = create_stat_features(df_combats)
    
    print("Criando variável alvo...")
    df_combats = create_target_variable(df_combats)
    
    print("Limpando e reordenando colunas...")
    df_combats = clean_and_reorder_columns(df_combats)
    
    print("Enriquecendo dados dos Pokémon com detalhes de tipo...")
    df_pokemon = enrich_pokemon_features(df_pokemon)
    
    print("Pipeline ETL concluído com sucesso!")
    
    return df_pokemon, df_combats


def run():
    """
    Executa a transformação completa: carrega dados crus, processa e salva no diretório processado.
    """
    print("Iniciando transformação dos dados...")

    # Caminhos dos arquivos de entrada (raw)
    pokemon_raw = RAW_PATH / "pokemon.json"
    combats_raw = RAW_PATH / "combats.json"

    # Verifica se os arquivos existem
    if not pokemon_raw.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pokemon_raw}")
    if not combats_raw.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {combats_raw}")

    # Processa os dados
    df_pokemon_enriched, df_combats_processed = process_combat_data(
        str(pokemon_raw), str(combats_raw)
    )

    # Define os caminhos de saída
    pokemon_output = PROCESSED_PATH / "pokemon_processed.parquet"
    combats_output = PROCESSED_PATH / "combats_processed.parquet"

    # Salva
    df_pokemon_enriched.write_parquet(pokemon_output)
    df_combats_processed.write_parquet(combats_output)

    print(f"Dados processados salvos em:\n  {pokemon_output}\n  {combats_output}")

if __name__ == "__main__":
    run()