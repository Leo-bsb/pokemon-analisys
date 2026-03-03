from pathlib import Path
import json
import time
from src.utils.config import RAW_DATA_PATH
from src.utils.api import PokemonApiClient


RAW_PATH = Path(RAW_DATA_PATH)
RAW_PATH.mkdir(parents=True, exist_ok=True)


CHECKPOINT_FILE = RAW_PATH / "pokemon_checkpoint.json"


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def enrich_pokemon(client, pokemon_list, batch_size=20, delay=0.2):
    """
    Enriquece pokémons com checkpoint e controle de carga.
    """

    enriched_data = load_checkpoint()
    processed_ids = {p["id"] for p in enriched_data if "id" in p}

    print(f"{len(processed_ids)} pokémons já processados (checkpoint).")

    count = 0

    for pokemon in pokemon_list:
        pokemon_id = pokemon.get("id")

        if not pokemon_id or pokemon_id in processed_ids:
            continue

        try:
            print(f"Buscando atributos do pokemon {pokemon_id}...")

            attributes = client.get_pokemon_attributes(pokemon_id)

            pokemon["attributes"] = attributes
            enriched_data.append(pokemon)

            count += 1

            # salva a cada batch
            if count % batch_size == 0:
                save_json(CHECKPOINT_FILE, enriched_data)
                print(f"Checkpoint salvo ({len(enriched_data)} registros).")

            time.sleep(delay)

        except Exception as e:
            print(f"Erro ao processar pokemon {pokemon_id}: {e}")

    # salva final
    save_json(CHECKPOINT_FILE, enriched_data)
    print("Enriquecimento finalizado.")

    return enriched_data


def run():
    client = PokemonApiClient()

    print("Extraindo lista de pokemons...")
    pokemon_list = client.get_all_pages("pokemon")
    print(f"{len(pokemon_list)} pokémons encontrados.\n")

    enriched_pokemon = enrich_pokemon(client, pokemon_list)

    final_file = RAW_PATH / "pokemon.json"
    save_json(final_file, enriched_pokemon)

    print(f"{len(enriched_pokemon)} pokémons salvos em {final_file}\n")

    # 🔒 Remove checkpoint SOMENTE após salvar com sucesso
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint removido.\n")

    print("Extraindo combats...")
    combats = client.get_all_pages("combats")

    combats_file = RAW_PATH / "combats.json"
    save_json(combats_file, combats)

    print(f"{len(combats)} combats salvos em {combats_file}")

    

    