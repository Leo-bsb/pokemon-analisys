import time
import requests
from requests.exceptions import RequestException, Timeout
from src.utils.config import API_BASE_URL, API_USERNAME, API_PASSWORD


class PokemonApiClient:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.token = self._authenticate()
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }

    def _authenticate(self):
        try:
            response = requests.post(
                f"{self.base_url}/login",
                json={
                    "username": API_USERNAME,
                    "password": API_PASSWORD
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("access_token")
        except RequestException as e:
            raise Exception(f"Erro na autenticação: {e}")

    def _get(self, endpoint, params=None, retries=5):
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=10
                )

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Rate limit atingido. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except Timeout:
                wait_time = 2 ** attempt
                print(f"Timeout. Tentando novamente em {wait_time}s...")
                time.sleep(wait_time)

            except RequestException as e:
                raise Exception(f"Erro na requisição: {e}")

        raise Exception("Falha após múltiplas tentativas.")

    def get_all_pages(self, endpoint):
        all_data = []
        page = 1

        while True:
            print(f"Buscando {endpoint} - página {page}")

            data = self._get(endpoint, params={"page": page})

            if not isinstance(data, dict):
                print("Resposta inesperada da API.")
                break

            # detecta automaticamente a chave da lista
            items_key = next(
                (k for k, v in data.items() if isinstance(v, list)),
                None
            )

            if not items_key:
                print("Nenhuma lista encontrada na resposta.")
                break

            items = data[items_key]

            if not items:
                print("Sem mais dados.")
                break

            all_data.extend(items)

            total = data.get("total")
            per_page = data.get("per_page", len(items))

            # se a API fornecer total
            if total and page * per_page >= total:
                break

            page += 1

        return all_data
    
    def get_pokemon_attributes(self, pokemon_id):
        return self._get(f"pokemon/{pokemon_id}")