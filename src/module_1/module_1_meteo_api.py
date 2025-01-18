import logging
import time
from typing import Any, Dict
import requests
from http import HTTPStatus

API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m"]

logger = logging.getLogger(__name__)

class TooManyRequestsError(Exception):
    """Excepción personalizada para errores de límite de solicitudes."""
    pass

def make_api_request(url: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    COOLOFF_TIME = 0.5  
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == HTTPStatus.OK:
                return response.json()
            
            elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  
                logger.warning(f"Rate limit alcanzado (intento {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    # Incrementar el tiempo de espera con cada intento
                    sleep_time = COOLOFF_TIME * (attempt + 2)
                    logger.info(f"Esperando {sleep_time} segundos antes del siguiente intento...")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise TooManyRequestsError("Se alcanzó el límite de intentos por rate limit")
            
            else:
                response.raise_for_status()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en la petición (intento {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(COOLOFF_TIME)
                continue
            raise
    
    raise Exception("Se alcanzó el número máximo de intentos sin éxito")

def get_data_meteo(city: str, COORDINATES: Dict[str, float], VARIABLES, start_year="2024", end_year="2024") -> Dict:
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "hourly": ",".join(VARIABLES),
    }

    return make_api_request(API_URL, params)

def main():
    get_data_meteo("Madrid", COORDINATES, VARIABLES)
    get_data_meteo("London", COORDINATES, VARIABLES)
    get_data_meteo("Rio", COORDINATES, VARIABLES)

if __name__ == "__main__":
    main()
