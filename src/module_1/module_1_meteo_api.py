import requests
from typing import Dict
import logging
import time
from urllib.parse import urlencode


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
 "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
 "London": {"latitude": 51.507351, "longitude": -0.127758},
 "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def make_api_call_with_cool_off(
    url: str, 
    headers: Dict[str, any], 
    payload: Dict[str, any] = None,
    num_attempts: int = 5,
    cool_off: int = 1
) -> Dict:
    for i in range(num_attempts):
        try:
            if payload:
                response = requests.post(url, headers=headers, json=payload)
            else:
                response = requests.get(url, headers=headers)

            response.raise_for_status()

            return response.json()

        except requests.exceptions.ConnectionError as e:
           logger.info(f"Connection error: {e}. Esperando {cool_off}segundos...")

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.info(f"Error 404 Not Found: {e}. Esperando {cool_off} segundos...")
                raise 
            else:
                logger.info(f"HTTP error: {e}. Esperando {cool_off}segundos...")

        except requests.exceptions.RequestException as req_err:
            logger.info(f"Request error: {req_err}. Esperando {cool_off}segundos...")

        if cool_off < num_attempts:
            time.sleep(cool_off)
            cool_off = cool_off * 2 
        else:
            raise 

    def get_data_meteo(latitude: str, longitude: str, start_date: str, end_date: str) -> Dict:
        headers = {}
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(VARIABLES),
        }

        return make_api_call_with_cool_off(API_URL + urlencode(params, safe=","), headers)

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
