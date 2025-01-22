import requests
from typing import Dict
import logging
import time
from urllib.parse import urlencode
import pandas as pd
import matplotlib.pyplot as plt


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
            logger.info(f"Connection error: {e}. Esperando {cool_off} segundos")

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.info(f"Error 404 Not Found: {e}. Esperando {cool_off} segundos")
                raise
            else:
                logger.info(f"HTTP error: {e}. Esperando {cool_off} segundos")

        except requests.exceptions.RequestException as req_err:
            logger.info(f"Request error: {req_err}. Esperando {cool_off} segundos")

        if cool_off < num_attempts:
            time.sleep(cool_off)
            cool_off = cool_off * 2
        else:
            raise


def get_data_meteo(
        latitude: str, longitude: str, start_date: str, end_date: str
) -> Dict:
    headers = {}
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
    }

    return make_api_call_with_cool_off(API_URL + urlencode(params, safe=","), headers)


def get_processed_df_by_month(data: pd.DataFrame) -> pd.DataFrame:
    data["time"] = pd.to_datetime(data["time"])

    grouped_data = data.groupby([data["city"], data["time"].dt.to_period("M")])

    processed_data = []

    for (city, month), group in grouped_data:
        monthly_data_per_city = {"city": city, "month": month.to_timestamp()}

        for variable in VARIABLES:
            monthly_data_per_city[f"{variable}_max"] = group[variable].max()
            monthly_data_per_city[f"{variable}_min"] = group[variable].min()
            monthly_data_per_city[f"{variable}_mean"] = group[variable].mean()
            monthly_data_per_city[f"{variable}_std"] = group[variable].std()

        processed_data.append(monthly_data_per_city)

    return pd.DataFrame(processed_data)


def plot_df(processed_df: pd.DataFrame):
    rows = len(VARIABLES)
    cols = len(processed_df["city"].unique())
    fig, axs = plt.subplots(rows, cols, figsize=(10, 6 * rows))

    for i, variable in enumerate(VARIABLES):
        for k, city in enumerate(processed_df["city"].unique()):
            city_data = processed_df[processed_df["city"] == city]

            # Plot mean values
            axs[i, k].plot(
                city_data["month"],
                city_data[f"{variable}_mean"],
                label=f"{city} (mean)",
                color=f"C{k}",
            )

            # Plot max and min as shaded area
            axs[i, k].fill_between(
                city_data["month"],
                city_data[f"{variable}_min"],
                city_data[f"{variable}_max"],
                alpha=0.2,
                color=f"C{k}",
            )

            # Plot std as error bars (optional, can be commented out if too noisy)
            axs[i, k].errorbar(
                city_data["month"],
                city_data[f"{variable}_mean"],
                yerr=city_data[f"{variable}_std"],
                fmt="none",
                ecolor=f"C{k}",
                alpha=0.5,
            )

            axs[i, k].set_xlabel("Date")
            axs[i, k].set_title(variable)

            if k == 0:
                axs[i, k].set_ylabel("Value")

            axs[i, k].legend()

    plt.tight_layout()
    plt.savefig("src/module_1/climate_evolution.png", bbox_inches="tight")


def main():
    cities_df_list = []
    start_date = "2010-01-01"
    end_date = "2020-12-31"

    for city, coordinates in COORDINATES.items():
        latitude = coordinates["latitude"]
        longitude = coordinates["longitude"]

        city_df = pd.DataFrame(
            get_data_meteo(latitude, longitude, start_date, end_date)["daily"]
        ).assign(city=city)

        cities_df_list.append(city_df)

    cities_df = pd.concat(cities_df_list)

    processed_df = get_processed_df_by_month(cities_df)

    plot_df(processed_df)


if __name__ == "__main__":
    main()
