"""Загрузка и базовая подготовка датасета космической погоды."""

import pandas as pd

HF_DATASET = "juliensimon/space-weather-indices"

# Солнечная активность — причина. Только эти колонки идут в предикторы.
SOLAR_CONTINUOUS = [
    "f107_obs",
    "f107_adj",
    "f107_obs_center81",
    "f107_obs_last81",
]
SOLAR_DISCRETE = [
    "sunspot_number",
    "bartels_day",
    "bartels_rotation",
]
PREDICTORS = SOLAR_CONTINUOUS + SOLAR_DISCRETE

# Геомагнитный отклик — следствие. В предикторы не берём: это утечка.
GEOMAGNETIC = [
    "kp_0000", "kp_0300", "kp_0600", "kp_0900",
    "kp_1200", "kp_1500", "kp_1800", "kp_2100", "kp_sum",
    "ap_0000", "ap_0300", "ap_0600", "ap_0900",
    "ap_1200", "ap_1500", "ap_1800", "ap_2100", "ap_avg",
    "cp", "c9", "is_storm", "storm_level",
]

TARGET = "ap_avg"


def load_raw() -> pd.DataFrame:
    """Сырой датасет с HuggingFace, как есть."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    return ds.to_pandas()


def keep_observed(df: pd.DataFrame) -> pd.DataFrame:
    """Убрать прогнозные строки: в файле есть даты до 2041 года."""
    return df[df["data_type"] == "observed"].copy()


def split_by_time(df: pd.DataFrame, cutoff: str = "2010-01-01"):
    """Хронологическое разбиение.

    Случайный split здесь некорректен: данные автокоррелированы,
    случайное перемешивание утечёт будущее в обучающую выборку.
    """
    df = df.sort_values("date")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    return train, test
