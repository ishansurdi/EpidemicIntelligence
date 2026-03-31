from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def load_jhu_confirmed() -> pd.DataFrame:
    file_path = DATA_ROOT / "time_series_covid19_confirmed_global.csv"
    return pd.read_csv(file_path)


def load_owid_table(name: str) -> pd.DataFrame:
    file_path = DATA_ROOT / "owid" / f"{name}.csv"
    return pd.read_csv(file_path)


def load_google_mobility() -> pd.DataFrame:
    lightweight = DATA_ROOT / "owid" / "google_mobility.csv"
    if lightweight.exists():
        return pd.read_csv(lightweight)
    return pd.read_csv(DATA_ROOT / "Global_Mobility_Report.csv")


def list_available_owid_tables() -> list[str]:
    owid_root = DATA_ROOT / "owid"
    names: list[str] = []
    for path in owid_root.glob("*.csv"):
        names.append(path.stem)
    return sorted(names)
