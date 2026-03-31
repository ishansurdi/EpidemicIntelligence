"""
COVID-19 Data Scraper
Downloads datasets from Our World in Data catalog into data/ subfolders.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent / "data"
OWID_DIR = Path(__file__).parent / "data" / "owid"

DATASETS = [
    {
        "name": "cases_deaths",
        "label": "Cases and Deaths",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/cases_deaths/cases_deaths.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/cases_deaths/cases_deaths.meta.json",
        ],
    },
    {
        "name": "excess_mortality",
        "label": "Excess Mortality",
        "files": [
            "https://catalog.ourworldindata.org/garden/excess_mortality/latest/excess_mortality/excess_mortality.csv",
            "https://catalog.ourworldindata.org/garden/excess_mortality/latest/excess_mortality/excess_mortality.meta.json",
        ],
    },
    {
        "name": "excess_mortality_economist",
        "label": "Excess Mortality (The Economist)",
        "files": [
            "https://catalog.ourworldindata.org/garden/excess_mortality/latest/excess_mortality_economist/excess_mortality_economist.csv",
            "https://catalog.ourworldindata.org/garden/excess_mortality/latest/excess_mortality_economist/excess_mortality_economist.meta.json",
        ],
    },
    {
        "name": "hospitalizations",
        "label": "Hospitalizations",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/hospital/hospital.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/hospital/hospital.meta.json",
        ],
    },
    {
        "name": "vaccinations_global",
        "label": "Vaccinations",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_global/vaccinations_global.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_global/vaccinations_global.meta.json",
        ],
    },
    {
        "name": "vaccinations_age",
        "label": "Vaccinations (by age)",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_age/vaccinations_age.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_age/vaccinations_age.meta.json",
        ],
    },
    {
        "name": "vaccinations_manufacturer",
        "label": "Vaccinations (by manufacturer)",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_manufacturer/vaccinations_manufacturer.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_manufacturer/vaccinations_manufacturer.meta.json",
        ],
    },
    {
        "name": "vaccinations_us",
        "label": "Vaccinations (US)",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_us/vaccinations_us.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/vaccinations_us/vaccinations_us.meta.json",
        ],
    },
    {
        "name": "testing",
        "label": "Testing",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/testing/testing.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/testing/testing.meta.json",
        ],
    },
    {
        "name": "reproduction_rate",
        "label": "Reproduction rate",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/tracking_r/tracking_r.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/tracking_r/tracking_r.meta.json",
        ],
    },
    {
        "name": "google_mobility",
        "label": "Google mobility",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/google_mobility/google_mobility.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/google_mobility/google_mobility.meta.json",
        ],
    },
    {
        "name": "government_response_policy",
        "label": "Government response policy",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/oxcgrt_policy/oxcgrt_policy.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/oxcgrt_policy/oxcgrt_policy.meta.json",
        ],
    },
    {
        "name": "attitudes_yougov",
        "label": "Attitudes (YouGov)",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/yougov/yougov_composite.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/yougov/yougov_composite.meta.json",
        ],
    },
    {
        "name": "donations_covax",
        "label": "Donations (COVAX)",
        "files": [
            "https://catalog.ourworldindata.org/garden/covid/latest/covax/covax.csv",
            "https://catalog.ourworldindata.org/garden/covid/latest/covax/covax.meta.json",
        ],
    },
]


def download_file(url: str, dest: Path) -> bool:
    filename = url.split("/")[-1]
    dest_path = dest / filename
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        dest_path.write_bytes(data)
        size_kb = len(data) / 1024
        print(f"    [OK] {filename} ({size_kb:.1f} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"    [HTTP {e.code}] {filename} — {e.reason}")
    except urllib.error.URLError as e:
        print(f"    [ERROR] {filename} — {e.reason}")
    except Exception as e:
        print(f"    [FAIL] {filename} — {e}")
    return False


def main():
    OWID_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OWID_DIR}\n")
    success_total = 0
    fail_total = 0

    for ds in DATASETS:
        print(f"[{ds['label']}]")
        for url in ds["files"]:
            ok = download_file(url, OWID_DIR)
            if ok:
                success_total += 1
            else:
                fail_total += 1
        print()

    print(f"Done. {success_total} file(s) downloaded, {fail_total} failed.")


if __name__ == "__main__":
    main()
