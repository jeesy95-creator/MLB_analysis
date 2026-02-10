"""
mlb_salary_cluster_analysis.py
---------------------------------

This script provides a complete pipeline for collecting Major League Baseball (MLB)
player salary and performance data, clustering players into archetypes (Power,
Contact and Speed), comparing salaries across these archetypes, and building a
predictive model for next‑season salaries.

Data sources
------------

* **Player salaries** – The script scrapes Spotrac's MLB salary rankings
  (https://www.spotrac.com/mlb/rankings/player) for a given season.  Spotrac
  lists the top salaries for active players and exposes filters for season,
  team and position【254683540446308†L1238-L1264】.  The scraping function uses
  `requests` with a desktop user‑agent string to download the HTML and
  `BeautifulSoup` to parse the player names, team/position and salaries from
  the ranking list【254683540446308†L1238-L1266】.  Spotrac may block automated
  requests in some environments; if you encounter HTTP 403 errors, consider
  using a headless browser (e.g. Selenium/ChromeDriver) to load the page and
  pass the resulting HTML into the parser.

* **Player performance** – The script queries MLB’s public Stats API
  (https://statsapi.mlb.com) to download season‑aggregated hitting statistics
  for all players.  The API returns metrics such as home runs, strikeouts,
  batting average, on‑base percentage, slugging percentage, hits and stolen
  bases.  A high `limit` parameter (e.g. 10 000) ensures that all players are
  returned in a single call.

The script merges salaries with performance data on player names, then
standardises the performance metrics, clusters players via k‑means into three
groups, labels each cluster according to its dominant characteristics, and
produces box plots and summary statistics comparing salary distributions by
player type.  Finally, a regression model is fit to predict next‑season
salaries based on the current season’s performance and cluster labels.

Example usage
-------------

```
from mlb_salary_cluster_analysis import (
    fetch_salary_data,
    fetch_hitting_stats,
    build_dataset,
    cluster_players,
    label_clusters,
    plot_salary_boxplot,
    train_salary_model,
)

# fetch data for 2025 (performance) and 2026 (salaries)
salary_df = fetch_salary_data(year=2026)
stats_df  = fetch_hitting_stats(year=2025)

# build combined dataset
data_df = build_dataset(salary_df, stats_df)

# cluster players into archetypes
data_df = cluster_players(data_df, n_clusters=3)
data_df = label_clusters(data_df)

# visualise salary distribution by archetype
plot_salary_boxplot(data_df)

# train a predictive model for next‑season salaries
model, features, target = train_salary_model(data_df)
```

Note
----

This script depends on third‑party libraries (`requests`, `beautifulsoup4`,
`pandas`, `numpy`, `scikit‑learn`, `matplotlib` and `seaborn`).  Install them
via pip if they are not already available:

```
pip install requests beautifulsoup4 pandas numpy scikit-learn matplotlib seaborn
```

Because Spotrac may enforce rate limits or block non‑browser requests, you
should run `fetch_salary_data` from an environment with network access that is
allowed to retrieve pages from spotrac.com.  If `requests` returns status 403
or 503, consider using Selenium or saving the HTML manually and passing it
directly to `parse_salary_html`.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Salary scraping
# -----------------------------------------------------------------------------

def fetch_salary_data(year: int, type_: str = "Payroll") -> pd.DataFrame:
    """Fetch MLB player salary data from Spotrac for a given season.

    Spotrac lists salaries for active players filtered by season, team and
    position【254683540446308†L1238-L1264】.  This function downloads the ranking
    page and parses the player name, team/position and salary.

    Parameters
    ----------
    year : int
        Target season (e.g. 2026).  Spotrac defaults to the current year if
        an unsupported year is requested.
    type_ : str, optional
        Salary type to retrieve.  Valid types include "Payroll", "Total Cash",
        "Base Salary", "Contract Value", "Contract Average", etc.  The type
        string is not used in the URL (Spotrac shows payroll rankings by
        default) but is retained for clarity and potential future extension.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns: `player`, `team_position`, `salary` and
        `season`.

    Notes
    -----
    Spotrac occasionally rejects automated requests.  If a 403 or 503 status
    code is returned, the function logs a warning and returns an empty
    dataframe.  You may need to use a headless browser to circumvent these
    restrictions.
    """
    url = "https://www.spotrac.com/mlb/rankings/player"
    params = {"year": year}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
    except Exception as exc:
        logger.warning("Failed to fetch salary page for %s: %s", year, exc)
        return pd.DataFrame(columns=["player", "team_position", "salary", "season"])

    if resp.status_code != 200:
        logger.warning(
            "Spotrac responded with status %s for year %s; returning empty dataframe",
            resp.status_code,
            year,
        )
        return pd.DataFrame(columns=["player", "team_position", "salary", "season"])

    html = resp.text
    return parse_salary_html(html, year)


def parse_salary_html(html: str, season: int) -> pd.DataFrame:
    """Parse salary HTML markup into a dataframe.

    The ranking page contains a list where each entry includes a rank number,
    player name (within an anchor tag), a team/position string and a salary
    formatted with dollar signs and commas.  This parser uses a regular
    expression to capture these fields in order and returns them as a
    dataframe.

    Parameters
    ----------
    html : str
        Raw HTML markup retrieved from Spotrac.
    season : int
        Season associated with the salaries.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns `player`, `team_position`, `salary` and
        `season`.
    """
    soup = BeautifulSoup(html, "html.parser")
    salaries = []
    players = []
    team_positions = []
    # salary pattern matches strings like "$61,875,000"
    salary_pattern = re.compile(r"\$[\d,]+")
    for sal in soup.find_all(string=salary_pattern):
        salary_str = sal.strip()
        salary_clean = int(salary_str.replace("$", "").replace(",", ""))
        parent = sal.parent
        current = parent
        player_name = None
        team_pos = None
        steps = 0
        while current and steps < 4:
            a = current.find("a") if hasattr(current, "find") else None
            if a and a.get_text(strip=True):
                player_name = a.get_text(strip=True)
                # The team and position usually follow the anchor tag as text
                following_text = a.find_next(string=True)
                if following_text:
                    txt = following_text.strip()
                    if "," in txt:
                        team_pos = txt
                break
            current = current.parent
            steps += 1
        if player_name and team_pos:
            players.append(player_name)
            team_positions.append(team_pos)
            salaries.append(salary_clean)
    df = pd.DataFrame({
        "player": players,
        "team_position": team_positions,
        "salary": salaries,
    })
    df["season"] = season
    df = df.drop_duplicates(subset=["player", "season"])
    return df


# -----------------------------------------------------------------------------
# Stats API fetching
# -----------------------------------------------------------------------------

def fetch_hitting_stats(year: int) -> pd.DataFrame:
    """Fetch season‑level hitting statistics for all MLB players using the Stats API.

    Parameters
    ----------
    year : int
        Season year (e.g. 2025).

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per player and columns for player id, name and
        various hitting metrics (HR, strikeouts, hits, at bats, batting
        average, on‑base percentage, slugging percentage, OPS, stolen bases).
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&playerPool=all&group=hitting&season={year}&limit=10000"
    )
    try:
        resp = requests.get(url, timeout=30)
    except Exception as exc:
        logger.warning("Failed to fetch hitting stats for %s: %s", year, exc)
        return pd.DataFrame()
    if resp.status_code != 200:
        logger.warning(
            "Stats API responded with status %s for year %s; returning empty dataframe",
            resp.status_code,
            year,
        )
        return pd.DataFrame()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    records = []
    for item in splits:
        player = item.get("player", {})
        stats = item.get("stat", {})
        record = {
            "player_id": player.get("id"),
            "player": player.get("fullName"),
            "team_id": item.get("team", {}).get("id"),
            "team_name": item.get("team", {}).get("name"),
            "season": year,
            "games": stats.get("gamesPlayed"),
            "home_runs": stats.get("homeRuns"),
            "strikeouts": stats.get("strikeOuts"),
            "hits": stats.get("hits"),
            "at_bats": stats.get("atBats"),
            "avg": float(stats.get("avg", "0") if stats.get("avg", "0") != "-.-" else 0.0),
            "obp": float(stats.get("obp", "0") if stats.get("obp", "0") != "-.-" else 0.0),
            "slg": float(stats.get("slg", "0") if stats.get("slg", "0") != "-.-" else 0.0),
            "ops": float(stats.get("ops", "0") if stats.get("ops", "0") != "-.-" else 0.0),
            "stolen_bases": stats.get("stolenBases"),
        }
        records.append(record)
    df = pd.DataFrame.from_records(records)
    numeric_cols = [
        "games", "home_runs", "strikeouts", "hits", "at_bats", "avg", "obp", "slg", "ops", "stolen_bases"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


# -----------------------------------------------------------------------------
# Dataset merging and preprocessing
# -----------------------------------------------------------------------------

def build_dataset(
    salary_df: pd.DataFrame, stats_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge salary and performance data on player names.

    Player names may differ slightly between Spotrac and the Stats API.  This
    function performs a case‑insensitive merge on the name string and drops
    entries that do not match.

    Parameters
    ----------
    salary_df : pandas.DataFrame
        DataFrame returned from `fetch_salary_data`.
    stats_df : pandas.DataFrame
        DataFrame returned from `fetch_hitting_stats`.

    Returns
    -------
    pandas.DataFrame
        Combined dataset containing salaries, performance metrics and a
        lower‑case name key used for merging.
    """
    salary_df = salary_df.copy()
    stats_df = stats_df.copy()
    salary_df["name_key"] = salary_df["player"].str.lower().str.strip()
    stats_df["name_key"] = stats_df["player"].str.lower().str.strip()
    merged = pd.merge(
        salary_df,
        stats_df,
        how="inner",
        on="name_key",
        suffixes=("_salary", "_stats"),
    )
    merged = merged.drop(columns=["name_key"])
    merged = merged.drop_duplicates(subset=["player_salary"])
    return merged


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def cluster_players(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Standardise selected performance metrics and assign players to clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined dataset returned from `build_dataset`.
    n_clusters : int, optional
        Number of clusters to form (default 3).
    random_state : int, optional
        Random state for k‑means reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional `cluster` column containing integer
        cluster labels.
    """
    metrics = ["home_runs", "strikeouts", "avg", "obp", "ops", "stolen_bases", "hits"]
    X = df[metrics].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["cluster"] = clusters
    df.attrs["scaler"] = scaler
    df.attrs["kmeans"] = kmeans
    df.attrs["metrics"] = metrics
    return df


def label_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Assign descriptive labels (Power, Contact, Speed) to clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a `cluster` column from `cluster_players`.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an additional `archetype` column containing the
        assigned label.
    """
    metrics_summary = df.groupby("cluster")[
        ["home_runs", "strikeouts", "avg", "obp", "stolen_bases", "hits"]
    ].mean()
    cluster_labels = {}
    power_cluster = (metrics_summary["home_runs"] + metrics_summary["strikeouts"]).idxmax()
    cluster_labels[power_cluster] = "Power"
    contact_cluster = (metrics_summary["avg"] + metrics_summary["obp"]).idxmax()
    cluster_labels[contact_cluster] = "Contact"
    speed_cluster = (metrics_summary["stolen_bases"] + metrics_summary["hits"]).idxmax()
    cluster_labels[speed_cluster] = "Speed"
    for c in df["cluster"].unique():
        if c not in cluster_labels:
            remaining_labels = {"Power", "Contact", "Speed"} - set(cluster_labels.values())
            cluster_labels[c] = remaining_labels.pop()
    df = df.copy()
    df["archetype"] = df["cluster"].map(cluster_labels)
    return df


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def plot_salary_boxplot(df: pd.DataFrame) -> None:
    """Display a box plot comparing salaries across archetypes."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="archetype", y="salary", data=df)
    sns.stripplot(x="archetype", y="salary", data=df, color="black", alpha=0.3, jitter=0.2)
    plt.title("Salary distribution by player archetype")
    plt.ylabel("Salary (USD)")
    plt.xlabel("Player archetype")
    plt.show()


# -----------------------------------------------------------------------------
# Salary prediction model
# -----------------------------------------------------------------------------

def train_salary_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, List[str], pd.Series]:
    """Train a regression model to predict salary based on performance metrics."""
    numeric_cols = ["home_runs", "strikeouts", "avg", "obp", "ops", "stolen_bases", "hits"]
    df_encoded = pd.get_dummies(df, columns=["archetype"], drop_first=False)
    archetype_cols = [col for col in df_encoded.columns if col.startswith("archetype_")]
    feature_cols = numeric_cols + archetype_cols
    X = df_encoded[feature_cols]
    y = df_encoded["salary"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    logger.info("Validation MAE for salary prediction: %s", mae)
    return model, feature_cols, y


__all__ = [
    "fetch_salary_data",
    "parse_salary_html",
    "fetch_hitting_stats",
    "build_dataset",
    "cluster_players",
    "label_clusters",
    "plot_salary_boxplot",
    "train_salary_model",
]