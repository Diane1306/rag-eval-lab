# src/bq_query_metrics.py
#
# Purpose:
#   Query BigQuery for experiment KPIs and return as pandas DataFrames.

import pandas as pd
from google.cloud import bigquery

def query_df(sql: str) -> pd.DataFrame:
    """Run a BigQuery SQL query and return a pandas DataFrame."""
    client = bigquery.Client()
    return client.query(sql).to_dataframe()
