import pyodbc
import pandas as pd
from config.db_config import DB_CONFIG

def get_connection():
    """Create database connection using Windows authentication"""
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"Trusted_Connection={DB_CONFIG['trusted_connection']};"
    )
    return pyodbc.connect(conn_str)

def load_churn_data():
    """Load churn training features from SQL Server view"""
    try:
        query = "SELECT * FROM vw_churn_training_features"
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_churn_data()
    if not df.empty:
        print("\nFirst 5 rows:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Churn distribution:\n{df['churn'].value_counts(normalize=True)}")