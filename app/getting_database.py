from sqlalchemy import create_engine
import pandas as pd


def get_event_data_from_database():
    # Buat koneksi ke database
    engine = create_engine(
        'postgres://postgres:adminch2ps374@34.128.99.62:5432/pg_testdb?sslmode=disable ')

    # Query data dari tabel event
    query = "SELECT * FROM event"

    # Baca data ke DataFrame
    df = pd.read_sql_query(query, engine)

    return df
