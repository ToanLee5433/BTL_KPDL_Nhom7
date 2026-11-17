"""Local SQLite storage. CSV hashes prevent accidental duplicate batch imports."""
from pathlib import Path
from contextlib import contextmanager
import os
import sqlite3
import pandas as pd
from pipeline import ROOT, SOURCE, COLUMNS, validate, data_fingerprint

DB = Path(os.environ.get('WHOLESALE_DB', str(ROOT / 'data' / 'customers.sqlite3')))

@contextmanager
def connect(db=DB):
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db, timeout=30)
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS batches (digest TEXT PRIMARY KEY, source TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP)')
        columns = ', '.join(f'"{c}" REAL NOT NULL' for c in COLUMNS)
        connection.execute(f'CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, batch TEXT NOT NULL, {columns})')
        connection.commit()
        with connection:
            yield connection
    finally:
        connection.close()

def add_batch(raw, source, db=DB):
    frame = validate(raw)
    digest = data_fingerprint(frame)
    with connect(db) as conn:
        # Serialize the duplicate check and insert across local writers.
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM batches WHERE digest=?', (digest,)).fetchone():
            return 0
        conn.execute('INSERT INTO batches(digest, source) VALUES (?, ?)', (digest, source))
        names = ', '.join(['batch'] + [f'"{c}"' for c in COLUMNS])
        placeholders = ','.join('?' for _ in range(len(COLUMNS)+1))
        conn.executemany(f'INSERT INTO customers({names}) VALUES ({placeholders})',
                         [(digest, *row) for row in frame.itertuples(index=False, name=None)])
    return len(frame)

def initialize(db=DB):
    with connect(db) as conn:
        # Preserve existing databases created by earlier fingerprint formats.
        if conn.execute('SELECT 1 FROM batches WHERE source=?', ('UCI Wholesale Customers 292',)).fetchone():
            return 0
    return add_batch(pd.read_csv(SOURCE), 'UCI Wholesale Customers 292', db)

def load_data(db=DB):
    with connect(db) as conn:
        frame = pd.read_sql_query('SELECT * FROM customers ORDER BY id', conn)
    return validate(frame)

def history(db=DB):
    with connect(db) as conn:
        return pd.read_sql_query('SELECT source, created, digest FROM batches ORDER BY created DESC', conn)