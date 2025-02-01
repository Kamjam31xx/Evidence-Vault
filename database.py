import os
import sqlite3
from datetime import datetime
import json
from app_types import TextResult  # Import TextResult from the other file

# SQLite database setup
DB_PATH = 'ocr_results.db'  # Path to the SQLite database file

# Create SQLite connection and cursor
def create_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create a table to store unprocessed text results if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unprocessed_text_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

# Function to insert a TextResult into the database
def insert_into_db(text_result: TextResult):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO unprocessed_text_results (text, engine_name, file_path, timestamp)
        VALUES (?, ?, ?, ?);
    ''', (text_result.text, text_result.engine_name, text_result.file_path, text_result.timestamp.isoformat()))
    conn.commit()
    conn.close()