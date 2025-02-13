import os
from datetime import datetime
import json
from modules.app_types import TextResult, FileScanData, FileHash, FileScanDataSQL  # Import TextResult from the other file
import sqlite3

def create_scan_log_table(cur):
    cur.execute(
        """CREATE TABLE scan_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        start_timestamp TEXT NOT NULL,
        end_timestamp TEXT NOT NULL,
        caller TEXT NOT NULL,
        event_type TEXT NOT NULL,
        processed_files INTEGER NOT NULL,
        total_time_sec REAL NOT NULL,
        vosk_scans INTEGER NOT NULL,
        vosk_time_sec REAL NOT NULL,
        faster_whisper_scans INTEGER NOT NULL,
        faster_whisper_time_sec REAL NOT NULL,
        tesseract_scans INTEGER NOT NULL,
        tesseract_time_sec REAL NOT NULL,
        easyocr_scans INTEGER NOT NULL,
        easyocr_time_sec REAL NOT NULL,
        non_scan_time_sec REAL NOT NULL,
        failures BLOB
        )""")
    return

def create_failures_table(cur):
    cur.execute(
        """CREATE TABLE scan_failures(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        caller TEXT NOT NULL,
        file_path TEXT NOT NULL,
        engines BLOB
        )"""
    )
    return

def create_file_scan_data_table(cur):
    cur.execute(
        """CREATE TABLE file_scan_data(
        file_path TEXT NOT NULL,
        directory TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_ext TEXT NOT NULL,
        meta_hash TEXT NOT NULL,
        data_hash TEXT NOT NULL,
        vosk_result TEXT,
        vosk_timestamp TEXT,
        faster_whisper_result TEXT,
        faster_Whisper_timestamp TEXT,
        tesseract_result TEXT,
        tesseract_timestamp TEXT,
        easyocr_result TEXT,
        easyocr_timestamp TEXT,
        biometric_data BLOB,
        vault_tags TEXT,
        tags TEXT,
        user_data BLOB
        )""")
    return

def insert_scan_log(cur, log):
    return

def insert_failures(cur, failures):
    return

def insert_file_scan_data(cur, file_scan_data):

    formatted = FileScanDataSQL(file_scan_data)

    try:
        cur.execute(
            """
            INSERT INTO file_scan_data (
                file_path, directory, file_name, file_ext, meta_hash, data_hash,
                vosk_result, vosk_timestamp, faster_whisper_result, faster_whisper_timestamp,
                tesseract_result, tesseract_timestamp, easyocr_result, easyocr_timestamp,
                biometric_data, vault_tags, tags, user_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                formatted.file_path,
                formatted.directory,
                formatted.file_name,
                formatted.file_ext,
                formatted.meta_hash,
                formatted.data_hash,
                formatted.vosk_result,
                formatted.vosk_timestamp,
                formatted.faster_whisper_result,
                formatted.faster_whisper_timestamp,
                formatted.tesseract_result,
                formatted.tesseract_timestamp,
                formatted.easyocr_result,
                formatted.easyocr_timestamp,
                None, # biometric data omitted for now
                formatted.vault_tags,
                formatted.tags,
                None # user data omitted for now
            )
        )

        cur.connection.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")


def print_all_file_scan_data(cur):

    cur.execute("SELECT * FROM file_scan_data")
    
    rows = cur.fetchall()
    
    for row in rows:
        print("File Path:", row[0])
        print("Directory:", row[1])
        print("File Name:", row[2])
        print("File Extension:", row[3])
        print("Meta Hash:", row[4])
        print("Data Hash:", row[5])
        print("Vosk Result:", row[6])
        print("Vosk Timestamp:", row[7])
        print("Faster Whisper Result:", row[8])
        print("Faster Whisper Timestamp:", row[9])
        print("Tesseract Result:", row[10])
        print("Tesseract Timestamp:", row[11])
        print("EasyOCR Result:", row[12])
        print("EasyOCR Timestamp:", row[13])
        print("Biometric Data:", row[14])
        print("Vault Tags:", row[15])
        print("Tags:", row[16])
        print("User Data:", row[17])
        print("-" * 40)  # Separator for readability

def db_connection(database = "scan_logs.db"):
    return sqlite3.connect(database)

def db_cursor(connection):
    return connection.cursor()

def main():
    connection = db_connection()
    cursor = db_cursor()
    print_all_file_scan_data(cursor)

if __name__ == "__main__":
    main()
