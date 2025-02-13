import sqlite3
from modules.database import *

def db_text_result_search(input):
    connection = db_connection()
    cursor = db_cursor(connection)
    
    query = """
    SELECT * FROM file_scan_data 
    WHERE faster_whisper_result LIKE ?
    """

    cursor.execute(query, (f"%{input}%",))
    output = cursor.fetchall()
    connection.close()

    return output

def db_print_file_scan_data_rows(rows) :
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