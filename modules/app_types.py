import json
from datetime import datetime
import os

class Dimensions:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Serializable:
    def serialize(self, format: str = "json") -> str:
        if format == "json":
            return json.dumps(self.to_dict(), indent=4)
        elif format == "str":
            return str(self.to_dict())
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

class FileHash(Serializable):
    def __init__(self, meta_hash, data_hash):
        self.meta_hash = meta_hash
        self.data_hash = data_hash

    def to_dict(self) -> dict:
        return {
            "meta_hash": self.meta_hash,
            "data_hash": self.data_hash
        }
        
    def __repr__(self) -> str:
        return f"""
        FileHash(
            meta_hash={self.meta_hash}, "
            data_hash={self.data_hash}
        )"""
    
    

# WARNING : AI generated class 
class TextResult(Serializable):
    def __init__(self, text: str, engine_name: str):
        self.text = text  # String member for the extracted text
        self.engine_name = engine_name  # String member for the OCR engine name
        self.timestamp = datetime.now()  # Timestamp of when the object was created

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "engine_name": self.engine_name,
            "timestamp": self.timestamp.isoformat()  # Convert timestamp to string
        }

    def __repr__(self) -> str:

        return f"""
        TextResult(
            text={self.text},"
            engine_name={self.engine_name}, "
            timestamp={self.timestamp}
        )"""
    
class FileScanData(Serializable):
    def __init__(self, file_path, hashes, text_results=[], biometric_data=[], vault_tags=[], tags=[], user_data=[]) :
        self.file_path = file_path
        self.hashes = hashes
        self.text_results = text_results
        self.biometric_data = biometric_data
        self.vault_tags = vault_tags
        self.tags = tags
        self.user_data = user_data

    def to_dict(self) -> dict:
        return {
            "file_path": os.path.abspath(self.file_path),
            "directory": os.path.dirname(self.file_path),
            "file_name": os.path.basename(self.file_path),
            "file_ext": os.path.splitext(self.file_path)[1].lower(),
            "meta_hash": self.hashes.meta_hash,
            "data_hash": self.hashes.data_hash,
            "text_results": [tr.to_dict() for tr in self.text_results] if self.text_results else [],
            "biometric_data": self.biometric_data,
            "vault_tags": self.vault_tags,
            "tags": self.tags,
            "user_data": self.user_data
        } 
    
    def __repr__(self) -> str:
        return f"""
        FileScanData(
            file_path={self.file_path}, 
            hashes={self.hashes}, 
            text_results={self.text_results}, 
            biometric_data={self.biometric_data}, 
            vault_tags={self.vault_tags}, 
            tags={self.tags}, 
            user_data={self.user_data}
        )"""
    
class FileScanDataSQL:
    def __init__(self, file_scan_data):
        
        self.file_path = os.path.abspath(file_scan_data.file_path)
        self.directory = os.path.dirname(file_scan_data.file_path)
        self.file_name = os.path.basename(file_scan_data.file_path)
        self.file_ext = os.path.splitext(file_scan_data.file_path)[1].lower()
        self.meta_hash = file_scan_data.hashes.meta_hash
        self.data_hash = file_scan_data.hashes.data_hash

        self.vosk_result = None
        self.vosk_timestamp = None
        self.faster_whisper_result = None
        self.faster_whisper_timestamp = None
        self.tesseract_result = None
        self.tesseract_timestamp = None
        self.easyocr_result = None
        self.easyocr_timestamp = None

        for it in file_scan_data.text_results:
            if it.engine_name == "vosk":
                try:
                    self.vosk_result = it.text
                    self.vosk_timestamp = it.timestamp
                except Exception as e:
                    print(f"Error : {e}")
            elif it.engine_name == "faster_whisper":
                try:
                    self.faster_whisper_result = it.text
                    self.faster_whisper_timestamp = it.timestamp
                except Exception as e:
                    print(f"Error : {e}")
            elif it.engine_name == "tesseract":
                try:
                    self.tesseract_result = it.text
                    self.tesseract_timestamp = it.timestamp
                except Exception as e:
                    print(f"Error : {e}")
            elif it.engine_name == "easyocr":  # Fixed typo here (engime_name → engine_name)
                try:
                    self.easyocr_result = it.text
                    self.easyocr_timestamp = it.timestamp
                except Exception as e:
                    print(f"Error : {e}")
            else:
                print(f"Error : invalid key<{it.engine_name}> for engine_name in TextResult")

        self.biometric_data = file_scan_data.biometric_data

        self.vault_tags = ", ".join(file_scan_data.vault_tags) if file_scan_data.vault_tags else ""
        self.tags = ", ".join(file_scan_data.tags) if file_scan_data.tags else ""

        self.user_data = file_scan_data.user_data


