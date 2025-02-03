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
    