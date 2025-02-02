import json
from datetime import datetime

class Dimensions:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class FileDataHash:
    def __init__(self, meta_hash, data_hash):
        self.meta_hash = meta_hash
        self.data_hash = data_hash

    def to_dict(self) -> dict:
        return {
            "meta_hash": self.meta_hash,
            "data_hash": self.data_hash
        }
    
    def serialize(self, format: str = "json") -> str:
        if format == "json":
            return json.dumps(self.to_dict(), indent=4)
        elif format == "str":
            return str(self.to_dict())
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
        
    def __repr__(self) -> str:
        return f"DataHash(meta_hash={self.meta_hash}, data_hash={self.data_hash})"
    
    

# WARNING : AI generated class 
class TextResult:
    def __init__(self, text: str, engine_name: str, file_path: str):
        self.text = text  # String member for the extracted text
        self.engine_name = engine_name  # String member for the OCR engine name
        self.file_path = file_path  # String member for the file path
        self.timestamp = datetime.now()  # Timestamp of when the object was created

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "engine_name": self.engine_name,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat()  # Convert timestamp to string
        }

    def serialize(self, format: str = "json") -> str:
        if format == "json":
            return json.dumps(self.to_dict(), indent=4)
        elif format == "str":
            return str(self.to_dict())
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    def __repr__(self) -> str:
        return f"TextResult(text={self.text}, engine_name={self.engine_name}, file_path={self.file_path}, timestamp={self.timestamp})"
    
