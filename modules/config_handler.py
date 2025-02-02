import configparser
import os

def load_config(config_path='evidence_vault.cfg'):
    """Load configuration from file"""
    config = configparser.ConfigParser()
    
    # Set default values if config file is missing
    config.read_dict({
        'OCR': {
            'visualize': 'false',
            'diagnostic_logging': 'false',
            'log_results': 'false',
            'log_individual_scan_timestamp': 'false',
            'tesseract_path': "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        },
        'STT': {
            'log_results': 'false',
            'log_individual_scan_timestamp': 'false',
            'vosk_model': '1'
        },
    })
    
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")
        
    return config