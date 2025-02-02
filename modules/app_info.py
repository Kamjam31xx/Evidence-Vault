def stt_info():
    info_text = """
    Speech-to-Text (STT) Script
    ===========================
    This script performs Speech-to-Text (STT) on audio files using multiple STT engines:
    - speech_recognition
    - pydub
    - vosk

    Usage:
    ------
    python script.py [options]

    Options:
    --------
    -a, --action_a <file>  Perform STT on the selected file with all available engines.
    -s, --action_s <file>  Perform STT on the selected file with speech_recognition.
    -p, --action_p <file>  Perform STT on the selected file with pydub.
    -v, --action_v <file>  Perform STT on the selected file with vosk.
    --recursive            Walk recursively through directories.
    --info                 Display this information.

    Example:
    --------
    python script.py -a input.wav
    python script.py -s input.mp3 --recursive
    python script.py --info
    """
    
    return info_text

def sst_vosk_info() :
    info_text = f"""
    Vosk (STT)
    ===========================
    Performs speech to text on audio files.
    
    Available models:
    ------
    [1] Light English
    [2] Generic English
    [3] Generic Dynamic English

    Selecting Model:
    --------
    Edit the corresponding entry "vosk_model" in the evidence_vault.cfg 
    file to select a model with an integer listed above.
    """
    
    return info_text