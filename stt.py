
# note fix issue with rust package manager "cargo" and installing openAI-whisper with "py -m pip install openai-whisper"

# note VOSK models
# https://alphacephei.com/vosk/models


import argparse
import os
import io
import sqlite3
from modules.app_types import TextResult, FileDataHash
import time
import numpy as np
from modules.string_comparison import _levenshtein_similarity, _sequence_matcher
from modules.app_colors import *
import json
from datetime import datetime
from modules.config_handler import load_config
import wave
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel
from enum import Enum
from modules.app_info import *
import contextlib
import struct
from modules.app_hash import get_hashes_for_file


class VoskModel(Enum):
    LIGHT = 1
    GENERIC = 2
    GENERIC_DYNAMIC = 3

PATH_VOSK_LIGHT = r"models\vosk-model-small-en-us-0.15"
PATH_VOSK_GENERIC = r"models\vosk-model-en-us-0.22"
PATH_VOSK_GENERIC_DYNAMIC = r"models\vosk-model-en-us-0.22-lgraph"

SetLogLevel(-1)

config = load_config()

total_processed_files_counter = 0

initialization_time = time.time()
initialization_datetime = datetime.now().isoformat()

total_speech_recognition_scans_counter = 0
total_speech_recognition_time = 0
total_pydub_scans_counter = 0
total_pydub_time = 0
total_vosk_scans_counter = 0
total_vosk_time = 0

LOG_RESULTS = config.getboolean('STT', 'log_results') if config.has_option('STT', 'log_results') else False
LOG_INDIVIDUAL_SCAN_TIMESTAMP = config.getboolean('STT', 'log_individual_scan_timestamp') if config.has_option('STT', 'log_individual_scan_timestamp') else False
VOSK_MODEL = VoskModel(int(config['STT']['vosk_model'])) if config.has_option('SST', 'vosk_model') else VoskModel(1)

print(f"[{initialization_datetime}] Started STT scan")

# Argument parser setup
parser = argparse.ArgumentParser(description="Perform STT on selected file or directory with multiple STT engines")
parser.add_argument('-a', '--action_a', type=str, help="Perform STT on the selected file with all available engines")
parser.add_argument('-s', '--action_s', type=str, help="Perform STT on the selected file with speech_recognition")
parser.add_argument('-p', '--action_p', type=str, help="Perform STT on the selected file with pydub")
parser.add_argument('-v', '--action_v', type=str, help="Perform STT on the selected file with vosk")
parser.add_argument('--recursive', action='store_true', help="Walk recursively through directories")
parser.add_argument('--info', action='store_true', help="Display information per selected flag")   ######### add automagically download the models u want thing maybe

args = parser.parse_args()

VOSK_ENGINE = "vosk_engine"


results = []
failure_paths = []

def log_scan_info(operation, engine, path, duration) :
    output = ""
    if(LOG_INDIVIDUAL_SCAN_TIMESTAMP):
        output += f"[{datetime.now().isoformat()}] "
    output += f"<{operation}> {engine} \"{path}\" {duration} seconds"
    if(LOG_RESULTS) :
        print(output)

def get_vosk_model_path(vosk_model):
    if(vosk_model == VoskModel(1)) :
        return PATH_VOSK_LIGHT
    elif(vosk_model == VoskModel(2)) :
        return PATH_VOSK_GENERIC
    elif(vosk_model == VoskModel(3)) :
        return PATH_VOSK_GENERIC_DYNAMIC
    else:
        print(f"[{datetime.now().isoformat()}] invalid vosk model. reverting to light model.")
        return PATH_VOSK_LIGHT

def convert_mp3_to_16bit_pcm_mono_wav_in_memory(mp3_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # Ensure PCM 16-bit mono

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)  # Reset buffer position for reading
        return wav_buffer
    except Exception as e:
        print(f"Error converting {mp3_file}: {e}")
        return None

def convert_wav_to_16bit_pcm_mono_wav_in_memory(wav_file):
    try:
        audio = AudioSegment.from_wav(wav_file)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # Ensure PCM 16-bit mono

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav", parameters=["-acodec", "pcm_s16le"])
        wav_buffer.seek(0)  # Reset buffer position for reading

        return wav_buffer
    except Exception as e:
        print(f"Error converting {wav_file}: {e}")
        return None
    
def extract_pcm_from_xwav(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    # Find the 'data' chunk where raw PCM starts
    data_offset = data.find(b'data')
    if data_offset == -1:
        raise ValueError("No 'data' chunk found. This may not be an XWAV file.")

    # Read chunk size (little-endian uint32)
    data_size = struct.unpack("<I", data[data_offset+4:data_offset+8])[0]
    pcm_data = data[data_offset+8:data_offset+8+data_size]

    # Wrap PCM data in a RIFF WAV container (16-bit, mono, little-endian)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_out:
        wav_out.setnchannels(1)  # Mono
        wav_out.setsampwidth(2)  # 16-bit
        wav_out.setframerate(16000)  # XWAVs often use 16kHz, change if needed
        wav_out.writeframes(pcm_data)

    wav_buffer.seek(0)  # Reset buffer position
    return wav_buffer


def transcribe_audio_with_vosk_from_memory(audio_buffer, model_path):
    model = Model(model_path)

    with wave.open(audio_buffer, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
            print("Audio file must be WAV format mono PCM, 16-bit, 8000 or 16000 Hz.")
            return None
        
        rec = KaldiRecognizer(model, wf.getframerate())
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = rec.Result()
                results.append(json.loads(result)["text"])
        final_result = rec.FinalResult()
        results.append(json.loads(final_result)["text"])
    return " ".join(results)

def transcribe_audio_with_vosk(file_path, model_path):
    global total_vosk_scans_counter
    global total_vosk_time

    try :
        start_time = time.time()
        if file_path.endswith(".mp3"):
            wav_buffer = convert_mp3_to_16bit_pcm_mono_wav_in_memory(file_path)
        elif file_path.endswith(".wav") or file_path.endswith(".wave"):
            wav_buffer = open(file_path, "rb")  # Read WAV file directly
            header = wav_buffer.read(12)
            wav_buffer.seek(0)
            if header[:4] == b'RIFF':
                wav_buffer = convert_wav_to_16bit_pcm_mono_wav_in_memory(file_path)
            elif header[:4] == b'FFIR':
                wav_buffer = extract_pcm_from_xwav(file_path)
        else:
            print("Unsupported file format. Please provide a .wav or .mp3 file.")
            return None

        transcription = transcribe_audio_with_vosk_from_memory(wav_buffer, model_path)

        if file_path.endswith(".wav"):
            wav_buffer.close()  # Close file only if it was opened
        total_vosk_scans_counter += 1
        elapsed_time = time.time() - start_time
        total_vosk_time += elapsed_time
        time_string = str(elapsed_time)[0:6]
        log_scan_info("STT", VOSK_ENGINE, file_path, time_string)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        failure_paths.append(file_path)
        return  None

    return transcription

# Function to process a directory (recursively or not)
def process_directory(directory, recursive=False, vosk_model_path=None):
    global total_processed_files_counter
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.mp3', '.wave', '.wav')): 
                if args.action_v:
                    results.append((transcribe_audio_with_vosk(file_path, vosk_model_path), get_hashes_for_file(file_path)))
                total_processed_files_counter += 1
        if not recursive:
            break  # Stop recursion if not set to walk recursively



# display info
if args.info :
    if(args.action_v):
        print(sst_vosk_info())
    else:
        print(stt_info())

# process with vosk
if args.action_v :
    vosk_model_path = get_vosk_model_path(VOSK_MODEL)
    if os.path.isfile(args.action_v) :
        results.append(transcribe_audio_with_vosk(args.action_v, vosk_model_path))
    elif os.path.isdir(args.action_v) :
        process_directory(directory=args.action_v, recursive=args.recursive, vosk_model_path=vosk_model_path)
    
if args.action_s :
    print()

if args.action_p :
    print()

for (it, hash) in results:
    print( hash, "    SST Result : ",it)