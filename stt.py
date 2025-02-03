
# note fix issue with rust package manager "cargo" and installing openAI-whisper with "py -m pip install openai-whisper"

# note VOSK models
# https://alphacephei.com/vosk/models


import argparse
import os
import io
import sqlite3
from modules.app_types import TextResult, FileHash, FileScanData
from modules.string_comparison import _levenshtein_similarity, _sequence_matcher
from modules.app_colors import *
from modules.app_hash import get_hashes_for_file
from modules.app_info import *
from modules.config_handler import load_config
import time
from enum import Enum
import numpy as np
import json
from datetime import datetime
import wave
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel
import speech_recognition as sr
import contextlib
import struct

from faster_whisper import WhisperModel

PATH_VOSK_LIGHT = r"models\vosk-model-small-en-us-0.15"
PATH_VOSK_GENERIC = r"models\vosk-model-en-us-0.22"
PATH_VOSK_GENERIC_DYNAMIC = r"models\vosk-model-en-us-0.22-lgraph"
PATH_VOSK_GIGA = r"models\vosk-model-en-us-0.42-gigaspeech"
PATH_VOSK_DAANZU = r"models\vosk-model-en-us-daanzu-20200905"

SetLogLevel(-1)

config = load_config()

total_processed_files_counter = 0
initialization_time = time.time()
initialization_datetime = datetime.now().isoformat()

total_speech_recognition_scans_counter = 0
total_speech_recognition_time = 0
total_faster_whisper_scans_counter = 0
total_faster_whisper_time = 0
total_vosk_scans_counter = 0
total_vosk_time = 0

LOG_RESULTS = config.getboolean('STT', 'log_results') if config.has_option('STT', 'log_results') else True
LOG_INDIVIDUAL_SCAN_TIMESTAMP = config.getboolean('STT', 'log_individual_scan_timestamp') if config.has_option('STT', 'log_individual_scan_timestamp') else False
VOSK_MODEL = int(config['STT']['vosk_model']) if config.has_option('SST', 'vosk_model') else 1
FASTER_WHISPER_MODEL = int(config['STT']['faster_whisper_model']) if config.has_option('SST', 'faster_whisper_model') else 1

print(f"[{initialization_datetime}] Started STT scan")

# Argument parser setup
parser = argparse.ArgumentParser(description="Perform STT on selected file or directory with multiple STT engines")
parser.add_argument('-a', '--action_a', type=str, help="Perform STT on the selected file with all available engines")
parser.add_argument('-s', '--action_s', type=str, help="Perform STT on the selected file with speech_recognition")
parser.add_argument('-w', '--action_w', type=str, help="Perform STT on the selected file with faster_whisper")
parser.add_argument('-v', '--action_v', type=str, help="Perform STT on the selected file with vosk")
parser.add_argument('--recursive', action='store_true', help="Walk recursively through directories")
parser.add_argument('--info', action='store_true', help="Display information per selected flag")   ######### add automagically download the models u want thing maybe

args = parser.parse_args()

VOSK_ENGINE = "vosk_engine"
FASTER_WHISPER_ENGINE = "faster_whisper_engine"
SPEECH_RECOGNITION_ENGINE = "speech_recognition_engine"

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
    if(vosk_model == 1) :
        return PATH_VOSK_LIGHT
    elif(vosk_model == 2) :
        return PATH_VOSK_GENERIC
    elif(vosk_model == 3) :
        return PATH_VOSK_GENERIC_DYNAMIC
    elif(vosk_model == 4) :
        return PATH_VOSK_GIGA
    elif(vosk_model == 5) :
        return PATH_VOSK_DAANZU
    else:
        print(f"[{datetime.now().isoformat()}] invalid vosk model. reverting to light model.")
        return PATH_VOSK_LIGHT
    
def get_faster_whisper_model_string(faster_whisper_model_index):
    if(faster_whisper_model_index == 1) :
        return 'tiny'
    elif(faster_whisper_model_index == 2) :
        return 'base'
    elif(faster_whisper_model_index == 3) :
        return 'small'
    elif(faster_whisper_model_index == 4) :
        return 'medium'
    elif(faster_whisper_model_index == 5) :
        return 'large'
    else:
        print(f"[{datetime.now().isoformat()}] invalid faster_whisper model index. reverting to tiny model.")
        return 'tiny'

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

    return TextResult(text=transcription, engine_name=VOSK_ENGINE)

def transcribe_audio_with_speech_recognition(file_path):
    global total_speech_recognition_scans_counter
    global total_speech_recognition_time
    """
    Perform STT on a WAV file using the CMU Sphinx engine (offline).
    can add support for alternative cloud based solutions as well in the future.
    """
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

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)  # Load the audio file

        try:
            # Use CMU Sphinx for offline transcription
            transcription = recognizer.recognize_sphinx(audio)
            
        except sr.UnknownValueError:
            return "CMU Sphinx could not understand the audio"
        except sr.RequestError as e:
            return f"CMU Sphinx error; {e}"

        if file_path.endswith(".wav"):
            wav_buffer.close()  # Close file only if it was opened
        total_speech_recognition_scans_counter += 1
        elapsed_time = time.time() - start_time
        total_speech_recognition_time += elapsed_time
        time_string = str(elapsed_time)[0:6]
        log_scan_info("STT", SPEECH_RECOGNITION_ENGINE, file_path, time_string)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        failure_paths.append(file_path)
        return  None

    return TextResult(text=transcription, engine_name=SPEECH_RECOGNITION_ENGINE)

def transcribe_audio_with_faster_whisper(file_path, model):
    global total_faster_whisper_scans_counter
    global total_faster_whisper_time

    try:
        start_time = time.time()
        # The model takes in the audio file path and returns the transcription
        segments, info = model.transcribe(file_path, beam_size=5)  # Adjust beam size for accuracy

        # Concatenate the transcriptions from all segments
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        total_faster_whisper_scans_counter += 1
        elapsed_time = time.time() - start_time
        total_faster_whisper_time += elapsed_time
        time_string = str(elapsed_time)[0:6]
        log_scan_info("STT", FASTER_WHISPER_ENGINE, file_path, time_string)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        failure_paths.append(file_path)
        return  None
    
    return TextResult(text="str(transcription.strip())", engine_name="FASTER_WHISPER_ENGINE")
    


# Function to process a directory (recursively or not)
def process_directory(directory, recursive=False):
    global total_processed_files_counter
    

    vosk_model_path = ""
    if args.action_a or args.action_v :
        vosk_model_path = get_vosk_model_path(VOSK_MODEL)

    faster_whisper_model = WhisperModel(get_faster_whisper_model_string(FASTER_WHISPER_MODEL))
    # Load the model
    #if args.action_a or args.action_w :
    #    faster_whisper_model = WhisperModel(get_faster_whisper_model_string(FASTER_WHISPER_MODEL)) # Choose the model size: 'tiny', 'base', 'small', 'medium', 'large'

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.mp3', '.wave', '.wav')): 
                file_scan_results = []
                if args.action_v or args.action_a:
                    file_scan_results.append(transcribe_audio_with_vosk(file_path, vosk_model_path))
                if args.action_s or args.action_a:
                    file_scan_results.append(transcribe_audio_with_speech_recognition(file_path))
                if args.action_w or args.action_a:
                    file_scan_results.append(transcribe_audio_with_faster_whisper(file_path, faster_whisper_model))
                total_processed_files_counter += 1
                results.append(FileScanData(file_path=file_path, hashes=get_hashes_for_file(file_path), text_results=file_scan_results))

        if not recursive:
            break  # Stop recursion if not set to walk recursively

    

# display info
if args.info :
    if(args.action_v):
        print(sst_vosk_info())
    else:
        print(stt_info())

# process
try:
    if args.action_a :
        if os.path.isfile(args.action_a) :
            file_scan_results = []
            file_scan_results.append(transcribe_audio_with_vosk(args.action_a, get_vosk_model_path(VOSK_MODEL)))
            file_scan_results.append(transcribe_audio_with_speech_recognition(args.action_a))
            file_scan_results.append(transcribe_audio_with_faster_whisper(args.action_a, WhisperModel(get_faster_whisper_model_string(FASTER_WHISPER_MODEL))))
            results.append(FileScanData(file_path=args.action_a, hashes=get_hashes_for_file(args.action_a), text_results=file_scan_results))
        elif os.path.isdir(args.action_a) :
            process_directory(directory=args.action_a, recursive=args.recursive)
    elif args.action_v :
        if os.path.isfile(args.action_v) :
            results.append(FileScanData(file_path=args.action_v, hashes=get_hashes_for_file(args.action_v), text_results=[transcribe_audio_with_vosk(args.action_v)]))
        elif os.path.isdir(args.action_v) :
            process_directory(directory=args.action_v, recursive=args.recursive)
    elif args.action_s :
        if os.path.isfile(args.action_s) :
            results.append(FileScanData(file_path=args.action_s, hashes=get_hashes_for_file(args.action_s), text_results=[transcribe_audio_with_speech_recognition(args.action_s)]))
        elif os.path.isdir(args.action_s) :
            process_directory(directory=args.action_s, recursive=args.recursive)
    elif args.action_w :
        if os.path.isfile(args.action_w) :
            results.append(FileScanData(file_path=args.action_w, hashes=get_hashes_for_file(args.action_w), text_results=[transcribe_audio_with_faster_whisper(args.action_w, WhisperModel(get_faster_whisper_model_string(FASTER_WHISPER_MODEL)))]))
        elif os.path.isdir(args.action_w) :
            process_directory(directory=args.action_w, recursive=args.recursive)
    else:
        print("Invalid entry")

    print("done") ###################################################################### <-------- why it not getting there? 
    for it in results:
        print("printing")
        if(it == None):
            continue
        else:
            json_string = it.serialize()
            print(json_string)
except Exception as e:
    print(f"Error processing : {e}")


total_elapsed_time = time.time() - initialization_time
total_scans = total_faster_whisper_scans_counter + total_speech_recognition_scans_counter + total_vosk_scans_counter
non_scan_time = total_elapsed_time - (total_vosk_time + total_speech_recognition_time + total_faster_whisper_time)
current_iso_time = datetime.now().isoformat()

# Human-readable console output
print(f"[{current_iso_time}] SCAN SUMMARY")
print(f"[{current_iso_time}] Processed {total_processed_files_counter} files performing {total_scans} in {total_elapsed_time:.3f}s")
print(f"[{current_iso_time}] Vosk: {total_vosk_scans_counter} scans ({total_vosk_time:.3f}s)")
print(f"[{current_iso_time}] Speech_Recognition: {total_speech_recognition_scans_counter} scans ({total_speech_recognition_time:.3f}s)")
print(f"[{current_iso_time}] Faster_Whisper: {total_faster_whisper_scans_counter} scans ({total_faster_whisper_time:.3f}s)")
print(f"[{current_iso_time}] Non-scan time: {non_scan_time:.3f}s")
print(f"[{current_iso_time}] Failures ({len(failure_paths)}):")
for path in failure_paths:
    print(f"    {path}")

# Structured JSON log (save to file instead of printing)
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "start_timestamp": initialization_datetime,
    "end_timestamp": current_iso_time,
    "event_type": "scan_summary",
    "metrics": {
        "processed_files": total_processed_files_counter,
        "total_time_sec": round(total_elapsed_time, 3),
        "vosk": {
            "scans": total_vosk_scans_counter,
            "time_sec": round(total_vosk_time, 3)
        },
        "speech_recognition": {
            "scans": total_speech_recognition_scans_counter,
            "time_sec": round(total_speech_recognition_time, 3)
        },
        "faster_whisper": {
            "scans": total_faster_whisper_scans_counter,
            "time_sec": round(total_faster_whisper_time, 3)
        },
        "non_scan_time_sec": round((non_scan_time), 3)
    },
    "failures": failure_paths
}

# Save JSON to file (appends to log file)
with open("scan_logs.json", "a") as log_file:
    json.dump(log_entry, log_file)
    log_file.write("\n")