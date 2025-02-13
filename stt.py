
# note fix issue with rust package manager "cargo" and installing openAI-whisper with "py -m pip install openai-whisper"

# note VOSK models
# https://alphacephei.com/vosk/models



import os
import io
import time
import contextlib
import struct
from enum import Enum
from datetime import datetime
import json
import wave
import ffmpeg
from modules.app_types import TextResult, FileHash, FileScanData
from modules.string_comparison import _levenshtein_similarity, _sequence_matcher
from modules.app_colors import *
from modules.app_hash import get_hashes_for_file
from modules.app_info import *
from modules.config_handler import load_config
from modules.database import *
import argparse
import numpy as np
import sqlite3
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel
from faster_whisper import WhisperModel
from modules.database import *

total_processed_files_counter = 0
initialization_time = time.time()
initialization_datetime = datetime.now().isoformat()

total_faster_whisper_scans_counter = 0
total_faster_whisper_time = 0
total_vosk_scans_counter = 0
total_vosk_time = 0

# Vosk model setup
PATH_VOSK_LIGHT = r"models\vosk-model-small-en-us-0.15"
PATH_VOSK_GENERIC = r"models\vosk-model-en-us-0.22"
PATH_VOSK_GENERIC_DYNAMIC = r"models\vosk-model-en-us-0.22-lgraph"
PATH_VOSK_GIGA = r"models\vosk-model-en-us-0.42-gigaspeech"
PATH_VOSK_DAANZU = r"models\vosk-model-en-us-daanzu-20200905"
SetLogLevel(-1)
voskModel = Model(PATH_VOSK_GENERIC_DYNAMIC)


# Whisper model setup
WHISPER_MODEL_TINY = 'tiny'
WHISPER_MODEL_BASE = 'base'
WHISPER_MODEL_SMALL = 'small'
WHISPER_MODEL_MEDIUM = 'medium'
WHISPER_MODEL_LARGE = 'large'
whisperModel = WhisperModel(WHISPER_MODEL_SMALL)

config = load_config()

LOG_RESULTS = config.getboolean('STT', 'log_results') if config.has_option('STT', 'log_results') else True
LOG_INDIVIDUAL_SCAN_TIMESTAMP = config.getboolean('STT', 'log_individual_scan_timestamp') if config.has_option('STT', 'log_individual_scan_timestamp') else False
#VOSK_MODEL = int(config['STT']['vosk_model']) if config.has_option('SST', 'vosk_model') else 1
#FASTER_WHISPER_MODEL = int(config['STT']['faster_whisper_model']) if config.has_option('SST', 'faster_whisper_model') else 1

print(f"[{initialization_datetime}] Started STT scan")

# Argument parser setup
parser = argparse.ArgumentParser(description="Perform STT on selected file or directory with multiple STT engines")
parser.add_argument('-a', '--action_a', type=str, help="Perform STT on the selected file with all available engines")
parser.add_argument('-w', '--action_w', type=str, help="Perform STT on the selected file with faster_whisper")
parser.add_argument('-v', '--action_v', type=str, help="Perform STT on the selected file with vosk")
parser.add_argument('--recursive', action='store_true', help="Walk recursively through directories")
parser.add_argument('--info', action='store_true', help="Display information per selected flag")   ######### add automagically download the models u want thing maybe

args = parser.parse_args()

VOSK_ENGINE = "vosk"
FASTER_WHISPER_ENGINE = "faster_whisper"

results = []
failure_paths = []

def log_scan_info(operation, engine, path, duration) :
    output = ""
    if(LOG_INDIVIDUAL_SCAN_TIMESTAMP):
        output += f"[{datetime.now().isoformat()}] "
    output += f"<{operation}> {engine} \"{path}\" {duration} seconds"
    if(LOG_RESULTS) :
        print(output)

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

'''   
def extract_and_convert_mp4_to_16bit_pcm_mono_wav_in_memory(file_path):
    try:
        print("extracting mp4")
        output = io.BytesIO()
        process = (ffmpeg.input(file_path).output(
            'pipe:1', 
            ac=1, 
            ar='16000',  
            acodec='pcm_s16le',
            format='wav'
        ).run(pipe_stdout=True))
        output.write(process.stdout.read())
        

        pcm_data = (ffmpeg.input(file_path).output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k').overwrite_output().run(capture_stdout=True))

        #output = io.BytesIO(pcm_data)
        #output.write(pcm_data)
        #print(pcm_data)
        #output.seek(0)
        #output = np.frombuffer(pcm_data, dtype=np.int16)
        #print(len(np.frombuffer(pcm_data, dtype=np.int16)))
        #print(len(np.frombuffer(pcm_data, dtype=np.int16))/16000)
        #return output.view(output.dtype.newbyteorder('<'))
        for it in pcm_data:
            print(it)
        #wav_buffer = io.BytesIO()
        #with wave.open(wav_buffer, "wb") as wav_out:
        #    wav_out.setnchannels(1)  # Mono
        #    wav_out.setsampwidth(2)  # 16-bit
        #    wav_out.setframerate(16000)  # XWAVs often use 16kHz, change if needed
        #    wav_out.writeframes(pcm_data)
        #wav_buffer.seek(0)  # Reset buffer position
        #return wav_buffer
        return pcm_data # output.getvalue()
    except Exception as e:
        print(f"Error extracting and converting {file_path}: {e}")
'''
def extract_pcm_from_xwav(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    data_offset = data.find(b'data')
    if data_offset == -1:
        raise ValueError("No 'data' chunk found. This may not be an XWAV file.")

    data_size = struct.unpack("<I", data[data_offset+4:data_offset+8])[0]
    pcm_data = data[data_offset+8:data_offset+8+data_size]

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_out:
        wav_out.setnchannels(1)  # Mono
        wav_out.setsampwidth(2)  # 16-bit
        wav_out.setframerate(16000)  # XWAVs often use 16kHz, change if needed
        wav_out.writeframes(pcm_data)

    wav_buffer.seek(0)  # Reset buffer position
    return wav_buffer

def transcribe_audio_with_vosk_from_memory(audio_buffer):
    with wave.open(audio_buffer, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
            print("Audio file must be WAV format mono PCM, 16-bit, 8000 or 16000 Hz.")
            return None
        
        rec = KaldiRecognizer(voskModel, wf.getframerate())
        
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

def transcribe_audio_with_vosk(file_path):
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
        elif file_path.endswith(".mp4") :
            return None
            #wav_buffer = extract_and_convert_mp4_to_16bit_pcm_mono_wav_in_memory(file_path)
        else:
            print("Unsupported file format. Please provide a .wav or .mp3 file.")
            return None

        transcription = transcribe_audio_with_vosk_from_memory(wav_buffer)

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

def transcribe_audio_with_faster_whisper(file_path):
    global total_faster_whisper_scans_counter
    global total_faster_whisper_time

    try:
        start_time = time.time()
        
        modelInput = file_path
        #if file_path.endswith(".mp4") :
        #    modelInput = extract_and_convert_mp4_to_16bit_pcm_mono_wav_in_memory(file_path)

        segments, info = whisperModel.transcribe(audio=modelInput, beam_size=5)
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
    
    return TextResult(text=transcription.strip(), engine_name=FASTER_WHISPER_ENGINE)

def process_directory(directory, functions=None, recursive=False):
    global total_processed_files_counter
    if( functions == None ):
        return []
    scan_results = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.mp3', '.wave', '.wav', '.mp4')): 
                
                transcriptions = []
                for func in functions:
                    transcriptions.append(func(file_path))

                scan_results.append(FileScanData(file_path=file_path, hashes=get_hashes_for_file(file_path=file_path), text_results=transcriptions))
                total_processed_files_counter += 1

        if not recursive:
            break  # Stop recursion if not set to walk recursively
        
    return scan_results

def process_file(file_path, functions=None) :
    transcriptions = []
    for func in functions:
        transcriptions.append(func(file_path))
    return [FileScanData(file_path=file_path, hashes=get_hashes_for_file(file_path=file_path), text_results=transcriptions)]

# display info
if args.info :
    if(args.action_v):
        print(sst_vosk_info())
    elif(args.action_w):
        print("sst_faster_whisper_info()")
    else:
        print(stt_info())

# process

path = None
function_ptrs = [] 
if args.action_a :
    path = args.action_a
    function_ptrs = [transcribe_audio_with_faster_whisper, transcribe_audio_with_vosk]
elif args.action_v :
    path = args.action_v
    function_ptrs = [transcribe_audio_with_vosk]
elif args.action_w :
    path = args.action_w
    function_ptrs = [transcribe_audio_with_faster_whisper]
else:
    print("Invalid entry")

if(path != None) :
    try:
        if(os.path.isfile(path)) :
            results = process_file(path, functions=function_ptrs)
        else :
            results = process_directory(path, functions=function_ptrs, recursive=args.recursive)
    except Exception as e :
        print(f"Error processing directory {path}: {e}")

        
for it in results:
    if(it == None):
        continue
    else:
        json_string = it.serialize()
        print(json_string)


total_elapsed_time = time.time() - initialization_time
total_scans = total_faster_whisper_scans_counter + total_vosk_scans_counter
non_scan_time = total_elapsed_time - (total_vosk_time + total_faster_whisper_time)
current_iso_time = datetime.now().isoformat()

# Human-readable console output
print(f"[{current_iso_time}] SCAN SUMMARY")
print(f"[{current_iso_time}] Processed {total_processed_files_counter} files performing {total_scans} total scans in {total_elapsed_time:.3f}s")
print(f"[{current_iso_time}] Vosk: {total_vosk_scans_counter} scans ({total_vosk_time:.3f}s)")
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
    "caller": "stt.py",
    "metrics": {
        "processed_files": total_processed_files_counter,
        "total_time_sec": round(total_elapsed_time, 3),
        "vosk": {
            "scans": total_vosk_scans_counter,
            "time_sec": round(total_vosk_time, 3)
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

save_results = input("Save results? (Y/n): ").strip().lower()
if save_results == "y":
    print("opening connection to db...")
    connection = db_connection()
    cursor = db_cursor(connection)
    print("saving results...")
    for it in results:
        insert_file_scan_data(cur=cursor, file_scan_data=it)
    print("results saved")

print_results_from_db = input("view results from db? (Y/n): ").strip().lower()
if print_results_from_db == "y":
    print("opening connection to db...")
    connection = db_connection()
    cursor = db_cursor(connection)
    print("fetching results...")
    print_all_file_scan_data(cursor)
   

'''
save_log = input("Save log? (Y/n): ").strip().lower()
if save_log == "y":
    print("saving log...")
    # Save JSON to file (appends to log file)
    with open("scan_logs.json", "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")
    insert_scan_log(log_entry)
else:
    print("log discarded.")

print_logs = input("view db logs? (Y/n): ").strip().lower()
if print_logs == "y":
    print_scan_logs()'''