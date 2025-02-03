import argparse
import os
import sqlite3
from PIL import Image
import pytesseract
import easyocr
from modules.app_types import TextResult
import time
import numpy as np
from modules.string_comparison import _levenshtein_similarity, _sequence_matcher
import cv2
from modules.app_colors import *
import json
from datetime import datetime
from modules.config_handler import load_config
from modules.app_hash import get_hashes_for_file

config = load_config()

initialization_time = time.time()
initialization_datetime = datetime.now().isoformat()
total_processed_files_counter = 0
total_tesseract_scans_counter = 0
total_tesseract_time = 0
total_easyocr_scans_counter = 0
total_easyocr_time = 0

pytesseract.pytesseract.tesseract_cmd = os.path.expanduser(config['OCR']['tesseract_path']) if config.has_option('OCR', 'tesseract_path') else r'C:\Program Files\Tesseract-OCR\tesseract.exe'
VISUALIZE = config.getboolean('OCR', 'visualize') if config.has_option('OCR', 'visualize') else False
LOG_RESULTS = config.getboolean('OCR', 'log_results') if config.has_option('OCR', 'log_results') else True
LOG_INDIVIDUAL_SCAN_TIMESTAMP = config.getboolean('OCR', 'log_individual_scan_timestamp') if config.has_option('OCR', 'log_individual_scan_timestamp') else False

print(f"[{initialization_datetime}] Started OCR scan")



# Argument parser setup
parser = argparse.ArgumentParser(description="Perform OCR on selected file or directory with multiple OCR engines")
parser.add_argument('-a', '--action_a', type=str, help="Perform OCR on the selected file with all available engines")
parser.add_argument('-t', '--action_t', type=str, help="Perform OCR on the selected file with tesseract")
parser.add_argument('-e', '--action_e', type=str, help="Perform OCR on the selected file with easyocr")
parser.add_argument('-f', '--folder', type=str, help="Directory to walk through for OCR processing")
parser.add_argument('--recursive', action='store_true', help="Walk recursively through directories")

args = parser.parse_args()

TESSERACT_ENGINE = "tesseract_engine"
EASYOCR_ENGINE = "easyocr_engine"

results = []
failure_paths = []

# Instantiate the EasyOCR reader outside of the function, so it's reused
easyocr_reader = easyocr.Reader(['en'])

def log_scan_info(operation, engine, path, duration) :
    output = ""
    if(LOG_INDIVIDUAL_SCAN_TIMESTAMP):
        output += f"[{datetime.now().isoformat()}] "
    output += f"<{operation}> {engine} \"{path}\" {duration} seconds"
    if(LOG_RESULTS) :
        print(output)

#warning ai code
def segment_image_for_easyocr(image_path, segment_height=2560, overlap=200):

    img = Image.open(image_path)
    width, total_height = img.size
    segments = []
    
    y = 0
    while y < total_height:

        # Calculate crop coordinates with overlap
        top = max(0, y - overlap) if y > 0 else 0
        bot = min(y + segment_height, total_height)
        
        # Crop segment with overlap (except first segment)
        segment = img.crop((0, top, width, bot))
        segments.append(segment)
        
        # Move down the image
        y += segment_height - overlap
        
        # Break if we've reached the bottom
        if bot == total_height:
            break
            
    return segments

#warning ai code
def process_image_with_easyocr(image_path):
    global total_easyocr_scans_counter
    # Segment the image
    segment_height = 2560
    segment_overlap = 200
    segments = segment_image_for_easyocr(image_path, segment_height, segment_overlap)
    
    # Process each segment with EasyOCR
    full_text = [[] for it in segments]
    for i in range(len(segments)) :
        np_image = np.array(segments[i].convert('RGB'))
        results = easyocr_reader.readtext(np_image, min_size=10, text_threshold=0.5, low_text=0.25, link_threshold=0.2)
        total_easyocr_scans_counter += 1
        for (_, text, _) in results :
            full_text[i].append(text)

        
        # Convert image to OpenCV format for visualization
        img_vis = np_image.copy()
        img_vis_sorted = np_image.copy()
        if(VISUALIZE) :
            t_index = 0
            for (bbox, text, _) in results:
                cv2.destroyAllWindows()
                full_text[i].append(text)
                # Bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                # Draw bounding box around text
                color = random_color()
                cv2.rectangle(img_vis, top_left, bottom_right, color, 2)
                cv2.putText(img_vis, str(t_index), (top_left[0] - 20, top_left[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.675, color, 1, cv2.LINE_AA)
                t_index += 1
            # Show the segment with bounding boxes
            cv2.imshow(f"Segment {i}", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        #sort and show sorted
        def bbox_center_from_corners(bbox):
            top_left, top_right, bottom_right, bottom_left = bbox
            # Compute the center as the average of all four points
            center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
            center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4
            return (center_x, center_y)
        class tempObj:
            def __init__(self, box, text, used) :
                self.box = box
                self.text = text
                self.used = used
                self.center = bbox_center_from_corners(box)
                
        def calculate_bounding_box_from_corners(elements):
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            for element in elements:
                top_left, top_right, bottom_right, bottom_left = element.box
                min_x = min(min_x, top_left[0], top_right[0], bottom_right[0], bottom_left[0])
                min_y = min(min_y, top_left[1], top_right[1], bottom_right[1], bottom_left[1])
                max_x = max(max_x, top_left[0], top_right[0], bottom_right[0], bottom_left[0])
                max_y = max(max_y, top_left[1], top_right[1], bottom_right[1], bottom_left[1])
            return ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))
        
        result_copy = []
        lines = []
        # new data type
        for it in results:
            sortable = tempObj(it[0], it[1], False)
            result_copy.append(sortable)
        # concatenate into lines
        y_thresh = 20
        for j in range(len(result_copy)) :
            a = result_copy[j]
            if(a.used == False) :
                matches = []
                matches.append(a)
                result_copy[j].used = True
                for k in range(len(result_copy)) :
                    b = result_copy[k]
                    if(b.used == False):
                        if((a.center[1] - y_thresh) <= b.center[1] <= (a.center[1] + y_thresh)) :
                            result_copy[k].used = True
                            matches.append(b)
                            continue
                sorted_matches = sorted(matches, key=lambda el: el.center[0])
                newBox = calculate_bounding_box_from_corners(sorted_matches)
                newText = ""
                for it in sorted_matches :
                    newText += " " + it.text
                
                lines.append((newBox, newText))

        if(VISUALIZE) :
            t2_index = 0
            for (box2, text2) in lines:
                # Bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = box2
                top_left = (int(top_left[0]), int(top_left[1]))  # Explicitly make it a tuple
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))  # Explicitly make it a tuple
                # Draw bounding box around text
                color = random_color()
                cv2.rectangle(img_vis_sorted, top_left, bottom_right, color, 2)
                cv2.putText(img_vis_sorted, str(t2_index) + " : " + str(top_left[1]), (top_left[0] - 20, top_left[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.675, color, 1, cv2.LINE_AA)
                t2_index += 1
            # Show the segment with bounding boxes
            cv2.imshow(f"Segment sorted {i}", cv2.cvtColor(img_vis_sorted, cv2.COLOR_RGB2BGR))

            # Wait for user input before moving to the next segment
            print(f"Press any key to continue to the next segment ({i+1}/{len(segments)})...")
            cv2.waitKey(0)
            print("continuing...")
        newFullText = []
        for (box, text2) in lines :
            newFullText.append(text2)
        full_text[i] = newFullText
    

    # concatenate strings
    # NOTE : can use bouding rects for text to compare them positionally
    result_text = "" 
    for i in range(len(full_text[0])) :
        result_text += full_text[0][i]

    character_min_height = 10
    search_padding = 2
    max_string_overlap = int(segment_overlap / character_min_height) + search_padding
    for i in range(len(full_text) - 1) :
        first = full_text[i]
        second = full_text[i + 1]
        rangeFirstMax = max(0, len(first) - max_string_overlap)
        rangeSecondMax = min(len(second), 0 + max_string_overlap)
        rangeFirst =  reversed(range(rangeFirstMax, len(first)))
        rangeSecond = range(0, rangeSecondMax)
        
        region_found = False
        for j in rangeSecond :
            for k in rangeFirst :
                
                strA = second[j]
                strB = first[k] 
                
                score_levenshtein = _levenshtein_similarity(strA, strB)
                score_sequence = _sequence_matcher(strA, strB)

                levenshtein_threshold = 0.875
                sequence_threshold = 0.8
                score_thresh_scaler = 1.0
                if(score_levenshtein >= (score_thresh_scaler * levenshtein_threshold)) and (score_sequence >= (score_thresh_scaler * sequence_threshold)) :
                    region_found = True
                    if(j+1 < len(second)):
                        for h in range(j + 1, len(second)) :
                            result_text += " " + second[h]
                    break
            if(region_found):
                break
        if(region_found == False) :
            for h in range(len(second)) :
                result_text += " " + second[h]
        
    cv2.destroyAllWindows()
    return result_text

# Function to perform OCR on a file
def perform_ocr_on_file(file_path, engine_name):
    start_time = time.time()
    #print(f"<task started> [{engine_name} ocr : {file_path}]")
    global total_tesseract_scans_counter
    global total_tesseract_time
    global total_easyocr_time
    
    if engine_name == TESSERACT_ENGINE:
        img = Image.open(file_path)
        custom_config = r'--psm 6'
        try:
            tesseract_txt = pytesseract.image_to_string(img, config=custom_config)
            elapsed_time = time.time() - start_time
            time_string = str(elapsed_time)[0:6]
            log_scan_info("OCR", engine_name, file_path, time_string)
            total_tesseract_scans_counter += 1
            total_tesseract_time += elapsed_time
            return TextResult(text=tesseract_txt, engine_name=TESSERACT_ENGINE)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failure_paths.append(file_path)
            return  None

    elif engine_name == EASYOCR_ENGINE:
        try:
            easyocr_txt = process_image_with_easyocr(file_path)
            elapsed_time = time.time() - start_time
            time_string = str(elapsed_time)[0:6]
            log_scan_info("OCR", engine_name, file_path, time_string)
            total_easyocr_time += elapsed_time
            return TextResult(text=easyocr_txt, engine_name=EASYOCR_ENGINE)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failure_paths.append(file_path)
            return  None

    return None

# Function to process a directory (recursively or not)
def process_directory(directory, recursive=False):
    global total_processed_files_counter
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.jpg', '.png', '.jpeg')):  # You can extend this as needed
                if args.action_a:
                    # Perform OCR with both engines
                    results.append(perform_ocr_on_file(file_path, TESSERACT_ENGINE))
                    results.append(perform_ocr_on_file(file_path, EASYOCR_ENGINE))
                elif args.action_t:
                    # Perform OCR with Tesseract only
                    results.append(perform_ocr_on_file(file_path, TESSERACT_ENGINE))
                elif args.action_e:
                    # Perform OCR with EasyOCR only
                    results.append(perform_ocr_on_file(file_path, EASYOCR_ENGINE))
                total_processed_files_counter += 1
        if not recursive:
            break  # Stop recursion if not set to walk recursively

# Check if the input is a file or directory, and process accordingly
if args.action_a:
    if os.path.isfile(args.action_a):  # If it's a single file
        results.append(perform_ocr_on_file(args.action_a, TESSERACT_ENGINE))
        results.append(perform_ocr_on_file(args.action_a, EASYOCR_ENGINE))
    elif os.path.isdir(args.action_a):  # If it's a directory
        process_directory(args.action_a, recursive=args.recursive)
elif args.action_t:
    if os.path.isfile(args.action_t):
        results.append(perform_ocr_on_file(args.action_t, TESSERACT_ENGINE))
    elif os.path.isdir(args.action_t):
        process_directory(args.action_t, recursive=args.recursive)
elif args.action_e:
    if os.path.isfile(args.action_e):
        results.append(perform_ocr_on_file(args.action_e, EASYOCR_ENGINE))
    elif os.path.isdir(args.action_e):
        process_directory(args.action_e, recursive=args.recursive)
else:
    print("Invalid entry")


for it in results:
    if(it == None):
        continue
    else:
        json_string = it.serialize()
        print(json_string)

total_elapsed_time = time.time() - initialization_time
total_scans = total_easyocr_scans_counter + total_tesseract_scans_counter
non_scan_time = total_elapsed_time - (total_easyocr_time + total_tesseract_time)
current_iso_time = datetime.now().isoformat()

# Human-readable console output
print(f"[{current_iso_time}] SCAN SUMMARY")
print(f"[{current_iso_time}] Processed {total_processed_files_counter} files in {total_elapsed_time:.3f}s")
print(f"[{current_iso_time}] Tesseract: {total_tesseract_scans_counter} scans ({total_tesseract_time:.3f}s)")
print(f"[{current_iso_time}] EasyOCR: {total_easyocr_scans_counter} scans ({total_easyocr_time:.3f}s)")
print(f"[{current_iso_time}] Non-scan time: {total_elapsed_time - (total_easyocr_time + total_tesseract_time):.3f}s")
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
        "tesseract": {
            "scans": total_tesseract_scans_counter,
            "time_sec": round(total_tesseract_time, 3)
        },
        "easyocr": {
            "scans": total_easyocr_scans_counter,
            "time_sec": round(total_easyocr_time, 3)
        },
        "non_scan_time_sec": round(total_elapsed_time - (total_easyocr_time + total_tesseract_time), 3)
    },
    "failures": failure_paths
}

# Save JSON to file (appends to log file)
with open("scan_logs.json", "a") as log_file:
    json.dump(log_entry, log_file)
    log_file.write("\n")
