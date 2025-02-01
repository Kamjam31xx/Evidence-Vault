import argparse
import os
import sqlite3
from PIL import Image
import pytesseract
import easyocr
from app_types import TextResult
import time
import numpy as np
from string_comparison import _levenshtein_similarity, _sequence_matcher
import cv2
from app_colors import *

initialization_time = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
VISUALIZE = False
DIAGNOSTIC_LOGGING = False


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

# Instantiate the EasyOCR reader outside of the function, so it's reused
easyocr_reader = easyocr.Reader(['en'])

#warning ai code
def segment_image_for_easyocr(image_path, segment_height=2560, overlap=200):

    img = Image.open(image_path)
    width, total_height = img.size
    DIAGNOSTIC_LOGGING and print("image size : ", width, " x ", total_height)
    segments = []
    
    y = 0
    while y < total_height:

        # Calculate crop coordinates with overlap
        top = max(0, y - overlap) if y > 0 else 0
        bot = min(y + segment_height, total_height)
        DIAGNOSTIC_LOGGING and print("top: ", top,"    bot: ", bot)
        
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
    # Segment the image
    segment_height = 2560
    segment_overlap = 200
    segments = segment_image_for_easyocr(image_path, segment_height, segment_overlap)
    
    # Process each segment with EasyOCR
    full_text = [[] for it in segments]
    for i in range(len(segments)) :
        np_image = np.array(segments[i].convert('RGB'))
        results = easyocr_reader.readtext(np_image, min_size=10, text_threshold=0.5, low_text=0.25, link_threshold=0.2)
        for (_, text, _) in results :
            full_text[i].append(text)
            DIAGNOSTIC_LOGGING and print("log[",i,"] : ",text)

        
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
                        DIAGNOSTIC_LOGGING and print(f"{(a.center[1] - y_thresh)} <= {b.center[1]} <= {(a.center[1] + y_thresh)} == {(a.center[1] - y_thresh) <= b.center[1] <= (a.center[1] + y_thresh)}")
                sorted_matches = sorted(matches, key=lambda el: el.center[0])
                newBox = calculate_bounding_box_from_corners(sorted_matches)
                newText = ""
                for it in sorted_matches :
                    DIAGNOSTIC_LOGGING and print(it.text)
                    newText += " " + it.text
                
                lines.append((newBox, newText))
        for it in lines:
            DIAGNOSTIC_LOGGING and print(it)
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
        DIAGNOSTIC_LOGGING and print(f">>> comparisons {i}")
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
                
                DIAGNOSTIC_LOGGING and print(f"comparing first:[{i}][{k}] and second[{i + 1}][{j}] <score: {score_levenshtein}, {score_sequence}>")
                DIAGNOSTIC_LOGGING and print(f"second[{j}]  {strA}    :    first[{k}]  {strB}")

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
        DIAGNOSTIC_LOGGING and print("")
        
    cv2.destroyAllWindows()
    return result_text

# Function to perform OCR on a file
def perform_ocr_on_file(file_path, engine_name):
    start_time = time.time()
    #print(f"<task started> [{engine_name} ocr : {file_path}]")
    if engine_name == TESSERACT_ENGINE:
        img = Image.open(file_path)
        custom_config = r'--psm 6'
        tesseract_txt = pytesseract.image_to_string(img, config=custom_config)
        elapsed_time = time.time() - start_time
        print(f"<task completed> [{engine_name} ocr : {file_path}]", elapsed_time, " seconds")
        return TextResult(text=tesseract_txt, engine_name=TESSERACT_ENGINE, file_path=file_path)
    elif engine_name == EASYOCR_ENGINE:
        #easyocr_output = easyocr_reader.readtext(file_path)
        #easyocr_txt = ' '.join([text for (_, text, _) in easyocr_output])
        easyocr_txt = process_image_with_easyocr(file_path)
        elapsed_time = time.time() - start_time
        print(f"<task completed> [{engine_name} ocr : {file_path}]", elapsed_time, " seconds")
        return TextResult(text=easyocr_txt, engine_name=EASYOCR_ENGINE, file_path=file_path)
    #elapsed_time = time.time() - start_time
    #print(f"<task completed> [{engine_name} ocr : {file_path}]", elapsed_time, " seconds")
    return None

# Function to process a directory (recursively or not)
def process_directory(directory, recursive=False):
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
elif args.folder:
    if os.path.isdir(args.folder):  # If folder is provided, process the directory
        process_directory(args.folder, recursive=args.recursive)
else:
    print("No valid file or directory specified.")

# Print results
for it in results:
    json_string = it.serialize()
    print(json_string)
    #for key,value in dictionary :
    #    print(f"{key}: {value}")
print(f"<tasks complete> {time.time() - initialization_time} seconds")
