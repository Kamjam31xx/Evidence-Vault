# Evidence-Vault
A tool for automatically scanning evidence to extract useful information with OCR, STT, and other methods for populating a database to query &amp; visualize it.

## OCR Tool
Using a terminal we can give the OCR tool a path for a file, or a directory. Giving a directory scans all files in the directory. Giving a directory with the --recursive flag recursively walks the sub-tree in that directory, scanning all files.

*Using command line to call the OCR tool*
<span style="color: blue;">py ocr.py -a "test_images"</span> *-a all OCR engines*
<span style="color: blue;">py ocr.py -t "test_images"</span> *-t tesseract OCR engine*
<span style="color: blue;">py ocr.py -e "test_images"</span> *-t easyocr OCR engine*
<span style="color: blue;">py ocr.py -a "test_images --recursive"</span> *--resursive walk tree*
![image](https://github.com/user-attachments/assets/e230c92f-4890-429e-9132-adaa41a214ba)
