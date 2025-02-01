# Evidence-Vault
A tool for automatically scanning evidence to extract useful information with OCR, STT, and other methods for populating a database to query &amp; visualize it.

## OCR Tool
Using a terminal we can give the OCR tool a path for a file, or a directory. Giving a directory scans all files in the directory. Giving a directory with the --recursive flag recursively walks the sub-tree in that directory, scanning all files.

### OCR Commands Example

- **All OCR engines**:
    ```bash
    python ocr.py -a "test_images"  # -a for all OCR engines
    ```

- **Tesseract OCR engine**:
    ```bash
    python ocr.py -t "test_images"  # -t for Tesseract OCR engine
    ```

- **EasyOCR engine**:
    ```bash
    python ocr.py -e "test_images"  # -e for EasyOCR engine
    ```

- **All engines with recursive directory processing**:
    ```bash
    python ocr.py -a "test_images" --recursive  # --recursive to walk through directories
    ```

#### Flag Description:
- `-a`: Perform OCR with all available engines
- `-t`: Use Tesseract OCR engine
- `-e`: Use EasyOCR engine
- `--recursive`: Walk through directories recursively

![image](https://github.com/user-attachments/assets/e230c92f-4890-429e-9132-adaa41a214ba)
