# Evidence-Vault
A tool for automatically scanning evidence to extract useful information with OCR, STT, and other methods for populating a database to query &amp; visualize it.



## OCR Tool
Using a terminal we can give the OCR tool a path for a file, or a directory. Giving a directory scans all files in the directory. Giving a directory with the ` --recursive flag ` recursively walks the sub-tree in that directory, scanning all files.



### OCR Commands Example

- **All OCR engines**:
    ```bash
    py ocr.py -a "test_images"  # -a for all OCR engines
    ```

- **Tesseract OCR engine**:
    ```bash
    py ocr.py -t "test_images"  # -t for Tesseract OCR engine
    ```

- **EasyOCR engine**:
    ```bash
    py ocr.py -e "test_images"  # -e for EasyOCR engine
    ```

- **All engines with recursive directory processing**:
    ```bash
    py ocr.py -a "test_images" --recursive  # --recursive to walk through directories
    ```

#### Flag Description:
- ` -a `: Perform OCR with all available engines
- ` -t `: Use Tesseract OCR engine
- ` -e `: Use EasyOCR engine
- ` --recursive `: Walk through directories recursively



# Terminal example
![image](https://github.com/user-attachments/assets/e25e228e-4e5c-45dd-b841-b21d96f0bd22)
![image](https://github.com/user-attachments/assets/1d2d8615-c69a-4919-9b41-d93e1f735605)



# CUDA support
If you get the following message `Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.`, you can try running `nvidia-smi` to check your CUDA version. 
```bash
nvidia-smi
```

![image](https://github.com/user-attachments/assets/331d06ef-17fb-41a3-97fa-6c808ca17119)

Once you have the correct version to support(12.1 in the example), you can download the appropriate support like so `py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. The end of the url "cu121" corresponds to CUDA 12.1.

![image](https://github.com/user-attachments/assets/d0dc3add-b970-4b42-92d9-8564b505593d)
