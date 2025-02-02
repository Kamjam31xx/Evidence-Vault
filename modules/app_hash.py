import hashlib
import os
import struct
from modules.app_types import FileDataHash

# uses odd number to avoid patterns in binary numbers
def hash_file_data_sparse_truncated(file_path, length=16, chunk_size=4096, step=127):

    hasher = hashlib.blake2b(digest_size=length)

    with open(file_path, 'rb') as f:
        file_size = os.path.getsize(file_path)
        num_chunks = file_size // chunk_size

        for i in range(0, num_chunks, step):  # Read every `step`th chunk
            f.seek(i * chunk_size)  # Move to chunk position
            chunk = f.read(chunk_size)  # Read chunk
            if not chunk:
                break
            hasher.update(chunk)  # Hash chunk

    return hasher.hexdigest()

def hash_file_metadata(file_path, length=16):

    hasher = hashlib.blake2b(digest_size=length)

    try:
        stats = os.stat(file_path)
        file_name = os.path.basename(file_path).encode("utf-8")  # Convert name to bytes
        file_extension = os.path.splitext(file_path)[1].encode("utf-8")  # Extract extension

        # Pack metadata into bytes (little-endian format for consistency)
        meta_data = struct.pack(
            "<QQQQQQQ",  # Format for size, mtime, ctime, mode, inode, device, and UID
            stats.st_size,         # File size in bytes
            int(stats.st_mtime),   # Last modified timestamp
            int(stats.st_ctime),   # Creation timestamp
            stats.st_mode,         # File mode (permissions)
            stats.st_ino,          # Inode number (UNIX systems)
            stats.st_dev,          # Device ID
            stats.st_uid           # User ID of owner (UNIX systems)
        )

        # Combine metadata, filename, and extension
        hasher.update(meta_data + file_name + file_extension)

    except Exception as e:
        print(f"Error hashing metadata: {e}")

    return hasher.hexdigest()

def get_hashes_for_file(file_path) :
    return FileDataHash(hash_file_metadata(file_path), hash_file_data_sparse_truncated(file_path))