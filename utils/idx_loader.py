import numpy as np
import struct

def load_idx_images(file_path):
    """loads idx image files"""
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_idx_labels(file_path):
    """loads label files"""
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data