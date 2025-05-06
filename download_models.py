import os
import requests
import subprocess
import sys

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("models/SAM2", exist_ok=True)
    os.makedirs("models/YOLO", exist_ok=True)
    os.makedirs("models/configs", exist_ok=True)

def download_file(url, filename):
    """Download a file from a URL."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error if download fails

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename} successfully.")

def download_gdown(file_id, output_path):
    """Download a file from Google Drive."""
    try:
        print(f"Downloading to {output_path}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "gdown", "--upgrade", "--quiet"
        ])
        subprocess.check_call([
            sys.executable, "-m", "gdown", file_id, "-O", output_path
        ])
        print(f"Downloaded {output_path} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {output_path}: {e}")

def download_sam_finetuned():
    """Download latest finetuned SAM model."""
    download_gdown(
        "1AI91ikS-wlu7Pl-FwK7lNHyBqczKiBLr", 
        "models/SAM2/best_miou_model_SAM_latest.pth"
    )

def download_sam_base():
    """Download base SAM2.1 model."""
    download_file(
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "models/SAM2/sam2.1_hiera_large.pt"
    )
    
    # Also download the configuration YAML
    download_file(
        "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "models/configs/sam2.1_hiera_l.yaml"
    )

def download_yolo():
    """Download YOLO model."""
    download_gdown(
        "1AoGPtKyW5SW5olxGLZOI-4r4QCTQFoH9", 
        "models/YOLO/best_large_model_yolo.pt"
    )

def main():
    """Main function to download all models."""
    create_directories()
    
    print("Starting model downloads...")
    
    # Download all models
    download_sam_finetuned()
    download_sam_base()
    download_yolo()
    
    print("All downloads complete.")

if __name__ == "__main__":
    main()
