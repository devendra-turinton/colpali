#!/usr/bin/env python3
"""
Script to download models separately with progress tracking and resume capability
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directory
CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

def download_colpali_model():
    """Download ColPali model with resume capability"""
    model_id = "vidore/colpali"
    
    logger.info(f"Downloading ColPali model to {CACHE_DIR}")
    logger.info("This is a one-time download of ~5.8GB")
    
    try:
        # Download with resume capability
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=CACHE_DIR,
            resume_download=True,
            local_dir_use_symlinks=False,
            max_workers=4  # Parallel downloads
        )
        logger.info(f"✓ ColPali model downloaded to: {local_dir}")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading ColPali: {e}")
        return False

def download_vlm_model():
    """Download Qwen2-VL model (optional, large)"""
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    logger.info(f"Downloading Qwen2-VL model to {CACHE_DIR}")
    logger.info("This is a LARGE download of ~15GB")
    
    response = input("Download Qwen2-VL? (y/n): ")
    if response.lower() != 'y':
        logger.info("Skipping VLM download")
        return False
    
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=CACHE_DIR,
            resume_download=True,
            local_dir_use_symlinks=False,
            max_workers=4
        )
        logger.info(f"✓ Qwen2-VL model downloaded to: {local_dir}")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading Qwen2-VL: {e}")
        return False

def check_existing_models():
    """Check what models are already downloaded"""
    logger.info("Checking for existing models...")
    
    models_to_check = [
        ("vidore/colpali", "ColPali (Required)"),
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL (Optional)")
    ]
    
    found_models = []
    for model_id, name in models_to_check:
        model_path = Path(CACHE_DIR) / "hub" / f"models--{model_id.replace('/', '--')}"
        if model_path.exists() and any(model_path.iterdir()):
            try:
                # Calculate size more carefully
                total_size = 0
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        file_path = Path(root) / f
                        if file_path.is_file() and not file_path.is_symlink():
                            total_size += file_path.stat().st_size
                
                size_gb = total_size / (1024**3)
                logger.info(f"✓ {name}: Found ({size_gb:.2f} GB)")
                found_models.append(model_id)
            except Exception as e:
                logger.warning(f"Error calculating size for {name}: {e}")
                logger.info(f"✓ {name}: Found")
                found_models.append(model_id)
        else:
            logger.info(f"✗ {name}: Not found")
    
    return found_models

def main():
    """Main download function"""
    logger.info("Model Download Manager for Multimodal RAG")
    logger.info("="*50)
    
    # Create cache directory
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check existing models
    found_models = check_existing_models()
    
    # Download ColPali if not present
    if "vidore/colpali" not in found_models:
        logger.info("\nColPali model is required for the system to work.")
        if not download_colpali_model():
            logger.error("Failed to download required model. Exiting.")
            sys.exit(1)
    else:
        logger.info("\n✓ ColPali model already downloaded")
    
    # Ask about VLM
    if "Qwen/Qwen2-VL-7B-Instruct" not in found_models:
        logger.info("\nQwen2-VL is optional. Without it, you'll use Groq API for responses.")
        download_vlm_model()
    else:
        logger.info("\n✓ Qwen2-VL model already downloaded")
    
    logger.info("\n" + "="*50)
    logger.info("Setup complete! You can now run multimodal_rag_implementations.py")
    logger.info("Models are cached and won't be downloaded again.")

if __name__ == "__main__":
    main()