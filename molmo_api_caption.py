#!/usr/bin/env python3
"""
Molmo vision captioning using Hugging Face Inference API
Lighter weight alternative that doesn't require downloading the model
"""

import os
import json
import time
import base64
from pathlib import Path
from typing import List, Dict, Any
import requests


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def caption_image_with_molmo_api(hf_token: str, image_path: str, prompt: str = None) -> str:
    """Caption an image using Molmo via Hugging Face Inference API"""
    
    if prompt is None:
        prompt = "Describe this image in detail. What do you see? What appears to be happening in this screenshot?"
    
    # Encode image
    image_b64 = encode_image_base64(image_path)
    
    # Hugging Face Inference API endpoint for Molmo
    url = "https://api-inference.huggingface.co/models/allenai/Molmo-7B-D-0924"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": {
            "text": prompt,
            "image": image_b64
        },
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            print(f"Model loading, waiting 20 seconds...")
            time.sleep(20)
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            elif "text" in result[0]:
                return result[0]["text"].strip()
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"].strip()
            elif "text" in result:
                return result["text"].strip()
        
        return f"No caption generated for {image_path}"
        
    except requests.exceptions.RequestException as e:
        return f"Request error for {image_path}: {str(e)}"
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"


def caption_images_with_molmo_api(hf_token: str, folder_path: str = "TaskImages", output_file: str = "molmo_api_captions.json"):
    """Caption all images in a folder using Molmo API"""
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist")
        return
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in folder.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to caption with Molmo API")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        caption = caption_image_with_molmo_api(hf_token, str(image_path))
        
        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "caption": caption,
            "timestamp": time.time(),
            "model": "allenai/Molmo-7B-D-0924"
        }
        results.append(result)
        
        print(f"Caption: {caption[:100]}...")
        
        # Rate limiting for API
        time.sleep(2)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Processed {len(results)} images")
    print(f"Results saved to {output_file}")
    
    # Print all captions
    print("\n" + "="*80)
    print("ALL MOLMO API CAPTIONS:")
    print("="*80)
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1}: {result['image_name']} ---")
        print(result['caption'])
    
    return results


def main():
    print("Molmo Vision Captioning Tool (Hugging Face API)")
    print("="*50)
    print("This script uses Hugging Face Inference API to run Molmo")
    print("No model download required, but needs HF token")
    print()
    
    # Get HF token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ").strip()
        if not hf_token:
            print("Hugging Face token required.")
            print("Get one at: https://huggingface.co/settings/tokens")
            return
    
    print("Using Hugging Face Inference API...")
    
    # Run captioning
    caption_images_with_molmo_api(hf_token)


if __name__ == "__main__":
    main()