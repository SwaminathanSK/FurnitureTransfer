#!/usr/bin/env python3
"""
Gemini vision captioning using REST API
Works with any Python version and directly calls Google's REST API
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


def caption_image_with_gemini_rest(api_key: str, image_path: str, prompt: str = None) -> str:
    """Caption an image using Gemini Pro Vision via REST API"""
    
    if prompt is None:
        prompt = "Describe this image in detail. What do you see? What appears to be happening in this screenshot?"
    
    # Encode image
    image_b64 = encode_image_base64(image_path)
    
    # Prepare the request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.8,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]
        
        return f"No caption generated for {image_path}"
        
    except requests.exceptions.RequestException as e:
        return f"Request error for {image_path}: {str(e)}"
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"



def caption_images_in_folder(api_key: str, folder_path: str = "TaskImages", output_file: str = "gemini_captions.json"):
    """Caption all images in a folder"""
    
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
    
    print(f"Found {len(image_files)} images to caption with Gemini 1.5 Flash")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        caption = caption_image_with_gemini_rest(
            api_key, 
            str(image_path)
        )
        
        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "caption": caption,
            "timestamp": time.time()
        }
        results.append(result)
        
        print(f"Caption: {caption[:100]}...")
        
        # Rate limiting - Gemini has generous limits but let's be safe
        time.sleep(1)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Processed {len(results)} images")
    print(f"Results saved to {output_file}")
    
    # Print all captions
    print("\n" + "="*80)
    print("ALL CAPTIONS:")
    print("="*80)
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1}: {result['image_name']} ---")
        print(result['caption'])
    
    return results


def main():
    print("Gemini Vision Captioning Tool")
    print("="*40)
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
        if not api_key:
            print("API key required. Set GOOGLE_API_KEY environment variable or enter when prompted.")
            return
    
    # Run captioning
    caption_images_in_folder(api_key)


if __name__ == "__main__":
    main()