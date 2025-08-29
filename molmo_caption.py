#!/usr/bin/env python3
"""
Molmo vision captioning script for TaskImages folder
Uses Hugging Face Transformers to run Molmo locally
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import torch
from PIL import Image

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: transformers not available. Install with: pip install transformers")
    exit(1)


class MolmoLocalCaptioner:
    """Local Molmo model for image captioning"""
    
    def __init__(self, model_name: str = "allenai/Molmo-7B-D-0924"):
        """Initialize Molmo model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")
        
        print(f"Loading Molmo model: {model_name}")
        print("This may take a few minutes on first run...")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto" if self.device == "cuda" else torch.float32
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto" if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("Molmo model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Molmo model: {e}")
            print("Trying with smaller model...")
            try:
                # Fallback to smaller model
                model_name = "allenai/Molmo-7B-O-0924"
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model = self.model.to(self.device)
                print(f"Loaded fallback model: {model_name}")
            except Exception as e2:
                raise Exception(f"Failed to load any Molmo model: {e2}")
    
    def caption_image(self, image_path: str, prompt: str = None) -> str:
        """Caption a single image"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            if prompt is None:
                prompt = "Describe this image in detail. What do you see? What appears to be happening in this screenshot?"
            
            # Process inputs
            inputs = self.processor.process(
                images=[image],
                text=prompt
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(
                        max_new_tokens=200,
                        stop_strings=["<|endoftext|>"],
                        temperature=0.7,
                        do_sample=True
                    ),
                    tokenizer=self.processor.tokenizer
                )
            
            # Decode the response
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            return f"Error captioning {image_path}: {str(e)}"


def caption_images_with_molmo(model_name: str = "allenai/Molmo-7B-D-0924", 
                             folder_path: str = "TaskImages", 
                             output_file: str = "molmo_captions.json"):
    """Caption all images in a folder using local Molmo model"""
    
    # Initialize Molmo
    try:
        captioner = MolmoLocalCaptioner(model_name)
    except Exception as e:
        print(f"Failed to initialize Molmo: {e}")
        return
    
    # Find image files
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in folder.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to caption with Molmo")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        caption = captioner.caption_image(str(image_path))
        
        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "caption": caption,
            "timestamp": time.time(),
            "model": model_name
        }
        results.append(result)
        
        print(f"Caption: {caption[:100]}...")
        
        # Small delay for memory management
        time.sleep(0.5)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Processed {len(results)} images")
    print(f"Results saved to {output_file}")
    
    # Print all captions
    print("\n" + "="*80)
    print("ALL MOLMO CAPTIONS:")
    print("="*80)
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1}: {result['image_name']} ---")
        print(result['caption'])
    
    return results


def main():
    print("Molmo Vision Captioning Tool")
    print("="*40)
    print("This script runs Molmo locally using Hugging Face Transformers")
    print("Note: First run will download the model (~14GB)")
    print()
    
    # Check available models
    available_models = [
        "allenai/Molmo-7B-D-0924",  # Default model
        "allenai/Molmo-7B-O-0924",  # Alternative
        "allenai/Molmo-72B-0924"    # Larger model (requires more VRAM)
    ]
    
    print("Available Molmo models:")
    for i, model in enumerate(available_models):
        print(f"  {i+1}. {model}")
    
    # Let user choose model or use default
    choice = input("\nEnter model number (1-3) or press Enter for default: ").strip()
    
    if choice in ['1', '2', '3']:
        model_name = available_models[int(choice) - 1]
    else:
        model_name = available_models[0]  # Default
    
    print(f"Using model: {model_name}")
    
    # Run captioning
    caption_images_with_molmo(model_name)


if __name__ == "__main__":
    main()