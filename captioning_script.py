#!/usr/bin/env python3
"""
Universal Vision Captioning Script
Supports multiple vision models including Gemini 2.5 Pro and Molmo
Handles both individual images/frames and video processing
"""

import os
import cv2
import base64
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod

# Import libraries based on available models
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google GenerativeAI not available. Install with: pip install google-generativeai")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")


class VisionModel(ABC):
    """Abstract base class for vision models"""
    
    @abstractmethod
    def caption_image(self, image_path: str, prompt: str = None) -> str:
        """Caption a single image"""
        pass
    
    @abstractmethod
    def caption_frame(self, frame_array, prompt: str = None) -> str:
        """Caption a frame from video (numpy array)"""
        pass


class GeminiModel(VisionModel):
    """Gemini vision model"""
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-pro-vision"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI library not available")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
    
    def caption_image(self, image_path: str, prompt: str = None) -> str:
        """Caption an image file"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL library required for Gemini image processing")
        
        image = Image.open(image_path)
        
        if prompt is None:
            prompt = "Describe this image in detail, focusing on the main subjects, actions, and setting."
        
        try:
            response = genai.generate_text(
                model=self.model_name,
                prompt=prompt,
                temperature=self.generation_config["temperature"]
            )
            return response.result if response.result else "No caption generated"
        except Exception as e:
            # Try alternative method for vision models
            try:
                response = genai.generate_text(
                    model="models/gemini-pro-vision",
                    prompt={"text": prompt, "image": image},
                    temperature=self.generation_config["temperature"]
                )
                return response.result if response.result else "No caption generated"
            except Exception as e2:
                return f"Error captioning image: {str(e)} / {str(e2)}"
    
    def caption_frame(self, frame_array, prompt: str = None) -> str:
        """Caption a video frame (numpy array)"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL library required for Gemini frame processing")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        if prompt is None:
            prompt = "Describe this video frame in detail, focusing on the main subjects, actions, and setting."
        
        try:
            response = genai.generate_text(
                model="models/gemini-pro-vision",
                prompt={"text": prompt, "image": image},
                temperature=self.generation_config["temperature"]
            )
            return response.result if response.result else "No caption generated"
        except Exception as e:
            return f"Error captioning frame: {str(e)}"


class MolmoModel(VisionModel):
    """Molmo vision model (via API endpoint)"""
    
    def __init__(self, api_endpoint: str, api_key: str = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not available")
        
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _encode_frame_base64(self, frame_array) -> str:
        """Encode frame to base64"""
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame_array)
        return base64.b64encode(buffer).decode('utf-8')
    
    def caption_image(self, image_path: str, prompt: str = None) -> str:
        """Caption an image file"""
        if prompt is None:
            prompt = "Describe this image in detail, focusing on the main subjects, actions, and setting."
        
        image_b64 = self._encode_image_base64(image_path)
        
        payload = {
            "model": "molmo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No caption generated")
        except Exception as e:
            return f"Error captioning image: {str(e)}"
    
    def caption_frame(self, frame_array, prompt: str = None) -> str:
        """Caption a video frame"""
        if prompt is None:
            prompt = "Describe this video frame in detail, focusing on the main subjects, actions, and setting."
        
        frame_b64 = self._encode_frame_base64(frame_array)
        
        payload = {
            "model": "molmo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No caption generated")
        except Exception as e:
            return f"Error captioning frame: {str(e)}"


class VisionCaptioner:
    """Main captioning class that handles different models and input types"""
    
    def __init__(self, model: VisionModel):
        self.model = model
    
    def caption_images(self, image_paths: List[str], prompt: str = None, output_file: str = None) -> List[Dict[str, Any]]:
        """Caption multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            caption = self.model.caption_image(image_path, prompt)
            
            result = {
                "image_path": image_path,
                "caption": caption,
                "timestamp": time.time()
            }
            results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def caption_video(self, video_path: str, frame_interval: int = 30, prompt: str = None, output_file: str = None) -> List[Dict[str, Any]]:
        """Caption frames from a video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        print(f"Sampling every {frame_interval} frames")
        
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                print(f"Processing frame {frame_count} at {timestamp:.2f}s")
                
                caption = self.model.caption_frame(frame, prompt)
                
                result = {
                    "video_path": video_path,
                    "frame_number": frame_count,
                    "timestamp_seconds": timestamp,
                    "caption": caption
                }
                results.append(result)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
            
            frame_count += 1
        
        cap.release()
        
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")


def create_model(model_type: str, **kwargs) -> VisionModel:
    """Factory function to create vision models"""
    if model_type.lower() == "gemini":
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
        model_name = kwargs.get("model_name", "models/gemini-pro-vision")
        return GeminiModel(api_key, model_name)
    
    elif model_type.lower() == "molmo":
        api_endpoint = kwargs.get("api_endpoint") or os.getenv("MOLMO_API_ENDPOINT")
        if not api_endpoint:
            raise ValueError("Molmo API endpoint required. Set MOLMO_API_ENDPOINT environment variable or pass api_endpoint parameter")
        api_key = kwargs.get("api_key") or os.getenv("MOLMO_API_KEY")
        return MolmoModel(api_endpoint, api_key)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: gemini, molmo")


def main():
    parser = argparse.ArgumentParser(description="Universal Vision Captioning Script")
    parser.add_argument("--model", required=True, choices=["gemini", "molmo"], 
                       help="Vision model to use")
    parser.add_argument("--input", required=True, 
                       help="Input file or directory (images/video)")
    parser.add_argument("--output", 
                       help="Output JSON file for results")
    parser.add_argument("--prompt", 
                       help="Custom prompt for captioning")
    parser.add_argument("--frame-interval", type=int, default=30,
                       help="Frame interval for video processing (default: 30)")
    
    # Model-specific arguments
    parser.add_argument("--api-key", 
                       help="API key (or set environment variable)")
    parser.add_argument("--api-endpoint", 
                       help="API endpoint for Molmo (or set MOLMO_API_ENDPOINT)")
    parser.add_argument("--model-name", default="models/gemini-pro-vision",
                       help="Gemini model name (default: models/gemini-pro-vision)")
    
    args = parser.parse_args()
    
    # Create model
    model_kwargs = {
        "api_key": args.api_key,
        "api_endpoint": args.api_endpoint,
        "model_name": args.model_name
    }
    model = create_model(args.model, **model_kwargs)
    
    # Create captioner
    captioner = VisionCaptioner(model)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        if input_path.suffix.lower() in video_extensions:
            print(f"Processing video: {input_path}")
            results = captioner.caption_video(str(input_path), args.frame_interval, args.prompt, args.output)
        elif input_path.suffix.lower() in image_extensions:
            print(f"Processing image: {input_path}")
            results = captioner.caption_images([str(input_path)], args.prompt, args.output)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
    
    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [str(f) for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {input_path}")
        
        print(f"Processing {len(image_files)} images from directory: {input_path}")
        results = captioner.caption_images(image_files, args.prompt, args.output)
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # Print results summary
    print(f"\nProcessed {len(results)} items")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"\nResult {i+1}:")
        if 'image_path' in result:
            print(f"Image: {result['image_path']}")
        else:
            print(f"Video: {result['video_path']}, Frame: {result['frame_number']}, Time: {result['timestamp_seconds']:.2f}s")
        print(f"Caption: {result['caption'][:100]}...")


def quick_caption_taskimages(api_key: str = None):
    """Quick function to caption all images in TaskImages folder with Gemini 2.5 Flash"""
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Please provide API key or set GOOGLE_API_KEY environment variable")
            return
    
    # Create Gemini model
    model = GeminiModel(api_key, "models/gemini-pro-vision")
    captioner = VisionCaptioner(model)
    
    # Process TaskImages folder
    taskimages_path = Path("TaskImages")
    if not taskimages_path.exists():
        print("TaskImages folder not found in current directory")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [str(f) for f in taskimages_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No image files found in TaskImages folder")
        return
    
    print(f"Found {len(image_files)} images in TaskImages folder")
    
    # Caption all images
    results = captioner.caption_images(
        image_files, 
        prompt="Describe this image in detail, focusing on what you see in this screenshot.",
        output_file="taskimages_captions.json"
    )
    
    # Print results
    print(f"\nCaptioned {len(results)} images. Results saved to taskimages_captions.json")
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1}: {Path(result['image_path']).name} ---")
        print(result['caption'])


if __name__ == "__main__":
    main()