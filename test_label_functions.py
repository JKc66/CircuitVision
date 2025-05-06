#!/usr/bin/env python3
import os
import sys
import cv2
from src.utils import gemini_labels_openrouter, summarize_components

def test_openrouter_labeling():
    """
    Test the gemini_labels_openrouter function with a sample image.
    """
    print("Testing gemini_labels_openrouter function...")
    
    # Load the test image
    image_path = r"D:\SDP_demo\static\uploads\1.png"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Read the image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}")
        return
    
    print(f"Successfully loaded image from {image_path}")
    print(f"Image shape: {image.shape}")
    
    try:
        # Call the OpenRouter labeling function
        print("Calling gemini_labels_openrouter...")
        results = gemini_labels_openrouter(image)
        
        # Display the results
        print("\nResults:")
        print(f"Identified {len(results)} components:")
        
        for idx, component in enumerate(results):
            print(f"\nComponent {idx+1}:")
            print(f"  Class: {component.get('class', 'unknown')}")
            print(f"  Value: {component.get('value', 'None')}")
            print(f"  Direction: {component.get('direction', 'None')}")
            print(f"  ID: {component.get('id', 'None')}")
        
        # Generate a summary
        summary = summarize_components(results)
        print(f"\nSummary: {summary}")
        
        return results
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting test...")
    results = test_openrouter_labeling()
    print("Test completed.")
