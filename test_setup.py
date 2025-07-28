#!/usr/bin/env python3
import sys

def test_minimal_setup():
    """Test only essential components"""
    print("Testing minimal setup...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except:
        print("✗ PyTorch not available")
        return False
        
    try:
        from byaldi import RAGMultiModalModel
        print("✓ Byaldi (ColPali) available")
    except:
        print("✗ Byaldi not available")
        return False
        
    try:
        import groq
        print("✓ Groq API client available")
    except:
        print("✗ Groq not available")
        return False
        
    try:
        from pdf2image import convert_from_path
        print("✓ PDF2Image available")
    except:
        print("✗ PDF2Image not available")
        return False
        
    return True

if __name__ == "__main__":
    if test_minimal_setup():
        print("\n✓ Minimal setup OK - You can use the Groq-based implementation")
    else:
        print("\n✗ Setup incomplete")
        sys.exit(1)