#!/usr/bin/env python3
"""Test script to understand FinBERT output format"""

from transformers import pipeline

# Test FinBERT pipeline
try:
    print("Loading FinBERT model...")
    finbert_pipeline = pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        return_all_scores=True
    )
    
    # Test with a simple headline
    test_text = "EUR/USD surges higher"
    print(f"\nTesting with: '{test_text}'")
    
    result = finbert_pipeline(test_text)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    if isinstance(result, list) and len(result) > 0:
        for i, item in enumerate(result):
            print(f"Item {i}: {item}")
            print(f"Item type: {type(item)}")
            if isinstance(item, dict):
                for key, value in item.items():
                    print(f"  {key}: {value} (type: {type(value)})")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()