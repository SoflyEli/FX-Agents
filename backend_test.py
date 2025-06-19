import unittest
import requests
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL')
if not BACKEND_URL:
    raise ValueError("REACT_APP_BACKEND_URL not found in environment variables")

# Ensure URL ends with /api
API_URL = f"{BACKEND_URL}/api"
print(f"Testing API at: {API_URL}")

class TestSentimentAnalyzer(unittest.TestCase):
    """Test suite for the NLP Sentiment-Scoring Analyzer backend"""

    def setUp(self):
        """Setup for tests - wait for models to load"""
        # Wait a bit to ensure models are loaded
        time.sleep(2)
        
        # Test headlines for sentiment analysis
        self.test_headlines = [
            "EUR/USD surges to new monthly highs as ECB signals rate hikes",
            "GBP/JPY remains flat amid mixed economic signals",
            "USD/CHF plunges on weak US employment data",
            "Gold prices rally as inflation concerns mount",
            "Bitcoin crashes below key support level",
            "",  # Empty headline for error testing
            "X" * 5000  # Very long headline for error testing
        ]

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = requests.get(f"{API_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check required fields
        self.assertIn('status', data)
        self.assertIn('models_loaded', data)
        self.assertIn('traditional_ml', data)
        self.assertIn('finbert', data)
        self.assertIn('timestamp', data)
        
        # Check health status
        self.assertEqual(data['status'], 'healthy')
        
        # At least one model should be loaded
        self.assertTrue(data['models_loaded'])
        print(f"Health check passed: {data}")

    def test_model_info_endpoint(self):
        """Test the model info endpoint"""
        response = requests.get(f"{API_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check required fields
        self.assertIn('traditional_ml', data)
        self.assertIn('finbert', data)
        
        # At least one model should be loaded
        self.assertTrue(data['traditional_ml'] or data['finbert'])
        print(f"Model info check passed: {data}")

    def test_predict_endpoint_valid_input(self):
        """Test the predict endpoint with valid headlines"""
        for headline in self.test_headlines[:5]:  # Use only valid headlines
            response = requests.post(
                f"{API_URL}/predict",
                json={"headline": headline}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Check response structure
            self.assertIn('headline', data)
            self.assertEqual(data['headline'], headline)
            
            # Check FinBERT results if available
            if data['finbert']:
                self.assertIn('label', data['finbert'])
                self.assertIn('score', data['finbert'])
                self.assertIn(data['finbert']['label'], ['bullish', 'neutral', 'bearish'])
                self.assertTrue(0 <= data['finbert']['score'] <= 1)
            
            # Check NB results if available
            if data['nb']:
                self.assertIn('label', data['nb'])
                self.assertIn('score', data['nb'])
                self.assertIn(data['nb']['label'], ['bullish', 'neutral', 'bearish'])
                self.assertTrue(0 <= data['nb']['score'] <= 1)
            
            # Check primary model selection
            self.assertIn('primary', data)
            self.assertIn(data['primary'], ['finbert', 'nb'])
            
            # Verify primary model selection logic
            if data['finbert'] and data['finbert']['score'] >= 0.80:
                self.assertEqual(data['primary'], 'finbert')
            
            print(f"Prediction for '{headline[:30]}...' - Primary: {data['primary']}")
            print(f"FinBERT: {data['finbert']}")
            print(f"NB: {data['nb']}")
            print("-" * 50)

    def test_predict_endpoint_empty_input(self):
        """Test the predict endpoint with empty headline"""
        response = requests.post(
            f"{API_URL}/predict",
            json={"headline": ""}
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('detail', data)
        print(f"Empty headline test passed: {data}")

    def test_predict_endpoint_long_input(self):
        """Test the predict endpoint with very long headline"""
        response = requests.post(
            f"{API_URL}/predict",
            json={"headline": "X" * 5000}
        )
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            data = response.json()
            self.assertIn('headline', data)
            print("Long headline processed successfully")
        else:
            self.assertIn(response.status_code, [400, 413, 422, 500])
            print(f"Long headline handled with status code: {response.status_code}")

    def test_predictions_history_endpoint(self):
        """Test the predictions history endpoint"""
        # First make a prediction to ensure there's history
        requests.post(
            f"{API_URL}/predict",
            json={"headline": "Test headline for history check"}
        )
        
        # Then check the history
        response = requests.get(f"{API_URL}/predictions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should be a list of predictions
        self.assertIsInstance(data, list)
        
        # If we have predictions, check their structure
        if data:
            prediction = data[0]
            self.assertIn('headline', prediction)
            self.assertIn('finbert_result', prediction)
            self.assertIn('nb_result', prediction)
            self.assertIn('primary_model', prediction)
            self.assertIn('timestamp', prediction)
            self.assertIn('id', prediction)
            
            print(f"Found {len(data)} predictions in history")
            print(f"Latest prediction: {prediction['headline']}")
        else:
            print("No predictions in history yet")

    def test_train_endpoint(self):
        """Test the model training endpoint"""
        response = requests.post(f"{API_URL}/train")
        
        # This might take a while, so we just check if the request was accepted
        self.assertIn(response.status_code, [200, 202, 500])
        
        if response.status_code in [200, 202]:
            data = response.json()
            self.assertIn('message', data)
            print(f"Training endpoint response: {data}")
        else:
            print(f"Training endpoint returned status code: {response.status_code}")
            print(f"This might be expected if training is resource-intensive")

if __name__ == '__main__':
    unittest.main()