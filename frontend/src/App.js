import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const SentimentAnalyzer = () => {
  const [headline, setHeadline] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchModelInfo();
    fetchHistory();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API}/model-info`);
      setModelInfo(response.data);
    } catch (e) {
      console.error("Error fetching model info:", e);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API}/predictions?limit=10`);
      setHistory(response.data);
    } catch (e) {
      console.error("Error fetching history:", e);
    }
  };

  const analyzeSentiment = async () => {
    if (!headline.trim()) {
      setError("Please enter a headline");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const response = await axios.post(`${API}/predict`, {
        headline: headline.trim()
      });
      setPrediction(response.data);
      fetchHistory(); // Refresh history
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to analyze sentiment");
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'bullish': return 'text-green-600 bg-green-100';
      case 'bearish': return 'text-red-600 bg-red-100';
      case 'neutral': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'bullish': return '📈';
      case 'bearish': return '📉';
      case 'neutral': return '➡️';
      default: return '❓';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🔍 Forex Sentiment Analyzer
          </h1>
          <p className="text-lg text-gray-600">
            Advanced NLP analysis for forex news headlines
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Powered by FinBERT & Traditional ML Models
          </p>
        </div>

        {/* Model Status */}
        {modelInfo && (
          <div className="bg-white rounded-lg shadow-md p-4 mb-6">
            <h3 className="font-semibold text-gray-800 mb-2">Model Status</h3>
            <div className="flex gap-4">
              <div className={`px-3 py-1 rounded-full text-sm ${
                modelInfo.finbert ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                FinBERT: {modelInfo.finbert ? 'Loaded' : 'Not Available'}
              </div>
              <div className={`px-3 py-1 rounded-full text-sm ${
                modelInfo.traditional_ml ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                TF-IDF + NB: {modelInfo.traditional_ml ? 'Loaded' : 'Not Available'}
              </div>
            </div>
          </div>
        )}

        {/* Main Analysis Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Analyze Headline Sentiment
          </h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter Forex News Headline:
            </label>
            <textarea
              value={headline}
              onChange={(e) => setHeadline(e.target.value)}
              placeholder="e.g., EUR/USD surges to new monthly highs as ECB signals rate hikes"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows="3"
            />
          </div>

          <button
            onClick={analyzeSentiment}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg transition duration-200 flex items-center justify-center"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Analyzing...
              </>
            ) : (
              'Analyze Sentiment'
            )}
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700">{error}</p>
            </div>
          )}
        </div>

        {/* Results */}
        {prediction && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              Analysis Results
            </h3>
            
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <h4 className="font-medium text-gray-700 mb-2">Headline:</h4>
              <p className="text-gray-800 italic">"{prediction.headline}"</p>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              {/* FinBERT Results */}
              {prediction.finbert && (
                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-gray-800">FinBERT Model</h4>
                    {prediction.primary === 'finbert' && (
                      <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                        Primary
                      </span>
                    )}
                  </div>
                  <div className={`px-3 py-2 rounded-lg ${getSentimentColor(prediction.finbert.label)}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-medium">
                        {getSentimentIcon(prediction.finbert.label)} {prediction.finbert.label.toUpperCase()}
                      </span>
                      <span className="text-sm">
                        {(prediction.finbert.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Naive Bayes Results */}
              {prediction.nb && (
                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-gray-800">TF-IDF + NB</h4>
                    {prediction.primary === 'nb' && (
                      <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                        Primary
                      </span>
                    )}
                  </div>
                  <div className={`px-3 py-2 rounded-lg ${getSentimentColor(prediction.nb.label)}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-medium">
                        {getSentimentIcon(prediction.nb.label)} {prediction.nb.label.toUpperCase()}
                      </span>
                      <span className="text-sm">
                        {(prediction.nb.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Primary Model:</strong> {prediction.primary === 'finbert' ? 'FinBERT' : 'TF-IDF + Naive Bayes'}
                {prediction.primary === 'finbert' && prediction.finbert.score >= 0.80 && (
                  <span className="ml-2 text-xs">(High confidence ≥ 80%)</span>
                )}
              </p>
            </div>
          </div>
        )}

        {/* Recent Predictions History */}
        {history.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              Recent Predictions
            </h3>
            <div className="space-y-3">
              {history.slice(0, 5).map((item, index) => (
                <div key={item.id} className="border-l-4 border-blue-200 pl-4 py-2">
                  <p className="text-sm text-gray-600 italic">"{item.headline}"</p>
                  <div className="flex items-center gap-4 mt-1">
                    {item.finbert_result && (
                      <span className={`text-xs px-2 py-1 rounded ${getSentimentColor(item.finbert_result.label)}`}>
                        FinBERT: {item.finbert_result.label}
                      </span>
                    )}
                    {item.nb_result && (
                      <span className={`text-xs px-2 py-1 rounded ${getSentimentColor(item.nb_result.label)}`}>
                        NB: {item.nb_result.label}
                      </span>
                    )}
                    <span className="text-xs text-gray-500">
                      Primary: {item.primary_model}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Sample Headlines */}
        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Try These Sample Headlines:
          </h3>
          <div className="grid md:grid-cols-2 gap-2">
            {[
              "EUR/USD surges to new monthly highs as ECB signals rate hikes",
              "GBP/JPY remains flat amid mixed economic signals",
              "USD/CHF plunges on weak US employment data",
              "Gold prices rally as inflation concerns mount",
              "Bitcoin crashes below key support level",
              "Fed maintains dovish stance, dollar weakens"
            ].map((sample, index) => (
              <button
                key={index}
                onClick={() => setHeadline(sample)}
                className="text-left p-2 text-sm text-blue-600 hover:bg-blue-50 rounded border border-blue-200 hover:border-blue-300 transition"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <SentimentAnalyzer />
    </div>
  );
}

export default App;