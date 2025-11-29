import React, { useState } from 'react';
import axios from 'axios';
import './FaceRecognition.css';

const FaceRecognition = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.error) {
        setError(response.data.error);
      } else {
        setResult(response.data);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error processing image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="face-recognition-container">
      <div className="main-card">
        <div className="header">
          <h1>
            <span className="icon">👤</span> Face Recognition
          </h1>
          <p>Upload a photo to identify the person</p>
        </div>

        {/* Upload Area */}
        <div
          className="upload-area"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
        >
          {preview ? (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <button className="change-image-btn" onClick={(e) => {
                e.stopPropagation();
                handleReset();
              }}>
                Change Image
              </button>
            </div>
          ) : (
            <>
              <div className="upload-icon">📤</div>
              <h3>Upload Photo</h3>
              <p>Drag and drop an image here or click to select</p>
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </>
          )}
        </div>

        {/* Action Buttons */}
        {preview && (
          <div className="action-buttons">
            <button
              className="btn btn-primary"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Processing...
                </>
              ) : (
                <>
                  <span>🔍</span> Predict Name
                </>
              )}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              Reset
            </button>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {/* Result Card */}
        {result && (
          <div className="result-card">
            <div className="result-header">
              <span className="success-icon">✅</span>
              <h2>Prediction Result</h2>
            </div>
            <div className="result-content">
              <div className="prediction-name">
                {result.prediction}
              </div>
              <div className="confidence-score">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </div>
              <div className="faces-detected">
                Faces detected: {result.faces_detected}
              </div>
              {result.all_probabilities && (
                <div className="all-probabilities">
                  <h4>All Probabilities:</h4>
                  {Object.entries(result.all_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, prob]) => (
                      <div key={name} className="prob-item">
                        <span className="prob-name">{name}:</span>
                        <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                        <div className="prob-bar">
                          <div 
                            className="prob-bar-fill" 
                            style={{width: `${prob * 100}%`}}
                          ></div>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FaceRecognition;
















