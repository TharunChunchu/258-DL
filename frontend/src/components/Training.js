import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './Training.css';

const Training = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [epochs, setEpochs] = useState(50);
  const [progress, setProgress] = useState(null);
  const [history, setHistory] = useState({
    train_losses: [],
    train_accuracies: [],
    val_losses: [],
    val_accuracies: []
  });
  const [status, setStatus] = useState(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    // Check initial training status
    checkStatus();
    
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkStatus = async () => {
    try {
      const response = await axios.get('/api/training/status');
      const data = response.data;
      setIsTraining(data.is_training);
      if (data.is_training) {
        setProgress({
          epoch: data.current_epoch,
          total_epochs: data.total_epochs,
          train_loss: data.train_loss,
          train_acc: data.train_acc,
          val_loss: data.val_loss,
          val_acc: data.val_acc
        });
        setHistory(data.history);
        startProgressStream();
      }
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  const startTraining = async () => {
    try {
      const response = await axios.post('/api/training/start', { epochs });
      if (response.data.success) {
        setIsTraining(true);
        setProgress(null);
        setHistory({
          train_losses: [],
          train_accuracies: [],
          val_losses: [],
          val_accuracies: []
        });
        setStatus(null);
        startProgressStream();
      }
    } catch (error) {
      alert('Error starting training: ' + (error.response?.data?.error || error.message));
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('/api/training/stop');
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      setIsTraining(false);
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  const startProgressStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource('/api/training/progress');
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.heartbeat) {
          return;
        }

        if (data.error) {
          alert('Training error: ' + data.error);
          setIsTraining(false);
          eventSource.close();
          return;
        }

        if (data.status === 'data_loaded') {
          const distMsg = data.class_distribution 
            ? `Class distribution: ${Object.entries(data.class_distribution).map(([k, v]) => `${k}: ${v}`).join(', ')}`
            : '';
          const weightsMsg = data.class_weights
            ? `Class weights: ${Object.entries(data.class_weights).map(([k, v]) => `${k}: ${v}`).join(', ')}`
            : '';
          setStatus({
            type: 'info',
            message: `Loaded ${data.train_samples} training, ${data.val_samples} validation, ${data.test_samples} test samples. ${distMsg} ${weightsMsg}`,
            classes: data.classes,
            distribution: data.class_distribution,
            weights: data.class_weights
          });
        } else if (data.status === 'model_created') {
          setStatus({
            type: 'info',
            message: `Model created with ${data.total_parameters.toLocaleString()} parameters, ${data.num_classes} classes`
          });
        } else if (data.status === 'training_started') {
          setStatus({
            type: 'success',
            message: 'Training started!'
          });
        } else if (data.status === 'completed') {
          setStatus({
            type: 'success',
            message: `Training completed! Test Accuracy: ${data.test_accuracy.toFixed(2)}%`
          });
          setIsTraining(false);
          eventSource.close();
        } else if (data.status === 'early_stopping') {
          setStatus({
            type: 'warning',
            message: `Early stopping triggered. Best validation accuracy: ${data.best_val_acc.toFixed(2)}%`
          });
        } else if (data.epoch) {
          // Progress update
          setProgress(data);
          
          // Update history
          setHistory(prev => ({
            train_losses: [...prev.train_losses, data.train_loss],
            train_accuracies: [...prev.train_accuracies, data.train_acc],
            val_losses: [...prev.val_losses, data.val_loss],
            val_accuracies: [...prev.val_accuracies, data.val_acc]
          }));
        }
      } catch (error) {
        console.error('Error parsing progress:', error);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setIsTraining(false);
    };
  };

  return (
    <div className="training-container">
      <div className="training-card">
        <div className="training-header">
          <h1>🧠 Model Training</h1>
          <p>Train the face recognition model with real-time progress</p>
        </div>

        {!isTraining && (
          <div className="training-controls">
            <div className="input-group">
              <label>Number of Epochs:</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                min="1"
                max="200"
                disabled={isTraining}
              />
            </div>
            <button
              className="btn btn-primary btn-large"
              onClick={startTraining}
              disabled={isTraining}
            >
              🚀 Start Training
            </button>
          </div>
        )}

        {isTraining && (
          <div className="training-controls">
            <button
              className="btn btn-danger"
              onClick={stopTraining}
            >
              ⏹ Stop Training
            </button>
          </div>
        )}

        {status && (
          <div className={`status-message status-${status.type}`}>
            {status.message}
            {status.classes && (
              <div className="classes-list">
                Classes: {status.classes.join(', ')}
              </div>
            )}
            {status.distribution && (
              <div className="distribution-list">
                <strong>Distribution:</strong> {Object.entries(status.distribution).map(([k, v]) => `${k}: ${v}`).join(', ')}
              </div>
            )}
            {status.weights && (
              <div className="weights-list">
                <strong>Weights (for balancing):</strong> {Object.entries(status.weights).map(([k, v]) => `${k}: ${v}`).join(', ')}
              </div>
            )}
          </div>
        )}

        {progress && (
          <div className="progress-section">
            <div className="progress-header">
              <h2>Training Progress</h2>
              <div className="epoch-info">
                Epoch {progress.epoch} / {progress.total_epochs}
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Training Loss</div>
                <div className="metric-value">{progress.train_loss.toFixed(4)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Training Accuracy</div>
                <div className="metric-value">{progress.train_acc.toFixed(2)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Validation Loss</div>
                <div className="metric-value">{progress.val_loss.toFixed(4)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Validation Accuracy</div>
                <div className="metric-value">{progress.val_acc.toFixed(2)}%</div>
              </div>
            </div>

            {history.train_losses.length > 0 && (
              <div className="charts-container">
                <div className="chart">
                  <h3>Loss</h3>
                  <div className="chart-placeholder">
                    <svg viewBox="0 0 400 200" className="chart-svg">
                      <polyline
                        points={history.train_losses.map((loss, i) => 
                          `${(i / (history.train_losses.length - 1 || 1)) * 380 + 10},${190 - (loss / Math.max(...history.train_losses, 1)) * 170}`
                        ).join(' ')}
                        fill="none"
                        stroke="#667eea"
                        strokeWidth="2"
                      />
                      <polyline
                        points={history.val_losses.map((loss, i) => 
                          `${(i / (history.val_losses.length - 1 || 1)) * 380 + 10},${190 - (loss / Math.max(...history.val_losses, 1)) * 170}`
                        ).join(' ')}
                        fill="none"
                        stroke="#e74c3c"
                        strokeWidth="2"
                      />
                    </svg>
                    <div className="chart-legend">
                      <span className="legend-item"><span className="legend-color" style={{background: '#667eea'}}></span> Train</span>
                      <span className="legend-item"><span className="legend-color" style={{background: '#e74c3c'}}></span> Val</span>
                    </div>
                  </div>
                </div>
                <div className="chart">
                  <h3>Accuracy</h3>
                  <div className="chart-placeholder">
                    <svg viewBox="0 0 400 200" className="chart-svg">
                      <polyline
                        points={history.train_accuracies.map((acc, i) => 
                          `${(i / (history.train_accuracies.length - 1 || 1)) * 380 + 10},${190 - (acc / 100) * 170}`
                        ).join(' ')}
                        fill="none"
                        stroke="#27ae60"
                        strokeWidth="2"
                      />
                      <polyline
                        points={history.val_accuracies.map((acc, i) => 
                          `${(i / (history.val_accuracies.length - 1 || 1)) * 380 + 10},${190 - (acc / 100) * 170}`
                        ).join(' ')}
                        fill="none"
                        stroke="#f39c12"
                        strokeWidth="2"
                      />
                    </svg>
                    <div className="chart-legend">
                      <span className="legend-item"><span className="legend-color" style={{background: '#27ae60'}}></span> Train</span>
                      <span className="legend-item"><span className="legend-color" style={{background: '#f39c12'}}></span> Val</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Training;



