import React, { useState } from 'react';
import './App.css';
import FaceRecognition from './components/FaceRecognition';
import Training from './components/Training';

function App() {
  const [activeTab, setActiveTab] = useState('recognition');

  return (
    <div className="App">
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'recognition' ? 'active' : ''}`}
          onClick={() => setActiveTab('recognition')}
        >
          👤 Face Recognition
        </button>
        <button
          className={`tab ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          🧠 Training
        </button>
      </div>
      {activeTab === 'recognition' && <FaceRecognition />}
      {activeTab === 'training' && <Training />}
    </div>
  );
}

export default App;






