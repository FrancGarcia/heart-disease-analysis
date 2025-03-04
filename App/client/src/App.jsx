import React, { useState } from 'react';
import './App.css';

function App() {
  const [age, setAge] = useState('');
  const [weight, setWeight] = useState('');
  const [cholesterol, setCholesterol] = useState('');
  const [bloodPressure, setBloodPressure] = useState('');
  const [sugar, setSugar] = useState('');
  const [familyHistory, setFamilyHistory] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Create data object from the form
    const data = {
      age,
      weight,
      cholesterol,
      blood_pressure: bloodPressure,
      sugar,
      family_history: familyHistory,
    };

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();
      if (result.prediction) {
        setPrediction(result.prediction);
      } else {
        setPrediction(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error:', error);
      setPrediction('Error occurred while making prediction');
    }
  };

  return (
    <div className="form-container">
      <h2>Heart Disease Prediction</h2>
      <form onSubmit={handleSubmit}>
        <label>Age:</label>
        <input
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          required
        />
        <label>Weight (kg):</label>
        <input
          type="number"
          value={weight}
          onChange={(e) => setWeight(e.target.value)}
          required
        />
        <label>Cholesterol Level:</label>
        <input
          type="number"
          value={cholesterol}
          onChange={(e) => setCholesterol(e.target.value)}
          required
        />
        <label>Blood Pressure:</label>
        <input
          type="number"
          value={bloodPressure}
          onChange={(e) => setBloodPressure(e.target.value)}
          required
        />
        <label>Sugar Level (1 for high, 0 for low):</label>
        <input
          type="number"
          value={sugar}
          onChange={(e) => setSugar(e.target.value)}
          required
        />
        <label>Family History of Heart Disease (1 for yes, 0 for no):</label>
        <input
          type="number"
          value={familyHistory}
          onChange={(e) => setFamilyHistory(e.target.value)}
          required
        />
        <button type="submit">Submit</button>
      </form>

      <h2>{prediction}</h2>
    </div>
  );
}

export default App;
