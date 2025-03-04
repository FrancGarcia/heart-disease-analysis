import React, { useState } from 'react';
import './App.css';

/**
 * Used to initalize a React App component to export and render for the app UI
 * @return The React App component for export and use by the main.jsx file
 */
function App() {
  // Create the state variables for user input and initially set them to empty strings
  const [age, setAge] = useState('');
  const [weight, setWeight] = useState('');
  const [cholesterol, setCholesterol] = useState('');
  const [bloodPressure, setBloodPressure] = useState('');
  const [sugar, setSugar] = useState('');
  const [familyHistory, setFamilyHistory] = useState('');
  const [prediction, setPrediction] = useState('');

  // Prevents page from reloading when the form is submitted
  // This is called when the user clicks on the "Submit" button 
  // as handled by the onSubmit event-handler as seen below
  const handleSubmit = async (event) => {
    event.preventDefault();

    // Create data object from the form and store the user input in it to use in the HTTP POST
    const data = {
      age,
      weight,
      cholesterol,
      blood_pressure: bloodPressure,
      sugar,
      family_history: familyHistory,
    };

    // Send POST request to Flask backend as JSON format string via HTTP
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Convert the data from above into a JSON string
        body: JSON.stringify(data),
      });
      // Wait for the response and parse it into a JSON
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

  // Create the UI and return it as a React component
  return (
    <div className="form-container">
      <h2>Heart Disease Prediction</h2>
      {/* On submit button, render all of the React state variables with the user input, 
      package them into a JSON string as seen above, and use them to make a POST request to Flask backend
      using the handleSubmit React function above*/}
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
        {/* Attach the submit event-handler to call handleSubmit React
        function from above whenever the Submit button is clicked on */}
        <button type="submit">Submit</button>
      </form>
      {/* Display the prediction response received from the Flask backend */}
      <h2>{prediction}</h2>
    </div>
  );
}

export default App;
