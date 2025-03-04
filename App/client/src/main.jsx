import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// Must import the App (main) React component to render it when we run "$npm run dev"
import App from './App.jsx'

/**
 * Entry point for the React application. Initializes the React app and renders
 * the App component inside the root HTML element.
 */

// document.getElementById('root') finds the <div id="root"> in the index.html file
// Then createRoot() creates a React root and attaches it to the root div found in index.html
// To actually render and run the React App
createRoot(document.getElementById('root')).render(
  <StrictMode>
    {/* Must call the App React component to finally render it */}
    <App />
  </StrictMode>,
)
