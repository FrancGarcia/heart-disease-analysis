# UCSD-ECE143-heart-disease-analysis
## Components:
### Analysis Pipeline with data cleaning and visualization
### ML Model to predict patient outcome given their data
### Simple webapp with the ML model running in the background on Flask API and simple frontend using React + Vite

To run the app:
1) Open one terminal and run $ cd App/server
2) Then in that same terminal activate the virtual environment with $ \venv\Scripts\activate
3) Then run the command $ python main.py to activate the backend server and follow the provided link after execution
4) Open another terminal (without closing the first terminal) and run $ cd App/client
5) Then in that same terminal run $ npm run dev to activate the frontend and follow the provided link after execution
6) Try out the frontend to see if you're susceptible to heart disease!


## Folder Structure
### 1. Data visualization:
- **[SubmissionFiles](./SubmissionFiles-Final-Version/)**  
    - **Our final submission of results and code for the project**
    - Data cleaning
    - Data visualization modules 
    - Visualization plots

### 2. Data Processing:
- [data](./data/)  
    - Original dataset
    - After cleaning datasets using hds, kmodes and majority

- [DataProcessing-Backup](./DataProcessing-Backup/)  
    - Old Version - dataset cleaning code 
    - Testing code for data processing

### 3. ML Model:  
- [ML-Model](./ML-Model/)  
    - Contains modularized codes of how to train the model
    - Model itself in .pth
    - Plots of ML 
    - Testing code for ML trainning

### 4. Web App related:  
- [App](./App/)  
    - Contains code for our webapp  


### 5. Dependencies
- [dependencies](./dependencies/)  
    - Contains the python library requirements 


## Authors:
- Francisco Garcia
- Qiyue Zhang
- Chengming Li
- Trivikram Choudhury
- Shail Vaidya


