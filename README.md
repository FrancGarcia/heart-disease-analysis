# heart-disease-analysis
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

## Folder Structure:
[App](./App/)  
- Contains code for our webapp  

[backend](./Backend/)  
- Contains backend code for our webapp  

[data](./data/)  
- Original, and after cleaning datasets

[DataProcessing](./DataProcessing/)  
- Data cleaning code for the dataset

[dependencies](./dependencies/)  
- Contains the python library requirements 

[SubmissionFiles](./SubmissionFiles/)  
- Data visualization modules and visualization plots

