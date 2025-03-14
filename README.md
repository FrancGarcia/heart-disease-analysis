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
### Data visualization:
**[SubmissionFiles](./SubmissionFiles/)**  
- **Our Final version of results and code**
- Data visualization modules and visualization plots

### Data Processing:
[data](./data/)  
- Original dataset
- After cleaning datasets using hds, kmodes and majority

[DataProcessing](./DataProcessing/)  
- Data cleaning code for the dataset

### Web App related:  
-[App](./App/)  
- Contains code for our webapp  

[backend](./Backend/)  
- Contains backend code for our webapp  


### Dependencies
[dependencies](./dependencies/)  
- Contains the python library requirements 


## Authors:
- Francisco Garcia
- Qiyue Zhang
- Chengming Li
- Trivikram Choudhury
- Shail Vaidya


