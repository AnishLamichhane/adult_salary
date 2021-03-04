## Predicting adult salary
[![Build status](https://badge.buildkite.com/2541d5fd605b982f6ce95b1fe75894143ae00d7113831a8e39.svg)](https://buildkite.com/al/adult-salary)

Predicting if an adult will earn less or more than $50k based on data collected in census data 1994. Two packages are available: *logistic_regression* for
training model and deploying pipeline and *ml_api* for deploying flask api for accessing the prediction functionality 
of logistic regression model. The docker- compose and docker file is provided to deploy packages in docker
container. The project has been tested by running pytest for a model deployment and api's prediction's endpoint for a sample data. 
 
- Data source:  "https://archive.ics.uci.edu/ml/datasets/adult"
- Initial data configuration
    - change the file name and extension of the data source.
        - adult.data -> adult_data.txt
        - adult.test -> adult_test.txt
- jupyter notebook contains the initial investigation of data, feature engineering, feature selection
 and model tuning. Model parameter is extracted.
- There is two packages inside src
    - logistic_regression
    - ml_api 
- Logistic regression packages
    - trains the prediction packages and deploys the packages. 
    - Logistic regression is used to algorithm as the ml model.
    - tox is used for testing the model in different python environment
    - Outputs dictionary of result with keys : Prediction and Version. 
        - Prediction is an array of array. The first array has value:
            - 1 = Salary >50K
            - 0 = Salary < 50K
        - The second array is the probability of the outcome.
    - AUC and F1 score was used to evaluate the model performance.
        - Train F1 score: 0.8876925434899227
        - Train roc-auc: 0.8726128068624428
        - Tests F1 score: 0.8888213158995021
        - Tests roc-auc: 0.8735438319824711
- ml_api
    - uses Flask to create to create endpoints to access the prediction model.
    - logistic regression model need to be trained and a model packaged before running the ml_api
- instruction to run the code
    - for local run
        - run tox within the logistic regression folder. 
            - it will train model pipeline and deploy the model locally.
        - under ml_api: python run.py - this will start the flask api
    - Docker deployment
        - to deploy with docker compose
            - docker-compose run
    - the current project runs pytest for model building and deployment and a second pytest
    for the api to check if the prediction end point is running.
- Notes:
    - the input data sequence need to be matched with what the model expects or the prediction will not be
    correct and it may throw error. 
    - unit test and try catch block for input validation is still needs to be added.
    
    

    


