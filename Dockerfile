# set the base image (host OS)
FROM python:3.6.12

# set the working directory in the container
workdir /adult_salary

# copy dependencies file to the working directory
# source folder -> destination folder
ADD src /adult_salary/src

# install dependencies for model training
RUN pip install -r src/logistic_regression/requirements.txt

# add the model package to python path
ENV PYTHONPATH "/adult_salary/src/logistic_regression/:/adult_salary/src/ml_api"

# NOTE: model is trained in the build as packaged model is used by the API which is the output of the training
# train logistic_regression model
RUN python /adult_salary/src/logistic_regression/logistic_regression/train_pipeline.py

# test single prediction output of the model
RUN pytest /adult_salary/src/logistic_regression/tests

# deploy the model locally
RUN python /adult_salary/src/logistic_regression/setup.py sdist bdist_wheel

# install dependencies for ml_api
RUN pip install -r src/ml_api/requirements.txt

#for debugging purposes
#ENTRYPOINT ["tail", "-f", "/dev/null"]

