# set the base image (host OS)
FROM python:3.6

# set the working directory in the container
workdir /adult_salary

# copy dependencies file to the working directory
# source folder -> destination folder
ADD src /adult_salary/src

# install dependencies
RUN pip install -r src/logistic_regression/requirements.txt

ENV PYTHONPATH "/adult_salary/src/logistic_regression/"

#ENTRYPOINT ["tail", "-f", "/dev/null"]

