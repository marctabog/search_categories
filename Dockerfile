FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
ADD dico_general.pickle dico_general.pickle
ADD dico.pickle dico.pickle
ADD embedding.pickle embedding.pickle
ADD data_model.csv data_model.csv
ADD app.py app.py

# Install required libraries
RUN pip install -r requirements.txt

# Run it once to trigger resnet download
RUN python app.py

EXPOSE 8008

# Start the server
CMD ["python", "app.py", "serve"]
