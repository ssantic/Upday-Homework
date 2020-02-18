Upday Data Science Challenge
============================

This README file describes the work done by candidate Srdjan Santic, for the upday Data Science challenge.

The solution contains multiple files, namely:

* Upday_DS_Challenge-Srdjan_Santic.ipynb
* requirements.txt
* train.py
* test.py
* app.py
* sample.json
* README.md

A description of the files follows.


Upday_DS_Challenge-Srdjan_Santic.ipynb
--------------------------------------

This is the main Jupyter notebook, in which the data analysis, text normalization, feature engineering,
modeling, and reporting of metrics is performed. The cells can be run in order, and it requires the
data_truncated.tsv file. It is meant to show off all of the work performed.


requirements.txt
----------------

The project relies on a number of dependencies that need to first be installed. This can be done by way of pip:

$ pip install -r requirements.txt


train.py
--------

This is a command line Python script, which does all of the text preprocessing, normalization, and feature engineering
steps. It trains a Random Forest model, and exports it as a .pickle file. It also outputs a TSV file with the test set
for future use.

It is run in the following way:

$ python train.py dataset_file


test.py
-------

This command line script is meant to be run after train.py. It loads the pickled model, as well as the test dataset,
and presents several accuracy metrics.

It is run in the following way:

$ python test.py test_dataset_file model_file


app.py
------

This is a minimal Flask web server, listening on port 5000. It loads the pickled model, and can be tested via,
for example, Postman. It's run simply:

$ python app.py


sample.json
-----------

This is a sample test file, to be sent to the Flask API gateway, in order to test the server and model. This can be done
using an application such as postman.

