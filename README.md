# A BERT-Powered Email Classifier System for Sales Representatives
Final Project for LING 227 - Language and Computation I (SP24)
Project Author - Luke Weinbach
Course Instructor - Tom McCoy
Date of Submission - 05/06/2024

This README describes the contents of the project directory. For a more detailed overview of the project's purpose and functionality, see the written report, which is available on my LinkedIn page. Or, reach out to me via email at luke.weinbach@yale.edu

**NOTE** This is a publicly available repository designed to exhibit the code for the project. The data-containing files "raw-data-final.xlsx" and "clean-data-final.csv" are not available for viewing, as they contain too much personal/company information of the sales team with whom I collaborated on this project.

## File Overview

The raw data used for the project is contained within
    raw-data-final.xlsx
and cleaned by dataclean.py to produce the actual training data input file
    clean-data-final.csv

**NOTE** The data used in this project is a set of real email threads from sales representatives at Reporting Xpress. Each example was hand-labeled with the appropriate category. This data was acquired with permission from Reporting Xpress for use in this project.

The directory 'model-save-0.95' contains the product of a model training run with a 95% accuracy rate. This is the training run discussed in the paper and in my presentation.

The directories 'save_model_here' and 'test_trainer' are empty, and installed within this directory by default to give users running the code in bert.py a place to save the model run if they choose

**NOTE** The trainer function in bert.py is set to store model output in test_trainer by default

## Code Files and Their Dependencies

dataclean.py is the code used to clean up the raw excel data and export it to a new csv file for use in training the model
dataclean.py depends on the Python modules:
    re,
    pandas

bert.py is the implementation of the model, including setup and training
bert.py depends on the Python modules:
    From PyTorch:
        torch,
        torch.nn
    From HuggingFace:
        transformers,
        datasets,
        evaluate
    From Scikit-learn
        sklearn,
        sklearn.model_selection
    From Python Standard Library
        pandas,
        numpy
