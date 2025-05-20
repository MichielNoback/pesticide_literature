#!/Users/michielnoback/opt/miniconda3/envs/nlp/bin/python

import os
import sys
from configparser import ConfigParser

import pandas as pd

import tensorflow as tf
import scripts.utils as utils

# global settings
global_vars = {}
global_vars['verbose'] = False
global_vars['config'] = None
global_vars['model'] = None
global_vars['pesticide_papers'] = None
global_vars['new_papers'] = None

def check_command_line(argv):
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist.")
        sys.exit(1)


def read_config(config_file):
    """
    Read the config file and store as global variable.
    Also performs some checks on required options
    """
    global config
    # global global_vars

    valid_config = True
    config = ConfigParser()
    config.read(config_file)
    if config.has_option('RUN_SETTINGS', 'verbose'):
        global_vars['verbose'] = config.getboolean('RUN_SETTINGS', 'verbose')
        print(f"Verbose mode is {'on' if global_vars['verbose'] else 'off'}")
    if not config.has_option('MODEL', 'classification_model_file'):
        print("Config file does not contain a [MODEL] section with 'classification_model_file' option.")
        valid_config = False
    if not config.has_option('DATA', 'pesticide_papers_file'):
        print("Config file does not contain a [DATA] section with 'pesticide_papers_file' option.")
        valid_config = False
    if not valid_config:
        print("Config file is not valid. Exiting.")
        sys.exit(1)

def load_model():
    """
    Load the model from the config file.
    """
    # Assuming the model is a simple text file for this example
    model_file = config.get('MODEL', 'classification_model_file')
    if not os.path.exists(model_file):
        print(f"Model file {model_file} does not exist.")
        sys.exit(1)
    
    with open(model_file, 'r') as f:
        print(f'Loading model from {model_file}...')
        model = tf.keras.models.load_model(model_file)
        global_vars['model'] = model
    print("Model loaded successfully.")
    if global_vars['verbose']:
        print(f"... Model summary:")
        model.summary()


def load_pesticide_papers():
    """
    Load the pesticide papers from the config file.
    """
    pesticide_papers_file = config.get('DATA', 'pesticide_papers_file')
    if not os.path.exists(pesticide_papers_file):
        print(f"Pesticide papers file {pesticide_papers_file} does not exist.")
        sys.exit(1)

    pesticide_papers = pd.read_csv(pesticide_papers_file, sep='\t')
    # preprocess the data
    utils.preprocess_text(pesticide_papers, remove_punctuation=True, lower_case=True, remove_digits=True)
    #print(pesticide_papers.head())
    # print papers details of one paper
    #utils.print_paper(pesticide_papers.iloc[2]['pmid'], pesticide_papers)

    global_vars['pesticide_papers'] = pesticide_papers
    print("Pesticide papers loaded successfully.")
    if global_vars['verbose']:
        print(f"... Shape of pesticide papers dataframe: {pesticide_papers.shape}")
        print(f"... Column names: {pesticide_papers.columns.tolist()}")

def load_new_papers():
    """
    Load the new papers from the config file.
    """
    new_papers_file = config.get('DATA', 'new_papers_file')
    if not os.path.exists(new_papers_file):
        print(f"New papers file {new_papers_file} does not exist.")
        sys.exit(1)

    new_papers = pd.read_csv(new_papers_file, sep='\t')
    # preprocess the data
    utils.preprocess_text(new_papers, remove_punctuation=True, lower_case=True, remove_digits=True)

    global_vars['new_papers'] = new_papers
    print("New papers loaded successfully.")
    if global_vars['verbose']:
        print(f"... Shape of new papers dataframe: {new_papers.shape}")
        print(f"... Column names: {new_papers.columns.tolist()}")

def classify_new_papers():
    """
    Classify the new papers using the model.
    """
    new_papers = global_vars['new_papers']
    model = global_vars['model']

    # Assuming the model expects a specific input format
    # For example, if the model expects a list of texts
    texts = new_papers['text'].tolist()
    
    # Make predictions
    predictions = model.predict(texts)
    
    # Assuming the model outputs probabilities for each class
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    
    # Add predictions to the dataframe
    new_papers['predicted_class'] = predicted_classes
    
    print("New papers classified successfully.")
    if global_vars['verbose']:
        print(f"... Predictions: {predicted_classes}")

def main():
    check_command_line(sys.argv)
    read_config(sys.argv[1])
    # load_model()
    load_pesticide_papers()
    # load_new_papers()
    # classify_new_papers()


if __name__ == "__main__":
    main()
