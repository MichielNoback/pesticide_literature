#!/Users/michielnoback/opt/miniconda3/envs/nlp/bin/python

import os
import sys
from configparser import ConfigParser
from typing import List

import pandas as pd # type: ignore
from sklearn.feature_extraction.text import  TfidfVectorizer # type: ignore

import tensorflow as tf # type: ignore
import scripts.utils as utils
import scripts.pesticides as pesticides

# global variables
config = None
global_vars = {}
global_vars['verbose'] = False
global_vars['config'] = None
global_vars['model'] = None
global_vars['pesticide_papers'] = None
global_vars['new_papers'] = None
global_vars['vectorizer'] = None


def check_command_line(argv: List[str]) -> None:
    """
    Check the command line arguments.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist.")
        sys.exit(1)


def read_config(config_file: str) -> None:
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


def load_model() -> None:
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


def load_pesticide_papers() -> None:
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


def make_vectorizer() -> TfidfVectorizer:
    """
    Make the vectorizer for the pesticide papers.
    """
    vectorizer = utils.make_vectorizer()
    global_vars['vectorizer'] = vectorizer
    print("Vectorizer created successfully.")
    if global_vars['verbose']:
        print(f"... Vectorizer: {vectorizer}") 


def make_embeddings(papers: pd.DataFrame, 
                    storage_key: str,
                    only_transform: bool) -> None: 
    """
    Make embeddings for the pesticide papers.
    """
    embeddings = utils.make_embeddings(global_vars['vectorizer'], papers, only_transform=only_transform)
    global_vars[storage_key] = embeddings
    
    print("Embeddings created successfully.")
    if global_vars['verbose']:
        print(f"... Embeddings type: {type(embeddings)}")
        print(f"... Shape of TF-IDF embeddings: {embeddings.shape}")


def load_new_papers() -> None:
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


def mark_new_papers_with_pesticide_terms() -> None:
    """
    Mark the new papers with pesticide terms.
    """
    new_papers = global_vars['new_papers']
    #print(pesticides.pesticide_terms)
    new_pesticide_papers = utils.find_pesticide_terms(new_papers, pesticides.pesticide_terms)
    # store in global_vars
    global_vars['new_pesticide_papers'] = new_pesticide_papers
    print("New papers with pesticide terms marked.")
    if global_vars['verbose']:
        print(f"... Number of new pesticide papers: {len(new_pesticide_papers)}")
        print(f"... Column names: {new_pesticide_papers.columns.tolist()}")
        print(new_pesticide_papers)


def find_nearest_neighbors() -> None:
    """
    Find the nearest neighbors of the new papers.
    """
    pesticide_papers = global_vars['pesticide_papers']
    new_pesticide_papers = global_vars['new_pesticide_papers']
    pesticide_paper_embeddings = global_vars['pesticide_papers_embeddings']
    new_pesticide_paper_embeddings = global_vars['new_pesticide_papers_embeddings']
    new_pesticide_papers = utils.find_nearest_neighbors(pesticide_papers, 
                                                        new_pesticide_papers, 
                                                        pesticide_paper_embeddings, 
                                                        new_pesticide_paper_embeddings)
    print("Nearest neighbors found.")
    if global_vars['verbose']:
        print(f"... Column names: {new_pesticide_papers.columns.tolist()}")
        print(new_pesticide_papers[['pmid', 'title', 'found_terms']])
        # print(new_pesticide_papers[['pmid', 'title', 'nearest_neighbors']])

    # # load the model
    # model = global_vars['model']
    # # find the nearest neighbors
    # utils.find_nearest_neighbors(new_pesticide_papers, model)
    # print("Nearest neighbors found.")
    # if global_vars['verbose']:
    #     print(f"... Number of nearest neighbors: {len(new_pesticide_papers)}")
    #     #print(f"... Column names: {new_pesticide_papers.columns.tolist()}")
    #     print(new_pesticide_papers)


def perform_keyword_extraction():
    """
    Perform keyword extraction on the new papers.
    """
    new_pesticide_papers = global_vars['new_pesticide_papers']
    utils.perform_keyword_extraction(new_pesticide_papers)
    raise RuntimeError("This function is not implemented yet.")


def main() -> None:
    """
    Main function to run the pipeline.
    """
    ## INITIALIZE
    check_command_line(sys.argv)
    read_config(sys.argv[1])
    # load_model()
    make_vectorizer()

    ## LOAD DATA
    load_pesticide_papers()
    load_new_papers()

    ## FIND CANDIDATE PESTICIDE PAPERS
    mark_new_papers_with_pesticide_terms()

    ## CREATE EMBEDDINGS
    make_embeddings(global_vars['pesticide_papers'], 'pesticide_papers_embeddings', only_transform=False)
    #global_vars['pesticide_papers_embeddings'] = pesticide_papers_embeddings
    make_embeddings(global_vars['new_pesticide_papers'], 'new_pesticide_papers_embeddings', only_transform=True)
    #global_vars['new_pesticide_paper_embeddings'] = new_pesticide_paper_embeddings

    ## FIND NEAREST NEIGHBORS
    find_nearest_neighbors()

    #perform_keyword_extraction()


if __name__ == "__main__":
    main()
