# File descriptions

- ActiveSubstanceExport_24-03-2025.xlsx
    > Downloaded from https://ec.europa.eu/food/plant/pesticides/eu-pesticides-database/start/screen/active-substances. (Plants > Pesticides > EU pesticides database > active substances) as is a an excel (`ActiveSubstanceExport_24-03-2025.xlsx`) with 1465 substances, all with scientific name only, it seems.
- abstract_set1.txt
    > downloaded from https://ironman.tenwisedev.nl/public/abstract_set1_michiel.txt. on feb 6, 2025 It was undisclosed which category the abstracts belonged to, but it is quite obvious that set1 contains pesticide-related abstracts. 
- abstract_set2.txt
    > downloaded from https://ironman.tenwisedev.nl/public/abstract_set2_michiel.txt. on feb 6, 2025 It was undisclosed which category the abstracts belonged to, but it is quite obvious that set2 contains disease-related abstracts
- abstracts_2025_04_09.csv
    > a single-day PubMed production of published papers
- common_names.csv
    > Downloaded from http://www.bcpcpesticidecompendium.org/index_cn_frame.html and parsed to simple text file on march 31, 2025
- curated_pesticides.txt
    > pubmed IDs for papers that were labeled by Wynand to belong to one of five groups: human, eco-animal, insect, animal, other. This file (obtained on april 10, 2025 from https://ironman.tenwisedev.nl/public//curated_pesticides.txt) contains 164 entries
- drugLib_raw.tsv
    > test file for Transformer experimentation
- parse_common_names.py
    > To parse data from http://www.bcpcpesticidecompendium.org/index_cn_frame.html
- pesticide_classes.csv
    > a small text file containing the general classes for pesticied (e.g. herbicides, fungicides etc.)
- tf_idf_keyword_extraction_test.csv
    > results of an experiment to generate keywords using a tf-idf embedding. See log for details 
- tf_idf_vocabulary_size_test.csv
    > It was investigated how small a TF-IDF can be for finding a correct recommendations using nearest neighbour search
    