# -*- coding: utf-8 -*-
"""
**Date :** Created on Tuesday January 12 2021  

**Group 8 - Innovation**

**integration_interface_web_v0** 

**@author :** Théo Saccareau. 
"""

##########################################################################
######################### NOTES POUR LE GROUPE 7 #########################
##########################################################################


### CODE A RAJOUTER Où VOUS VOULEZ (dans d'autres fichier .py):
    
    #(1) INSTALLATION LIBRAIRIES AVEC PIP 

    # * Install librairy
    #!pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier
    #!python -m spacy download fr_core_news_md
    #!pip3 install 'https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-2.3.0/fr_core_news_md-2.3.0.tar.gz'

    # OU AVEC SUBPROCESS
    #command = "pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier"
    #subprocess.call(command, shell=True)
    #command = "python -m spacy download fr_core_news_md"
    #subprocess.call(command, shell=True)
    #command = "pip3 install \'https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-2.3.0/fr_core_news_md-2.3.0.tar.gz\'"
    #subprocess.call(command, shell=True)

    # * Download Librairy
    #import nltk
    #nltk.download('punkt')
    #nltk.download('stopwords')
    #import spacy
    #import fr_core_news_m

    # (2) IMPORT 

    # * Useful library :
    #import pandas as pd
    #import numpy as np
    #from tqdm import trange
    #from google.colab import drive
    #import pickle
    #import subprocess

    # * System library :
    # import os 

    # * Text library : 
    #import pyterrier as pt
    #import fr_core_news_md
    #import re
    #import string
    #import unicodedata
    #from nltk import word_tokenize
    #from nltk.corpus import stopwords

    # * Machine Learning Libraries :
    #import lightgbm as lgb
    #from sklearn.ensemble import RandomForestRegressor
    #import xgboost as xgb


####################################################################################
# A NE FAIRE QU'UNE FOIS  => Chargement des données + indexation + génération qrel #
####################################################################################

### (1) CHARGEMENT DES DONNEES 
def Load_data(helper_path: str) -> pd.DataFrame:
    """Documentation
    
    Parameters :
        - helper_path : the file path

    Output (if exists) :
        - df : My Dataframe cleaned and reindexed

    """
    
    # Data Load with pandas librairy
    df = pd.read_csv(helper_path)

    # Drop articles with no abstract
    df = df[df['abstract'] != '']

    # Reset my dataframe index
    df = df.reset_index(drop = True)
    
    # Returns my clean dataframe
    return df


def Load_Pickle(helper_path: str):
    """Documentation
    
    Parameters :
        - helper_path : the file path

    Output (if exists) :
        - pick_file : My pickle file

    """
    with open(helper_path, 'rb') as f1:
        
        # Load Pickle file
        pick_file = pickle.load(f1)
        
    return pick_file



# (!)(!)(!)(!) A COMPLETER : RECUPERER LE  DATAFRAME SUR LES ABSTRACTS DU GROUPE 6 (!)(!)(!)(!)
# (!) Il faut absoluments les colonnes 'art_title', 'art_url', 'art_id' et 'abstract' et le DataFrame doit s'appeller 'df' (!)

#Helper_path : str = '/content/drive/MyDrive/Projet Interpromo 2021/Notebooks/Data/abstract_V0.csv' # Chemin vers le DataFrame sur les abstracts du groupe 6 
#df : pd.DataFrame = Load_data(Helper_path)
#df = df.iloc[list_index]  # (!) A COMPLETER : liste_index = liste des numéros de lignes correspondant à des articles GAMME GESTION / INNOVANT (!)

# Connect the drive folder
drive.mount('/content/drive')

# My file path for the fonction
Helper_path_P = '/content/drive/MyDrive/request_word_weight' # Chemin vers le fichier 'request_word_weight', voir avec Flora pour l'accès

# My Pickle variable
List_topics = Load_Pickle(Helper_path_P)

# Get topics DataFrame
Indices = np.arange(1, \
                    len(List_topics) + 1)

Topics = pd.DataFrame(List_topics, \
                      columns = ['query'])

# Create new Column
Topics['qid'] = Indices


### (2) INDEXATION

# Déclaration de la variable JAVA_HOME
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
Command = "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64"
subprocess.call(Command, 
                shell = True)

# Initialization de JVM
if not pt.started():
  pt.init()

# Création du dossier pour le stockage des indexes
Command = "rm -rf ./pd_index" # (!)(!)(!)(!) CHEMIN A MODIFIER (doit être identique au chemin plus bas (*)) (!)(!)(!)(!)
subprocess.call(Command, 
                shell = True)
pd_indexer = pt.DFIndexer("./pd_index") # (!)(!)(!)(!) CHEMIN A MODIFIER (doit être identique au chemin plus bas (*)) (!)(!)(!)(!)

# Select columns to the docs
Docs_columns : list = ['art_title', 'art_url', 'abstract']

# My Docs DataFrame
Docs = df[Docs_columns].copy()

# Add Column to my DataFrame
Docs['docno'] = df['art_id'].astype(str)

# My reference index creation
Indexref = pd_indexer.index(Docs["abstract"], \
                            Docs["docno"], \
                            Docs["art_url"], \
                            Docs["art_title"])


# (3) ON GENERE LE DATAFRAME QRELS (CONTIENENT POUR CHAQUE REQUETE LA LISTE DES DOCS REELLEMENT PERTINENTS)
def Remove_querys(topics : pd.DataFrame, model) -> pd.DataFrame :
    """Documentation
        
    Parameters :
        -topics: data frame of queries in french
        -model: learning to rank model

    Output (if exists) :
        -topics_: data frame of queries with at least one corresponding document
        
    """
    
    # Create my topics variable
    topics_ : pd.DataFrame = topics
    
    # Step 1 : Check matching document
    for i in range(len(topics)):

        # Get length of query data frame
        l = len(model.transform(topics["query"][i]))
    
        # Remove query if it doesn't match any document
        if l == 0:
            
            # Remove query if any match
            topics_ : pd.DataFrame = topics_.drop(index = i, \
                                                  axis = 0)
            
            # Optional Command : view the length of my topic
            # print(len(topics_))
    
    # Return my query DataFrame
    return topics_


def Get_qrels(topics : pd.DataFrame, model) -> pd.DataFrame :
    """Documentation
    
    Parameters :
        - topics: data frame of queries in french
        - model: learning to rank model

    Output (if exists) :
        - qrels_bis: new qrels data frame 
        
    """

    # My selected list
    columns : list = ["qid", "docno"]
    
    # Define new DataFrame
    qrels_bis : pd.DataFrame = pd.DataFrame(columns = columns)

    # Step 1 : Browse every queries 
    for i in topics.index:

        # Use my model for get qrels
        result = model.transform(Topics_final["query"][i])
      
        # Select queries randomly
        result_ = result.sample(n = 10)
        result_["qid"] = i + 1
      
        # Concatenate results 
        qrels_bis : pd.DataFrame = pd.concat([qrels_bis, \
                                            result_[["qid", "docno"]]])
    
    # Add label
    qrels_bis["label"] : pd.DataFrame = '1'

    # Return my new Qrels DataFrame
    return qrels_bis


# First Rankers : Make the candidate set of documents for each query
BM25 = pt.BatchRetrieve(Indexref,\
                        controls = {"wmodel": "BM25"})

# Fonction Application to get Topics and Qrels :
Topics_final = Remove_querys(Topics, BM25)
Qrels_bis = Get_qrels(Topics_final, BM25)

# Change the data type for learning in my Qrels DataFrame
Qrels_bis['qid'] = Qrels_bis['qid'].apply(str)
Qrels_bis['docno'] = Qrels_bis['docno'].apply(str)
Qrels_bis['label'] = Qrels_bis['label'].apply(str)

# Save on files
Qrels_bis.to_csv('./qrels_bis.csv', \
                 index=False) # (!)(!)(!)(!) CHEMIN A MODIFIER (doit être identique au chemin plus bas (**)) (!)(!)(!)(!)

Topics_final.to_csv('./topics_final_querys.csv', \
                    index=False) # (!)(!)(!)(!) CHEMIN A MODIFIER (doit être identique au chemin plus bas (***)) (!)(!)(!)(!)



#####################################################################################################################################
# A FAIRE UNE FOIS PAR SESSION (LORS DE LA CONNEXION AU SERVEUR ?) => Entrainement du modèle de ML + Chargement de fr_core_news_md  #
#####################################################################################################################################

# Load the french model for the lemmatization
Nlp = spacy.load('fr_core_news_md') # 
    
# Declaration of the variable JAVA_HOME
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
Command = "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64"
subprocess.call(Command, \
                shell=True)

# Initialization of JVM
if not pt.started():
  pt.init()

# We retrieve the index of abstracts  
Indexref = pt.IndexRef.of('/content/drive/MyDrive/Projet Interpromo 2021/Data/pd_index/data.properties') # (!)(!)(!)(!) CHEMIN A MODIFIER (doit être identique au chemin plus haut (*)) (!)(!)(!)(!)

# Data Load with pandas librairy
Qrels_bis = pd.read_csv('./qrels_bis.csv')
Topics = pd.read_csv('./topics_final_querys.csv')

# Create a random forest regressor
Rf = RandomForestRegressor(n_estimators = 100)

# Generate the LTR pipeline
BM25 = pt.BatchRetrieve(Indexref, \
                        controls = {"wmodel": "BM25"})

TF_IDF =  pt.BatchRetrieve(Indexref, \
                           controls = {"wmodel": "TF_IDF"})

DPH = pt.BatchRetrieve(Indexref, \
                       controls = {"wmodel": "DPH"})

# Using my Rankers to generate
Pipeline = BM25 >> (DPH ** TF_IDF) 
Pipeline.fit(Topics, \
             Qrels_bis)

# Using my regressor 
Rf_pipe = Pipeline >> pt.ltr.apply_learned_model(Rf)
Rf_pipe.fit(Topics, \
            Qrels_bis)

# Set the LightGBM parameters
Lmart_l_map = lgb.LGBMRanker(task = "train", \
                             min_data_in_leaf = 1,
                             min_sum_hessian_in_leaf=100, \
                             max_bin = 255, \
                             num_leaves = 7, \
                             objective = "lambdarank", \
                             metric = "map", \
                             ndcg_eval_at = [1, 3, 5, 10], \
                             learning_rate = .1, \
                             importance_type = "gain", \
                             num_iterations = 10)

# Generate the boosted LTR pipeline
Lmart_l_pipe_map = Rf_pipe >> pt.ltr.apply_learned_model(Lmart_l_map, \
                                                         form = "ltr")

# Execute the boosted LTR pipeline
Lmart_l_pipe_map.fit(Topics, \
                     Qrels_bis)

# (!)(!)(!)(!) lmart_l_pipe_map doit être une VARIABLE GLOBALE et être accessible dans votre fonction update_card_div (!)(!)(!)(!) 
# (!)(!)(!)(!) Nlp doit être une VARIABLE GLOBALE et être accessible dans votre fonction update_card_div (!)(!)(!)(!) 


################################################################################################################################################################################################################
###########################################################################  RETOUR A VOTRE FICHIER .py INITIAL  ###############################################################################################
################################################################################################################################################################################################################

@app.callback(
    Output('out_card', 'children'), \
    [Input('number', 'value')], \
    [Input('submit-button', 'n_clicks')], \
    [State('username', 'value')],
)


def Update_card_div(input_value2, clicks,input_value):
    
    filtered_df : pd.DataFrame = articles[:input_value2] # (!)(!)(!)(!) Remplacer 'articles' par le DataFrame sur les abstracts ? (!)(!)(!)(!)

    cards = affichage_card(range(len(filtered_df)))

    cards += [html.Div([dbc.Button("Return to top", \
                                   color="primary", \
                                   className="mr-1", \
                                   href='#', \
                                   style= button_style)])]
    
    
    if clicks is not None:
        
        # My initialized list
        element : list = []
        
        # My initialized list
        search_page : list = []
        
        # Check condition
        if input_value == '':
            
            return cards
        
        ### DEBUT PARTIE A SUPPRIMER
        for nb in range(len(filtered_df)):
            
            t = input_value
            if (t.lower() in filtered_df['art_title'][nb].lower()) 
            or (t.lower() in filtered_df['art_title'][nb].lower()) 
            or (t.lower() in filtered_df['art_title'][nb].lower()):
                
                element.append(nb)
                range(len(filtered_df))
        ### FIN PARTIE A SUPPRIMER

        ### A REMPLACER PAR : 
        element = get_relevant_art(input_value) # Function to obtain the list of documents ('art_id') relevant to the query
        ###

        if element != []:
            search_page.append(html.Div([html.H2("Article(s) trouvé(s) : " + str(len(element))),
                                         html.Br()]))
            search_page = affichage_card(element, search_page)
            search_page += [html.Div([dbc.Button("Return to top", color="primary", className="mr-1",href='#'),
                                     dbc.Button("Return", color="primary", className="mr-1",href='/')])]
        else:
            search_page.append(html.Div([html.H2("Aucun article trouvé")]))
        return search_page
    else:
        return cards


################################################
#  FONCTIONS A RAJOUTER (dans ce fichier .py)  #
################################################
def Get_relevant_art(query):
    """Documentation
    
    Parameters :
        - query: query entered by the user in the search bar 

    Output (if exists) :
        - result: list of documents sorted by relevance to the query
        
    """

    # Create my list of special characters
    Special_character : list = ["!","\"","#","$","%","&","\\","(", \
                                ")","*","+",",","-",".","/",":",";", \
                                "<","=",">","?","@","[","]","^","_", \
                                "{","|","}","~","«","»","’","•","…", \
                                "â","€","™","—","�","–","“","”"]
    
    # Query Preprocessing
    try : 

        # Lemmatization
        query = Lemmatize(query) 

        # Cleanning
        query = Cleanning(query, Special_character)
    
    except :
        
        pass

    # Lets look at the top results
    try : 

        result : list  = list(lmart_l_pipe_map.search(query)['docno']) # (!)(!)(!)(!) lmart_l_pipe_map doit être une variable globale pour ne pas ré-entrainer le modèle à chaque requête (!)(!)(!)(!) 
    
    except :
        
        result : list = []

    # List of documents sorted by relevance to the query
    return result


def Lemmatize(sentence: str) -> str :
    """Documentation
    
    Parameters :
        - sentence: all sentences not lemmatized 
            in the column ["art_content"]

    Output (if exists) :
        - sentence: all sentences lemmatize. 
            thanks to the library 'fr_core_web_sm'  

    """
    
    # Use my lemmatization model
    s = Nlp(sentence)
    
    # My initialized sentence
    lemmatized : str = ''
    
    # Step 1 : Lemmatization of each word
    for w in s:
        
        # Lemmatization of my sentence
        lemmatized += w.lemma_ + ' '
    
    # Return my lemmatized sentence
    return lemmatized



def Cleanning(item: str, special_character : list) -> str :
    """Documentation
    
    Parameters :
        - item: all articles without the removal of unnecessary words
        (for example : "stopWords") in the column ["art_content"]
        - special_character: a list of specials characters to 
        remove to the articles in the column ["art_content"]

    Output (if exists) :
        - result: all articles without unnecessary words 
        and characters 
        
    """

    # Convert text to lowercase
    item : str = item.lower()

    # Remove mail
    item : str = re.sub("(\w+)@(\w+).(\w+)","",item)

    # Remove twitter name
    item : str = re.sub("@(\w+)","",item)

    # Remove site ".com"
    item : str = re.sub("(\S+).com(\S+)","",item)
    item : str = re.sub("(\S+).com","",item)
        
    # Remove site ".fr"   
    item : str = re.sub("(\S+).fr(\S+)","",item)
    item : str = re.sub("(\S+).fr","",item)

    # Remove numbers
    # item : str = re.sub(r'\d+', '', item)

    # Remove hastags
    item : str = re.sub("#(\w+)","",item)

    # Remove years
    # item : str = re.sub("en (\d+)","",item)
    
    # Remove punctuation
    item : str = item.translate(str.maketrans("", "", string.punctuation))
    
    # Step 1 : Remove French accents
    for i in range(len(item)):
    
        # Get the article
        try:
        
            # Transform to 'utf-8'
            item = unicode(item , 'utf-8')
    
        except NameError: # unicode is a default on python 3 
        
            pass
        
        # Remove the accents
        item = unicodedata.normalize('NFD', str(item)) \
                            .encode('ascii', 'ignore') \
                            .decode("utf-8")
    
    # Step 2 : Remove Special character 
    for i in special_character:
        
        item = item.replace(i, "")
    
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('french'))
    tokens = word_tokenize(item)
    result = [i for i in tokens if not i in stop_words]
    result = " ".join(result)
    
    # Remove whitespaces
    result = result.strip()
    
    # Return my cleaned article
    return result
