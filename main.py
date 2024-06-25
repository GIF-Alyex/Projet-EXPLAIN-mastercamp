import streamlit as st
import mysql.connector
import pandas as pd
from io import StringIO

import torch

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import re
from bs4 import BeautifulSoup

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
import streamlit.components.v1 as components

from transformers_interpret import MultiLabelClassificationExplainer, SequenceClassificationExplainer
from transformers import TextClassificationPipeline, pipeline, DistilBertForSequenceClassification, DistilBertTokenizer


st.write("Hello world !")


DB_CONFIG = st.secrets["mysql"]

try:
    # Etablir la connexion à la base de données
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    if conn.is_connected():
        print('Connected to MySQL database')

except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")

finally:
    # Close the cursor and connection
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print('MySQL connection closed')


import re
from bs4 import BeautifulSoup



# Chargement du modèle
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    problem_type="multi_label_classification",
    num_labels = 128
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True, trust_remote_code=True, use_fast=True)

model.load_state_dict(torch.load("models\distil_mlt_label_128_10000"))
model.eval()

# Créer un pipeline
identificateur_label = pipeline("text-classification", model=model, tokenizer=tokenizer)





input_type = st.radio("Choisissez la manière de d'entre la description", ["Uploader un fichier", "Copier la description du brevet"], index=None)



def remove_html_tags_func(text):
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup.find_all(True):
        tag.name = "p"
    text = soup.get_text(separator=' ')
    return re.sub(r'\s\s+', ' ', text)

def remove_url_func(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_extra_whitespaces_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def replace_fig_with_img(text):
     return re.sub(r'(fig)(ure)?(s)?(.)? \d+(-\d+)?(\sand\s\d+(-\d+)?)?', '<img>', text,  flags=re.I)

def remove_appos(text):
    return re.sub(r"^'|'$", "", text)



def prediction_analyse(texte_input):
    if len(texte_input) > 512:
        texte_input = texte_input[:512]
    label_predict = identificateur_label(texte_input, top_k=2)
    list_label = [label_predict[i]["label"] for i in range(len(label_predict))]
    cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer
    )
    temp_list = [(label ,(sorted(cls_explainer(texte_input, class_name=label), key=(lambda x: x[1]), reverse=True))[:2]) for label in list_label]
    key_word = dict(temp_list)
    return key_word


def afficheur_resultat(resulat_analyse):
    res = "\n".join([i + ", Mots clées : " + (", ".join([j[0] for j in resulat_analyse[i]])) for i in resulat_analyse])
    return res


if input_type == "Uploader un fichier":
    file_uploaded = st.file_uploader("Mettez votre brevet", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if file_uploaded is not None:
        df_user = pd.read_csv(file_uploaded, sep=";")
        df_user['description'] = df_user['description'].apply(remove_html_tags_func)
        df_user['description'] = df_user['description'].apply(remove_url_func)
        df_user['description'] = df_user['description'].apply(remove_extra_whitespaces_func)
        df_user['description'] = df_user['description'].apply(replace_fig_with_img)
        st.write(df_user)
        X_user = df_user['description'].tolist()
        text_instance = X_user[0][:2000]
        analysis = prediction_analyse(text_instance)
        st.write(afficheur_resultat(analysis))
        
        

elif input_type == "Copier la description du brevet":
    st.title("Veuillez copier la description dans le chat")
    #initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # affiche de l'historique des message de la session
    for messages in st.session_state.messages:
        with st.chat_message(messages["role"]):
            st.markdown(messages["content"])
    
    # widget accpetant l'input de l'utilisateur
    if prompt := st.chat_input("Copiez la description"):
        #affichage du message
        with st.chat_message("utilisateur"):
            st.markdown(prompt)
        #ajout du nouveau message 
        st.session_state.messages.append({"role": "utilisateur", "content": prompt})
        prompt = remove_html_tags_func(prompt)
        prompt = remove_url_func(prompt)
        prompt = remove_extra_whitespaces_func(prompt)
        prompt = replace_fig_with_img(prompt)
        text_instance = prompt
        # Obtenir les probabilités de prédiction pour cet exemple
        analysis = prediction_analyse(text_instance)

        # Afficher l'explication pour les deux classes les plus représentées
        #affichage de la reponse dans la chat
        tempo_string = afficheur_resultat(analysis)
        with st.chat_message("Identifieur"):
            st.markdown(tempo_string)
        #ajout du message à l'historique
        st.session_state.messages.append({"role": "Identifieur", "content": tempo_string})



