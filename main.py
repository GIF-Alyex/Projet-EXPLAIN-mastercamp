import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import mysql.connector

import torch

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import re
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
import streamlit.components.v1 as components

from transformers_interpret import MultiLabelClassificationExplainer, SequenceClassificationExplainer
from transformers import TextClassificationPipeline, pipeline, DistilBertForSequenceClassification, DistilBertTokenizer
import time as ti

import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate    
from streamlit_navigation_bar import st_navbar
import pages as pg
import os
from streamlit_option_menu import option_menu
import yaml

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

# Configuration du nom et de l'îcone de la page 
st.set_page_config(page_title="EXPLAIN", page_icon="🔲", layout="wide", initial_sidebar_state="auto")

# Configuration de la première barre de tâche 
pages = ["Documentation", "Tutorials", "Examples", "Community", "GitHub","                                     ","                   ","Settings","Features"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "./assets/logo.svg")
settings_image_path = os.path.join(parent_dir, "./assets/settings.svg")
urls = {"GitHub": "https://github.com/GIF-Alyex/Projet-EXPLAIN-mastercamp",}
styles = {
    "nav": {
        "background-color": "#1D2A35",
        "justify-content": "space-between",
        "height": "75px",
    },
    "img": {
        "height":"350px",
        "margin-top": "14px",
        "margin-left": "-10px",
        "padding-left": "-200px",
    },
    "span": {
        "color": "#DDDDDD",
        "padding": "30px",
        "font-family": "Source Sans Pro Topnav, sans-serif",
        "font-size": "20px",
    },
    "active": {
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "30px",
        "font-family": "Source Sans Pro Topnav, sans-serif",
        "font-size": "20px",
    },
    "hover": {
        "background-color": "#04AA6D",
    },
    
}
options = {"show_sidebar": False, "show_menu": False,}

page = st_navbar(
    pages,
    logo_path=logo_path,
    urls=urls,
    styles=styles,
    options=options,
)

# Configuration de la deuxième barre de tâche
st.markdown(
    """
    <style>
    .navbar {
        overflow-x: auto;
        background-color: #2a2d30;
        width: 100%;
        position: fixed;
        top: 75px;
        left: 0;
        z-index: 100;
        white-space: nowrap;
    }

    .navbar::-webkit-scrollbar {
        display: none;
    }

    .navbar a {
        display: inline-block;
        width: auto;
        color: inherit;
        padding: 5px 15px 5px 15px;
        text-decoration: none;
        font-size: 23px;
        line-height: 1.5;
        box-sizing: inherit;
        font-family: 'Source Sans Pro Topnav', sans-serif;
        border-radius: 15px; 
    }

    .navbar a:hover {
        background-color: black;
        color: #f2f2f2;
    }

    .navbar a.selected {
        background-color: black;
        color: #f2f2f2;
        border-radius: 15px;
    }

    .navbar img {
        height: 40px;
        margin-right: 16px;
    }
    
    </style>
    <div class="navbar">
        <a href="#">Fichier .txt</a>
        <a href="#">Fichier .md</a>
        <a href="#">Fichier .doc</a>
        <a href="#">Fichier .odt</a>
        <a href="#">Fichier .rtf</a>
        <a href="#">Fichier .tex</a>
        <a href="#">Fichier .rst</a>
        <a href="#">Fichier .todo</a>
        <a href="#">Fichier .org</a>
        <a href="#">Fichier .adoc</a>
        <a href="#">Fichier .rdoc</a>
        <a href="#">Fichier .ppt</a>
        <a href="#">Fichier .odp</a>
    </div>
    """
, unsafe_allow_html=True)

# Mise en place d'une image en fond de l'application
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
width: 100%;
height: 100%;
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
height: 100vh;
background-image: url("https://raw.githubusercontent.com/Theooo91/image/c5faa5305508abcdbdd105bf56f3f1d17f472d5a/magicpattern-starry-night-1719566881352.png");
}


</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Charger le fichier css
with open("./style/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# En tête de l'application
st.markdown("<br><br><br><br>", unsafe_allow_html=True)
st.markdown('<h1 class="custom-text">Simplify your patent</h1>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h1 class="custom-text2">Explainable Patent Learning for Artificial Intelligence</h1>', unsafe_allow_html=True)


st.markdown("<br><br><br><br>", unsafe_allow_html=True)

dictionnaire_convertion_label = {0: 'A01', 1: 'A21', 2: 'A22', 3: 'A23', 4: 'A24', 5: 'A41', 6: 'A42', 7: 'A43', 8: 'A44', 9: 'A45', 10: 'A46', 11: 'A47', 12: 'A61', 13: 'A62', 14: 'A63', 15: 'B01', 16: 'B02', 17: 'B03', 18: 'B04', 19: 'B05', 20: 'B06', 21: 'B07', 22: 'B08', 23: 'B09', 24: 'B21', 25: 'B22', 26: 'B23', 27: 'B24', 28: 'B25', 29: 'B26', 30: 'B27', 31: 'B28', 32: 'B29', 33: 'B30', 34: 'B31', 35: 'B32', 36: 'B33', 37: 'B41', 38: 'B42', 39: 'B43', 40: 'B44', 41: 'B60', 42: 'B61', 43: 'B62', 44: 'B63', 45: 'B64', 46: 'B65', 47: 'B66', 48: 'B67', 49: 'B68', 50: 'B81', 51: 'B82', 52: 'C01', 53: 'C02', 54: 'C03', 55: 'C04', 56: 'C05', 57: 'C06', 58: 'C07', 59: 'C08', 60: 'C09', 61: 'C10', 62: 'C11', 63: 'C12', 64: 'C13', 65: 'C14', 66: 'C21', 67: 'C22', 68: 'C23', 69: 'C25', 70: 'C30', 71: 'C40', 72: 'D01', 73: 'D02', 74: 'D03', 75: 'D04', 76: 'D05', 77: 'D06', 78: 'D07', 79: 'D10', 80: 'D21', 81: 'E01', 82: 'E02', 83: 'E03', 84: 'E04', 85: 'E05', 86: 'E06', 87: 'E21', 88: 'F01', 89: 'F02', 90: 'F03', 91: 'F04', 92: 'F05', 93: 'F15', 94: 'F16', 95: 'F17', 96: 'F21', 97: 'F22', 98: 'F23', 99: 'F24', 100: 'F25', 101: 'F26', 102: 'F27', 103: 'F28', 104: 'F41', 105: 'F42', 106: 'G01', 107: 'G02', 108: 'G03', 109: 'G04', 110: 'G05', 111: 'G06', 112: 'G07', 113: 'G08', 114: 'G09', 115: 'G10', 116: 'G11', 117: 'G16', 118: 'G21', 119: 'H01', 120: 'H02', 121: 'H03', 122: 'H04', 123: 'H05', 124: 'H10', 125: 'Y02', 126: 'Y04', 127: 'Y10'}
dictionnaire_convertion_label_2 = dict([(f"LABEL_{i[0]}", i[1]) for i in dictionnaire_convertion_label.items()])

def convertit_label(nom_label):
    return "<h5>Label : " + dictionnaire_convertion_label[int(nom_label[6:])] + "</h5>"


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







def remove_html_tags_func_regex(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = re.sub(r'-->', '', text)
    return text

def remove_url_func(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_extra_whitespaces_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def replace_fig_with_img(text):
     return re.sub(r'(fig)(ure)?(s)?(.)? \d+(-\d+)?(\sand\s\d+(-\d+)?)?', '<img>', text,  flags=re.I)

def remove_appos(text):
    return re.sub(r"^'|'$", "", text)

def dict_get_keys(dictionnaire, key):
    return dictionnaire[key]


def apply_keyword_style(key_tuple, word):
    res = word
    if res.lower() == key_tuple[0].lower():
        #print(res)
        res = f"<strong style='font-size:{round(100 + 100 * key_tuple[1])}%; background-color:#{hex(round((1 - key_tuple[1]) * 255))[2:]}FF{hex(round((1 - key_tuple[1]) * 255))[2:]};'>{res}</strong>"
        #print(res)
    #print(f"RESULT : {res}")
    return res
    
def highlight_key_word(tuple_key, texte_input):
    return " ".join(list(map(lambda x: apply_keyword_style(tuple_key, x), texte_input.split()))) 

dictionnaire_categorie = {"A":"Nécessités humaines", "B":"Operation et transport","C":"Chimie et Métalurgie", "D":"Tissue et papier", "E": "Construction", "F":"Mechanique, électricité, chauffage, arme et explosion", "G":"Physique", "H":"Électricité","Y":"Nouvelle technologie, nouvelle technologie concerant plusieurs domaines"}

def One_label_report(analyze_result, analyzed_texte):
    #print(list(map(lambda x: apply_keyword_style("pigs", x), analyzed_texte.split())))
    tempo = dictionnaire_convertion_label[int(analyze_result[0][6:])]
    res = f"<h2>{tempo}</h2>"
    res += f"<h3> Catégorie {tempo[0]} : {dictionnaire_categorie[tempo[0]]}</h3>"
    tempo = dict_get_keys(analyze_result[1], "score")
    res += f"<h3>Score : {(tempo*100):.2f} % </h3>"
    tempo = dict_get_keys(analyze_result[1], "mot_cle")
    res += "<br>".join([f"{i[0]} (contribution : {float(i[1]):.2f})" for i in tempo])
    res += "<br>"
    return res

def add_key_word(list_key_word):
    return list(filter(lambda x: x[1] + 0.02 >= list_key_word[0][1], list_key_word))


liste_ponctuation = [",","?",";","!","."]

def recreate_sentence(sentence):
    res = sentence[0]
    for i in sentence[1:]:
        if i in liste_ponctuation:
            res += i
        else:
            res +=  (" " + i)
    return res


def text_highlight(list_word):
    maximum = max(list_word[1:-1], key=lambda x: x[1])[1]
    temp_list = [f"<strong style='font-size:{round(100 + 100 * i[1])}%; background-color:#{hex(round((1 - i[1]) * 255))[2:]}FF{hex(round((1 - i[1]) * 255))[2:]};'>{i[0]}</strong>" if i[1] + 0.02 >= maximum else str(i[0])  for i in list_word[1:-1]]
    res = recreate_sentence(temp_list)
    return res


def remove_hachtag(heatlist):
    res = list()
    for i in heatlist:
        if "##" == i[0][:2]:
            res[-1] = (res[-1][0] + i[0][2:], (res[-1][1] + i[1])/2)
        else:
            res.append(i)
    return res

def prediction_analyse(texte_input):
    if len(texte_input) > 512:
        texte_input = texte_input[:512]
    label_predict = identificateur_label(texte_input, top_k=2)
    cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer
    )
    temp_list = []
    content_file = ""
    for i in range(len(label_predict)):
        tempo_explainer = cls_explainer(texte_input, class_name=label_predict[i]["label"])
        tempo_explainer = remove_hachtag(tempo_explainer)
        #print(sorted(tempo_explainer, reverse=True, key=lambda x: x[1]))
        temp_list.append((label_predict[i]["label"] , dict([("mot_cle", add_key_word(sorted(tempo_explainer[1:-1], key=(lambda x: x[1]), reverse=True))), ("score", label_predict[i]["score"])])))
        content_file += One_label_report(temp_list[-1], texte_input)
        content_file += ("<p>" + text_highlight(tempo_explainer) +"</p><br>")
    key_word = dict(temp_list)
    return key_word, content_file


def get_score(dic_label, label):
    return dic_label[label]["score"]

def afficheur_resultat(resulat_analyse):
    tempo_list_string = [convertit_label(i) + f"score : {get_score(dic_label=resulat_analyse, label=i) * 100 :.2f} %" + "<br>Mots clés :<ul> " + ("".join([f"<li>{j[0]} (correlation : {j[1]:.2f})</li>"  for j in resulat_analyse[i]["mot_cle"]])) + "</ul>" for i in resulat_analyse ] 
    res = "<ol>" + ("".join(f"<li>{e}</li>" for e in tempo_list_string)) + "</ol>"
    return res



if True:
    #initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # affiche de l'historique des message de la session
    for messages in st.session_state.messages:
        with st.chat_message(messages["role"]):
            if messages["role"] != "Identifieur":
                st.markdown(messages["content"])
            else:
                st.html(messages["content"][0])
                st.download_button(label="rapport", data=messages["content"][1], file_name=f"rapport_{ti.time()}.html")
    
    # widget accpetant l'input de l'utilisateur
    if prompt := st.chat_input("Copiez la description"):
        #affichage du message
        with st.chat_message("utilisateur"):
            st.markdown(prompt)
        #ajout du nouveau message 
        st.session_state.messages.append({"role": "utilisateur", "content": prompt})
        prompt = remove_html_tags_func_regex(prompt)
        prompt = remove_url_func(prompt)
        prompt = remove_extra_whitespaces_func(prompt)
        prompt = replace_fig_with_img(prompt)
        text_instance = prompt
        # Obtenir les probabilités de prédiction pour cet exemple
        analysis, file_data = prediction_analyse(text_instance)

        # Afficher l'explication pour les deux classes les plus représentées
        #affichage de la reponse dans la chat
        tempo_string = afficheur_resultat(analysis)
        with st.chat_message("Identifieur"):
            st.html(tempo_string)
            st.download_button(label="rapport", data=file_data, file_name=f"rapport_{ti.time()}.html")
        #ajout du message à l'historique
        st.session_state.messages.append({"role": "Identifieur", "content": [tempo_string, file_data]})

st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
