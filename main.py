import streamlit as st
import mysql.connector
import pandas as pd
from io import StringIO

import torch

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import re

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
import time as ti



DB_CONFIG = st.secrets["mysql"]


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


input_type = st.radio("Choisissez la manière de d'entre la description", ["Uploader un fichier", "Copier la description du brevet"], index=None)


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
        temp_list.append((label_predict[i]["label"] , dict([("mot_cle",(sorted(tempo_explainer, key=(lambda x: x[1]), reverse=True))[:2]), ("score", label_predict[i]["score"])])))
        cls_explainer.visualize("distilbert_viz.html")
        file = open(r"./distilbert_viz.html","r")
        content_file += file.read()
        print(content_file)
        content_file += "<br>"
        print(content_file)
        file.close()
    #temp_list = [(label_predict[i]["label"] , dict([("mot_cle",(sorted(cls_explainer(texte_input, class_name=label_predict[i]["label"]), key=(lambda x: x[1]), reverse=True))[:2]), ("score", label_predict[i]["score"])])) for i in range(len(label_predict))]
    #content_file = content_file.replace(label_predict[len(label_predict) - 1]["label"][6:], dictionnaire_convertion_label_2[label_predict[len(label_predict) - 1]["label"]])
    key_word = dict(temp_list)
    file = open(r"./distilbert_viz.html","w")
    file.write("")
    file.close()
    for i in range(len(label_predict)):
        content_file = content_file.replace(label_predict[i]["label"], dictionnaire_convertion_label_2[label_predict[i]["label"]])
        content_file = content_file.replace(label_predict[i]["label"][6:], dictionnaire_convertion_label_2[label_predict[i]["label"]])
    return key_word, content_file


def get_score(dic_label, label):
    return dic_label[label]["score"]

def afficheur_resultat(resulat_analyse):
    tempo_list_string = [convertit_label(i) + f"score : {get_score(dic_label=resulat_analyse, label=i) * 100 :.2f} %" + "<br>Mots clés :<ul> " + ("".join([f"<li>{j[0]} (correlation : {j[1]:.2f})</li>"  for j in resulat_analyse[i]["mot_cle"]])) + "</ul>" for i in resulat_analyse ] 
    res = "<ol>" + ("".join(f"<li>{e}</li>" for e in tempo_list_string)) + "</ol>"
    return res


if input_type == "Uploader un fichier":
    file_uploaded = st.file_uploader("Mettez votre brevet", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if file_uploaded is not None:
        df_user = pd.read_csv(file_uploaded, sep=";")
        df_user['description'] = df_user['description'].apply(remove_html_tags_func_regex)
        df_user['description'] = df_user['description'].apply(remove_url_func)
        df_user['description'] = df_user['description'].apply(remove_extra_whitespaces_func)
        df_user['description'] = df_user['description'].apply(replace_fig_with_img)
        st.write(df_user)
        X_user = df_user['description'].tolist()
        text_instance = X_user[0][:2000]
        analysis, file_data = prediction_analyse(text_instance)
        st.html(afficheur_resultat(analysis))
        st.download_button(label="rapport", data=file_data, file_name=f"rapport_{ti.time()}.html")
        
        

if input_type == "Copier la description du brevet":
    st.title("Veuillez copier la description dans le chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] != "Identifieur":
                st.markdown(message["content"])
            else:
                st.html(message["content"][0])
                st.download_button(label="Rapport", data=message["content"][1], file_name=f"rapport_{ti.time()}.html")

    if prompt := st.chat_input("Copiez la description"):
        with st.chat_message("utilisateur"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "utilisateur", "content": prompt})

        if "description" not in st.session_state:
            st.session_state.description = ""

        if st.session_state.description == "":
            with st.spinner("Prétraitement de la description..."):
    
                st.session_state.description = remove_html_tags_func_regex(prompt)
                st.session_state.description = remove_url_func(st.session_state.description)
                st.session_state.description = remove_extra_whitespaces_func(st.session_state.description)
                st.session_state.description = replace_fig_with_img(st.session_state.description)
                text_instance = st.session_state.description

                analysis, file_data = prediction_analyse(text_instance)

                tempo_string = afficheur_resultat(analysis)
                with st.chat_message("Identifieur"):
                    st.html(tempo_string)
                    st.download_button(label="Rapport", data=file_data, file_name=f"rapport_{ti.time()}.html")

                st.session_state.messages.append({"role": "Identifieur", "content": [tempo_string, file_data]})
        
        # Formulaire pour saisir le titre du brevet
        with st.form(key='brevet_title_form'):
            brevet_title = st.text_input("Entrez le titre du brevet :")
            submit_button = st.form_submit_button(label='Sauvegarder les données')

            if submit_button:
                st.write(f"Soumission du formulaire confirmée. Titre du brevet : {brevet_title}")  
                if brevet_title:
                    st.write(f"Titre du brevet confirmé : {brevet_title}")  # Débogage
                    with st.spinner('Stockage des informations...'):
                        try:
                            # Connexion à la base de données et insertion du titre du brevet
                            conn = mysql.connector.connect(
                                host=DB_CONFIG['host'],
                                port=DB_CONFIG['port'],
                                database=DB_CONFIG['database'],
                                user=DB_CONFIG['user'],
                                password=DB_CONFIG['password']
                            )
                            if conn.is_connected():
                                cursor = conn.cursor()
                                
                                cursor.execute("INSERT INTO Brevet (brevet_titre) VALUES (%s)", (brevet_title,))
                                brevet_id = cursor.lastrowid
                                conn.commit()
                                
                                # Insertion des labels CPC et mots-clés associés
                                for label, info in analysis.items():
                                    label_cpc = label[6:]
                                    score = info["score"]
                                    for mot, correlation in info["mot_cle"]:
                                        cursor.execute("SELECT * FROM CPC WHERE label_cpc = %s", (label_cpc,))
                                        result = cursor.fetchone()
                                        if not result:
                                            cursor.execute("INSERT INTO CPC (label_cpc) VALUES (%s)", (label_cpc,))
                                            conn.commit()
                                        
                                        cursor.execute("SELECT id_mot FROM Mot WHERE mot_cle = %s", (mot,))
                                        result = cursor.fetchone()
                                        if result:
                                            id_mot = result[0]
                                        else:
                                            cursor.execute("INSERT INTO Mot (mot_cle) VALUES (%s)", (mot,))
                                            id_mot = cursor.lastrowid
                                            conn.commit()
                                        
                                        cursor.execute("INSERT INTO Labeliser_cpc (id_brevet, label_cpc, id_mot) VALUES (%s, %s, %s)", (brevet_id, label_cpc, id_mot))
                                        conn.commit()

                                st.success("Les informations ont été stockées avec succès.")

                            else:
                                st.error("Connexion à la base de données échouée.")

                        except mysql.connector.Error as e:
                            st.error(f"Erreur lors de la connexion à la base de données : {e}")

                        finally:
                            if 'conn' in locals() and conn.is_connected():
                                cursor.close()
                                conn.close()
                                st.write("Connexion à la base de données fermée.")

                else:
                    st.warning("Veuillez saisir un titre pour le brevet.")







