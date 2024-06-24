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

        cursor = conn.cursor()
        cursor.execute('SELECT utilisateur_nom, utilisateur_prenom FROM Utilisateur')
        rows = cursor.fetchall()
        for row in rows:
            st.write(f"{row[0]} {row[1]}")

except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")

finally:
    # Close the cursor and connection
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print('MySQL connection closed')


df_ref = pd.read_csv("data/EFREI - LIPSTIP - 50k elements EPO.csv", nrows=1000)


df_ref = df_ref[df_ref['IPC'] != '[]']

import re
from bs4 import BeautifulSoup


def extract_first_letters(ipc_str):
    # Convertir la chaîne de caractères en liste
    ipc_list = eval(ipc_str)
    # Extraire la première lettre de chaque élément et prendre les distinctes
    first_letters = list({item[0] for item in ipc_list})
    return first_letters

# Appliquer la fonction à la colonne 'IPC'
df_ref['IPC level0'] = df_ref['IPC'].apply(extract_first_letters)



unique_elements = set()

# Parcourir chaque liste dans la colonne et ajouter les éléments à l'ensemble
for sublist in df_ref['IPC level0']:
    for item in sublist:
        unique_elements.add(item)

# Compter le nombre d'éléments distincts
unique_elements = sorted(list(unique_elements))
element_to_index = {element: idx for idx, element in enumerate(unique_elements)}


def replace_with_binary_list(sublist, element_to_index, num_unique_elements):
    binary_list = [0.0] * num_unique_elements
    for item in sublist:
        if item in element_to_index:
            binary_list[element_to_index[item]] = 1.0
    return binary_list

# Appliquer la fonction à chaque sous-liste dans df['IPC level0']
num_unique_elements = len(unique_elements)
df_ref['IPC level0'] = df_ref['IPC level0'].apply(lambda sublist: replace_with_binary_list(sublist, element_to_index, num_unique_elements))


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=8
)

model.load_state_dict(torch.load("models/mlt_label0"))
model.eval()

# Exemple de données

sampled_df = df_ref.sample(n=100, random_state=1)

X = sampled_df['description'].tolist()
y = np.array(sampled_df['IPC level0'].tolist())  # Labels correspondants

# Diviser les données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Définir le vectorizer et le modèle
vectorizer = TfidfVectorizer()
model = OneVsRestClassifier(LogisticRegression())

# Créer un pipeline
pipeline = make_pipeline(vectorizer, model)

# Entraîner le modèle
pipeline.fit(X, y)

import lime
import lime.lime_text

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Créer un explainer LIME pour le texte
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)





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
        # Obtenir les probabilités de prédiction pour cet exemple
        proba = pipeline.predict_proba([text_instance])
        # Identifier les deux classes avec les probabilités les plus élevées
        top_labels = np.argsort(proba[0])[::-1][:2]
        st.write(top_labels)
        # Obtenir une explication pour cet exemple
        exp = explainer.explain_instance(text_instance, pipeline.predict_proba, num_features=6, labels=top_labels)
        # Afficher l'explication pour les deux classes les plus représentées
        for label in top_labels:
            tempo_html = exp.as_html(labels=(label,))
            components.html(tempo_html, scrolling=True)


elif input_type == "Copier la description du brevet":
    st.title("Veuillez copier la description dans le chat")
    #initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # affiche de l'historique des message de la session
    for messages in st.session_state.messages:
        with st.chat_message(messages["role"]):
            if messages["role"] == "utilisateur":
                st.markdown(messages["content"])
            else:
                components.html(messages["content"], scrolling=True)
    
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
        proba = pipeline.predict_proba([text_instance])
        # Identifier les deux classes avec les probabilités les plus élevées
        top_labels = np.argsort(proba[0])[::-1][:2]
        exp = explainer.explain_instance(text_instance, pipeline.predict_proba, num_features=6, labels=top_labels)
        # Afficher l'explication pour les deux classes les plus représentées
        #affichage de la reponse dans la chat
        with st.chat_message("Identifieur"):
            for label in top_labels:
                components.html(exp.as_html(text=True, labels=(label,)), scrolling=True)
        #ajout du message à l'historique
        print(top_labels)
        for label in top_labels:
            tempo_html = exp.as_html(text=True, labels=(label,))
            st.session_state.messages.append({"role": "Identifieur", "content": tempo_html})



