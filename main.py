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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Définir le vectorizer et le modèle
vectorizer = TfidfVectorizer()
model = OneVsRestClassifier(LogisticRegression())

# Créer un pipeline
pipeline = make_pipeline(vectorizer, model)

# Entraîner le modèle
pipeline.fit(X_train, y_train)




class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']



file_uploaded = st.file_uploader("Mettez votre brevet", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

if file_uploaded is not None:
        df_user = pd.read_csv(file_uploaded, sep=";")
        st.write(df_user)
        X_user = df_user['description'].tolist()
        text_instance = X_user[0][:2000]
        # Obtenir les probabilités de prédiction pour cet exemple
        proba = pipeline.predict_proba([text_instance])
        # Identifier les deux classes avec les probabilités les plus élevées
        top_labels = np.argsort(proba[0])[::-1][:2]
        print(top_labels)
        st.write(top_labels)
        
        

