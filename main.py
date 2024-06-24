import streamlit as st
import mysql.connector
import pandas as pd
from io import StringIO

import torch

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertTokenizer
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

def extract_first_letters2(cpc_str):
    # Convertir la chaîne de caractères en liste
    ipc_list = eval(cpc_str)
    # Extraire la première lettre de chaque élément et prendre les distinctes
    first_letters = list({item[:3] for item in ipc_list})
    return first_letters

def extract_first_letters3(cpc_str):
    # Convertir la chaîne de caractères en liste
    ipc_list = eval(cpc_str)
    # Extraire la première lettre de chaque élément et prendre les distinctes
    first_letters = list({item for item in ipc_list})
    return first_letters

# Appliquer la fonction à la colonne 'IPC'
df_ref['CPC level0'] = df_ref['CPC'].apply(extract_first_letters)
df_ref['CPC level1'] = df_ref['CPC'].apply(extract_first_letters2)
df_ref['CPC level2'] = df_ref['CPC'].apply(extract_first_letters3)


unique_elements = set()

# Parcourir chaque liste dans la colonne et ajouter les éléments à l'ensemble
for sublist in df_ref['CPC level0']:
    for item in sublist:
        unique_elements.add(item)

# Compter le nombre d'éléments distincts
unique_elements = sorted(list(unique_elements))
element_to_index = {element: idx for idx, element in enumerate(unique_elements)}

unique_elements_1 = set()

# Parcourir chaque liste dans la colonne et ajouter les éléments à l'ensemble
for sublist in df_ref['CPC level1']:
    for item in sublist:
        unique_elements_1.add(item)

# Compter le nombre d'éléments distincts
unique_elements_1 = sorted(list(unique_elements_1))
element_to_index_1 = {element: idx for idx, element in enumerate(unique_elements_1)}

def replace_with_binary_list(sublist, element_to_index_1, num_unique_elements_1):
    binary_list = [0.0] * num_unique_elements_1
    for item in sublist:
        if item in element_to_index_1:
            binary_list[element_to_index_1[item]] = 1.0
    return binary_list

# Appliquer la fonction à chaque sous-liste dans df['IPC level0']
num_unique_elements_1 = len(unique_elements_1)
df_ref['CPC level1'] = df_ref['CPC level1'].apply(lambda sublist: replace_with_binary_list(sublist, element_to_index_1, num_unique_elements_1))


print(len(element_to_index_1))

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    problem_type="multi_label_classification",
    num_labels = 128
)

model.load_state_dict(torch.load("models\distil_mlt_label_128_10000"))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)






import lime
#import lime.lime_text
from lime.lime_text import LimeTextExplainer






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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_proba(texts):
    # Tokenisation et conversion en tenseurs
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Conversion en probabilités (si nécessaire)
    probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
    return probabilities




def traitement(val_text):
    #chargement de la description à expliquer
    inputs = tokenizer(val_text,padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    #selction des labels avec une probabilité supérieur 0.5
    # Désactivez les gradients pour économiser de la mémoire
    with torch.no_grad():
        outputs = model(**inputs)

    # Sorties du modèle
    logits = outputs.logits

    # Appliquez une fonction sigmoïde pour obtenir les probabilités
    probabilities = torch.sigmoid(logits)

    # Convertir les probabilités en étiquettes avec un seuil (par exemple, 0.5)
    threshold = 0.5
    predictions = (probabilities > threshold).int()

    # Initialisation l'explainer LIme pour le texte
    explainer = LimeTextExplainer(class_names=list(element_to_index_1.keys()))

    # Trouver les trois premières phrases

    sentences = re.split(r'(?<=[.:;])\s', val_text)
    summary = ' '.join(sentences[:1])

    # Explication de la prédiction
    explanation = explainer.explain_instance(
    summary,
    predict_proba,
    num_features=10,  # Nombre de mots à afficher dans l'explication
    labels=predictions[0].nonzero().squeeze().tolist()  # Labels à expliquer
    )
    return explanation.as_html(text=True)


if input_type == "Uploader un fichier":
    file_uploaded = st.file_uploader("Mettez votre brevet", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if file_uploaded is not None:
        df_user = pd.read_csv(file_uploaded, sep=";")
        df_user['description'] = df_user['description'].apply(remove_html_tags_func)
        df_user['description'] = df_user['description'].apply(remove_url_func)
        df_user['description'] = df_user['description'].apply(remove_extra_whitespaces_func)
        df_user['description'] = df_user['description'].apply(replace_fig_with_img)
        st.write(df_user['description'].to_list())
        tempo_html = traitement(df_user['description'][0])
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
        tempo_html = traitement(prompt)
        # Afficher l'explication pour les deux classes les plus représentées
        #affichage de la reponse dans la chat
        with st.chat_message("Identifieur"):
            components.html(tempo_html, scrolling=True)
        #ajout du message à l'historique
        st.session_state.messages.append({"role": "Identifieur", "content": tempo_html})



