import streamlit as st
import mysql.connector
import pandas as pd
from io import StringIO

import torch

#from torch.utils.data.dataset import Dataset
#from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification

# streamlit run main.py 
# pip install mysql-connector-python

st.write("Hello world !")

DB_CONFIG = st.secrets["mysql"]

try:
    # Establish a connection to the database
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    if conn.is_connected():
        print('Connected to MySQL database')

        # Now you can execute SQL queries
        cursor = conn.cursor()

        # Execute the query
        cursor.execute('SELECT utilisateur_nom, utilisateur_prenom FROM Utilisateur')

        # Fetch all rows from the result set
        rows = cursor.fetchall()

        # Display the results using Streamlit
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



"""
import re
from bs4 import BeautifulSoup

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


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=8
)

model.load_state_dict(torch.load("models/mlt_label0"))
model.eval()


file_uploaded = st.file_uploader("Mettez votre brevet", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

if file_uploaded is not None:
        df = pd.read_xml(file_uploaded)
        st.write(df)
        df['description'] = df['description'].apply(remove_html_tags_func)
        df['description'] = df['description'].apply(remove_url_func)
        df['description'] = df['description'].apply(remove_extra_whitespaces_func)
        df['description'] = df['description'].apply(replace_fig_with_img)
        uploaded_texts = df['description'].tolist()
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        uploaded_encodings = tokenizer(uploaded_texts, padding="max_length", truncation=True, max_length=512)

"""