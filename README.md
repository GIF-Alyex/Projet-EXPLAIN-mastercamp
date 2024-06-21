# Projet-EXPLAIN-mastercamp

## Streamlit

Pour run il faut entrer la commande :    
`streamlit run main.py`

Il faudrait aussi regarder le fichier *requirements.txt*

Dans le *.gitignore* il y a un fichier *secrets.toml* stockant des variables environnementales.   
Par exemple pour la base de données MySQL vous aurez à mettre :

```toml
[mysql]   
host = "localhost"   
port = 3306   
database = "explain_db"   
user = "XXX"   
password = "XXX"
``` 

## Connexion MySQL

Il faut installer MySQL Connector:   
`pip install mysql-connector-python`

Il faut installer SQLAlchemy:   
`pip install SQLAlchemy`

