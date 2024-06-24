import streamlit as st
import pandas as pd
import numpy as np
import torch
from bs4 import BeautifulSoup
import re
  
from streamlit_navigation_bar import st_navbar
import os


# Configuration du nom et de l'îcone de la page 
st.set_page_config(page_title="EXPLAIN", page_icon="🔲", layout="wide", initial_sidebar_state="auto")

# Configuration de la première barre de tâche 
pages = ["Install", "User Guide", "API", "Examples", "Community", "GitHub"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
urls = {"GitHub": "https://github.com/gabrieltempass/streamlit-navigation-bar"}
styles = {
    "nav": {
        "background-color": "#1D2A35",
        "justify-content": "left",
        "height": "75px",
    },
    "img": {
        "padding-right": "30px",
    },
    "span": {
        "color": "#ddd",
        "padding": "30px",
        "font-family": "Source Sans Pro Topnav, sans-serif",
        "font-size":"20px",
    },
    "active": {
        "background-color": "#04AA6D",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "30px",
        "font-family": "Source Sans Pro Topnav, sans-serif",
        "font-size":"20px",
    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages,
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
        font-size: 20px;
        line-height: 1.5;
        box-sizing: inherit;
        font-family: 'Source Sans Pro Topnav', sans-serif;
        
    }

    .navbar a:hover {
        background-color: black;
        color: #f2f2f2;
    }

    .navbar img {
            height: 40px;
            margin-right: 16px;
        }
    
    </style>
    <div class="navbar">
        <a href="#">Exemple1</a>
        <a href="#">Exemple2</a>
        <a href="#">Exemple3</a>
        <a href="#">Exemple4</a>
        <a href="#">Exemple5</a>
    </div>
    """
, unsafe_allow_html=True)

# Application d'une image en fond de l'application
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
width: 100%;
height: 100%;
background-size: cover;
background-position: center center;
background-repeat: repeat;
background-image: url("data:image/svg+xml;utf8,%3Csvg viewBox=%220 0 2000 1400%22 xmlns=%22http:%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cdefs%3E%3Cfilter id=%22b%22 x=%22-200%25%22 y=%22-200%25%22 width=%22500%25%22 height=%22500%25%22%3E%3CfeGaussianBlur in=%22SourceGraphic%22 stdDeviation=%2220%22%2F%3E%3C%2Ffilter%3E%3C%2Fdefs%3E%3Cpath fill=%22%232a2d30%22 d=%22M0 0h2000v1400H0z%22%2F%3E%3Cellipse cx=%221949.971%22 cy=%225.354%22 rx=%222.634%22 ry=%222.311%22 fill=%22%23fff%22 opacity=%22.169%22%2F%3E%3Cellipse cx=%22413.132%22 cy=%2217.264%22 rx=%222.612%22 ry=%222.235%22 fill=%22%23fff%22 opacity=%22.432%22%2F%3E%3Cellipse cx=%221558.962%22 cy=%2237.419%22 rx=%222.737%22 ry=%222.652%22 fill=%22%23fff%22 opacity=%22.298%22%2F%3E%3Cellipse cx=%22500.211%22 cy=%2247.18%22 rx=%222.719%22 ry=%222.362%22 fill=%22%23fff%22 opacity=%22.236%22%2F%3E%3Cellipse cx=%221016.455%22 cy=%2259.334%22 rx=%222.761%22 ry=%222.659%22 fill=%22%23fff%22 opacity=%22.29%22%2F%3E%3Cellipse cx=%221084.156%22 cy=%2278.679%22 rx=%221.565%22 ry=%221.378%22 fill=%22%23fff%22 opacity=%22-.164%22%2F%3E%3Cellipse cx=%22447.786%22 cy=%2295.373%22 rx=%221.943%22 ry=%221.626%22 fill=%22%23fff%22 opacity=%22.838%22%2F%3E%3Cellipse cx=%22752.07%22 cy=%22101.553%22 rx=%222.056%22 ry=%221.971%22 fill=%22%23fff%22 opacity=%22.565%22%2F%3E%3Cellipse cx=%22359.279%22 cy=%22122.491%22 rx=%222.612%22 ry=%222.507%22 fill=%22%23fff%22 opacity=%22.389%22%2F%3E%3Cellipse cx=%22925.217%22 cy=%22136.334%22 rx=%221.392%22 ry=%221.364%22 fill=%22%23fff%22 opacity=%22.795%22%2F%3E%3Cellipse cx=%221800.437%22 cy=%22141.066%22 rx=%221.429%22 ry=%221.359%22 fill=%22%23fff%22 opacity=%22.44%22%2F%3E%3Cellipse cx=%22574.591%22 cy=%22156.565%22 rx=%222.304%22 ry=%222.068%22 fill=%22%23fff%22 opacity=%22.478%22%2F%3E%3Cellipse cx=%22939.025%22 cy=%22180.869%22 rx=%222.774%22 ry=%222.275%22 fill=%22%23fff%22 opacity=%22.477%22%2F%3E%3Cellipse cx=%22426.505%22 cy=%22193.413%22 rx=%221.702%22 ry=%221.676%22 fill=%22%23fff%22 opacity=%22-.028%22%2F%3E%3Cellipse cx=%221032.691%22 cy=%22208.565%22 rx=%222.617%22 ry=%222.578%22 fill=%22%23fff%22 opacity=%22.122%22%2F%3E%3Cellipse cx=%22469.222%22 cy=%22217.743%22 rx=%223.166%22 ry=%222.692%22 fill=%22%23fff%22 opacity=%22-.168%22%2F%3E%3Cellipse cx=%22416.71%22 cy=%22230.391%22 rx=%221.761%22 ry=%221.728%22 fill=%22%23fff%22 opacity=%22-.066%22%2F%3E%3Cellipse cx=%22393.203%22 cy=%22246.268%22 rx=%222.351%22 ry=%222.026%22 fill=%22%23fff%22 opacity=%22-.234%22%2F%3E%3Cellipse cx=%221504.373%22 cy=%22253.667%22 rx=%221.892%22 ry=%221.741%22 fill=%22%23fff%22 opacity=%22.824%22%2F%3E%3Cellipse cx=%221331.049%22 cy=%22274.089%22 rx=%222.21%22 ry=%221.892%22 fill=%22%23fff%22 opacity=%22.324%22%2F%3E%3Cellipse cx=%22547.213%22 cy=%22285.277%22 rx=%221.85%22 ry=%221.608%22 fill=%22%23fff%22 opacity=%22-.067%22%2F%3E%3Cellipse cx=%22193.45%22 cy=%22299.361%22 rx=%222.858%22 ry=%222.681%22 fill=%22%23fff%22 opacity=%22.276%22%2F%3E%3Cellipse cx=%22250.078%22 cy=%22315.616%22 rx=%221.895%22 ry=%221.512%22 fill=%22%23fff%22 opacity=%22.722%22%2F%3E%3Cellipse cx=%221389.926%22 cy=%22330.528%22 rx=%222.318%22 ry=%222.017%22 fill=%22%23fff%22 opacity=%22.267%22%2F%3E%3Cellipse cx=%22712.6%22 cy=%22337.913%22 rx=%222.936%22 ry=%222.675%22 fill=%22%23fff%22 opacity=%22-.245%22%2F%3E%3Cellipse cx=%22117.603%22 cy=%22360.284%22 rx=%221.601%22 ry=%221.488%22 fill=%22%23fff%22 opacity=%22.035%22%2F%3E%3Cellipse cx=%221139.287%22 cy=%22368.188%22 rx=%222.834%22 ry=%222.664%22 fill=%22%23fff%22 opacity=%22.688%22%2F%3E%3Cellipse cx=%221500.606%22 cy=%22380.637%22 rx=%221.878%22 ry=%221.646%22 fill=%22%23fff%22 opacity=%22.409%22%2F%3E%3Cellipse cx=%221693.487%22 cy=%22398.218%22 rx=%222.001%22 ry=%221.659%22 fill=%22%23fff%22 opacity=%22-.031%22%2F%3E%3Cellipse cx=%221453.113%22 cy=%22407.273%22 rx=%222.229%22 ry=%221.969%22 fill=%22%23fff%22 opacity=%22.219%22%2F%3E%3Cellipse cx=%221439.942%22 cy=%22420.266%22 rx=%221.422%22 ry=%221.418%22 fill=%22%23fff%22 opacity=%22-.081%22%2F%3E%3Cellipse cx=%22568.033%22 cy=%22441.452%22 rx=%222.055%22 ry=%222.005%22 fill=%22%23fff%22 opacity=%22.141%22%2F%3E%3Cellipse cx=%221145.914%22 cy=%22450.672%22 rx=%222.396%22 ry=%222.318%22 fill=%22%23fff%22 opacity=%22.244%22%2F%3E%3Cellipse cx=%22367.354%22 cy=%22464.525%22 rx=%221.788%22 ry=%221.482%22 fill=%22%23fff%22 opacity=%22.53%22%2F%3E%3Cellipse cx=%221711.925%22 cy=%22477.077%22 rx=%221.853%22 ry=%221.437%22 fill=%22%23fff%22 opacity=%22.905%22%2F%3E%3Cellipse cx=%22536.227%22 cy=%22490.671%22 rx=%222.258%22 ry=%221.83%22 fill=%22%23fff%22 opacity=%22-.121%22%2F%3E%3Cellipse cx=%22156.575%22 cy=%22513.422%22 rx=%221.951%22 ry=%221.712%22 fill=%22%23fff%22 opacity=%22.25%22%2F%3E%3Cellipse cx=%22413.887%22 cy=%22521.129%22 rx=%222.658%22 ry=%222.458%22 fill=%22%23fff%22 opacity=%22.674%22%2F%3E%3Cellipse cx=%221783.483%22 cy=%22545.471%22 rx=%221.755%22 ry=%221.396%22 fill=%22%23fff%22 opacity=%22.382%22%2F%3E%3Cellipse cx=%22874.045%22 cy=%22558.325%22 rx=%221.782%22 ry=%221.399%22 fill=%22%23fff%22 opacity=%22.44%22%2F%3E%3Cellipse cx=%221908.908%22 cy=%22569.716%22 rx=%222.04%22 ry=%221.92%22 fill=%22%23fff%22 opacity=%22-.223%22%2F%3E%3Cellipse cx=%221879.682%22 cy=%22584.786%22 rx=%222.423%22 ry=%222.017%22 fill=%22%23fff%22 opacity=%22.656%22%2F%3E%3Cellipse cx=%22455.126%22 cy=%22595.014%22 rx=%222.538%22 ry=%222.069%22 fill=%22%23fff%22 opacity=%22.004%22%2F%3E%3Cellipse cx=%221494.427%22 cy=%22603.179%22 rx=%223.045%22 ry=%222.684%22 fill=%22%23fff%22 opacity=%22.523%22%2F%3E%3Cellipse cx=%22247.854%22 cy=%22618.838%22 rx=%222.11%22 ry=%222.043%22 fill=%22%23fff%22 opacity=%22.253%22%2F%3E%3Cellipse cx=%221466.402%22 cy=%22637.132%22 rx=%221.706%22 ry=%221.454%22 fill=%22%23fff%22 opacity=%22.348%22%2F%3E%3Cellipse cx=%22219.085%22 cy=%22647.455%22 rx=%222.332%22 ry=%222.196%22 fill=%22%23fff%22 opacity=%22.302%22%2F%3E%3Cellipse cx=%221668.924%22 cy=%22665.108%22 rx=%221.776%22 ry=%221.675%22 fill=%22%23fff%22 opacity=%22.515%22%2F%3E%3Cellipse cx=%22821.523%22 cy=%22674.185%22 rx=%222.403%22 ry=%222.115%22 fill=%22%23fff%22 opacity=%22.34%22%2F%3E%3Cellipse cx=%22804.362%22 cy=%22692.156%22 rx=%222.895%22 ry=%222.529%22 fill=%22%23fff%22 opacity=%22.258%22%2F%3E%3Cellipse cx=%221931.827%22 cy=%22708.22%22 rx=%222.349%22 ry=%222.073%22 fill=%22%23fff%22 opacity=%22.704%22%2F%3E%3Cellipse cx=%22830.55%22 cy=%22719.637%22 rx=%221.999%22 ry=%221.756%22 fill=%22%23fff%22 opacity=%22-.009%22%2F%3E%3Cellipse cx=%22270.6%22 cy=%22730.818%22 rx=%222.391%22 ry=%222.255%22 fill=%22%23fff%22 opacity=%22.872%22%2F%3E%3Cellipse cx=%22124.862%22 cy=%22747.734%22 rx=%221.922%22 ry=%221.649%22 fill=%22%23fff%22 opacity=%22.651%22%2F%3E%3Cellipse cx=%221109.959%22 cy=%22760.639%22 rx=%222.26%22 ry=%222.073%22 fill=%22%23fff%22 opacity=%22-.065%22%2F%3E%3Cellipse cx=%221021.998%22 cy=%22781.086%22 rx=%222.369%22 ry=%222.079%22 fill=%22%23fff%22 opacity=%22.756%22%2F%3E%3Cellipse cx=%22643.333%22 cy=%22791.101%22 rx=%222.596%22 ry=%222.143%22 fill=%22%23fff%22 opacity=%22-.179%22%2F%3E%3Cellipse cx=%221544.631%22 cy=%22800.545%22 rx=%222.468%22 ry=%222.147%22 fill=%22%23fff%22 opacity=%22.532%22%2F%3E%3Cellipse cx=%22320.468%22 cy=%22813.338%22 rx=%222.689%22 ry=%222.551%22 fill=%22%23fff%22 opacity=%22.119%22%2F%3E%3Cellipse cx=%2214.549%22 cy=%22827.446%22 rx=%221.883%22 ry=%221.415%22 fill=%22%23fff%22 opacity=%22.228%22%2F%3E%3Cellipse cx=%221334.702%22 cy=%22848.186%22 rx=%222.153%22 ry=%221.891%22 fill=%22%23fff%22 opacity=%22.031%22%2F%3E%3Cellipse cx=%221506.187%22 cy=%22859.256%22 rx=%221.688%22 ry=%221.602%22 fill=%22%23fff%22 opacity=%22.303%22%2F%3E%3Cellipse cx=%22908.89%22 cy=%22880.217%22 rx=%221.888%22 ry=%221.449%22 fill=%22%23fff%22 opacity=%22-.041%22%2F%3E%3Cellipse cx=%22395.695%22 cy=%22892.813%22 rx=%222.91%22 ry=%222.586%22 fill=%22%23fff%22 opacity=%22.042%22%2F%3E%3Cellipse cx=%221659.932%22 cy=%22901.099%22 rx=%221.579%22 ry=%221.374%22 fill=%22%23fff%22 opacity=%22.52%22%2F%3E%3Cellipse cx=%221735.506%22 cy=%22922.779%22 rx=%221.851%22 ry=%221.626%22 fill=%22%23fff%22 opacity=%22.56%22%2F%3E%3Cellipse cx=%221537.646%22 cy=%22930.791%22 rx=%222.949%22 ry=%222.558%22 fill=%22%23fff%22 opacity=%22.824%22%2F%3E%3Cellipse cx=%221573.75%22 cy=%22949.955%22 rx=%221.874%22 ry=%221.568%22 fill=%22%23fff%22 opacity=%22.579%22%2F%3E%3Cellipse cx=%22895.853%22 cy=%22962.92%22 rx=%221.798%22 ry=%221.478%22 fill=%22%23fff%22 opacity=%22.577%22%2F%3E%3Cellipse cx=%221340.66%22 cy=%22977.507%22 rx=%222.663%22 ry=%222.269%22 fill=%22%23fff%22 opacity=%22.365%22%2F%3E%3Cellipse cx=%22652.328%22 cy=%22990.841%22 rx=%221.799%22 ry=%221.52%22 fill=%22%23fff%22 opacity=%22.273%22%2F%3E%3Cellipse cx=%22792.241%22 cy=%221003.076%22 rx=%221.554%22 ry=%221.409%22 fill=%22%23fff%22 opacity=%22.632%22%2F%3E%3Cellipse cx=%221889.087%22 cy=%221011.215%22 rx=%222.827%22 ry=%222.655%22 fill=%22%23fff%22 opacity=%22.391%22%2F%3E%3Cellipse cx=%221302.627%22 cy=%221025.05%22 rx=%221.815%22 ry=%221.551%22 fill=%22%23fff%22 opacity=%22.635%22%2F%3E%3Cellipse cx=%22620.271%22 cy=%221037.864%22 rx=%222.811%22 ry=%222.556%22 fill=%22%23fff%22 opacity=%22.758%22%2F%3E%3Cellipse cx=%22562.569%22 cy=%221061.413%22 rx=%221.752%22 ry=%221.743%22 fill=%22%23fff%22 opacity=%22.256%22%2F%3E%3Cellipse cx=%22311.051%22 cy=%221070.017%22 rx=%222.779%22 ry=%222.485%22 fill=%22%23fff%22 opacity=%22.212%22%2F%3E%3Cellipse cx=%221370.689%22 cy=%221079.162%22 rx=%221.645%22 ry=%221.59%22 fill=%22%23fff%22 opacity=%22.381%22%2F%3E%3Cellipse cx=%22409.326%22 cy=%221097.594%22 rx=%221.933%22 ry=%221.801%22 fill=%22%23fff%22 opacity=%22.023%22%2F%3E%3Cellipse cx=%22559.464%22 cy=%221106.702%22 rx=%222.6%22 ry=%222.455%22 fill=%22%23fff%22 opacity=%22.237%22%2F%3E%3Cellipse cx=%22225.672%22 cy=%221132.94%22 rx=%222.651%22 ry=%222.367%22 fill=%22%23fff%22 opacity=%22.701%22%2F%3E%3Cellipse cx=%221974.318%22 cy=%221139.483%22 rx=%222.698%22 ry=%222.668%22 fill=%22%23fff%22 opacity=%22.238%22%2F%3E%3Cellipse cx=%22885.71%22 cy=%221149.584%22 rx=%222.125%22 ry=%221.819%22 fill=%22%23fff%22 opacity=%22-.235%22%2F%3E%3Cellipse cx=%22217.032%22 cy=%221163.63%22 rx=%221.615%22 ry=%221.444%22 fill=%22%23fff%22 opacity=%22.717%22%2F%3E%3Cellipse cx=%2262.427%22 cy=%221186.78%22 rx=%221.782%22 ry=%221.684%22 fill=%22%23fff%22 opacity=%22.637%22%2F%3E%3Cellipse cx=%22897.602%22 cy=%221194.782%22 rx=%221.68%22 ry=%221.454%22 fill=%22%23fff%22 opacity=%22.2%22%2F%3E%3Cellipse cx=%22290.213%22 cy=%221210.888%22 rx=%221.908%22 ry=%221.479%22 fill=%22%23fff%22 opacity=%22.557%22%2F%3E%3Cellipse cx=%22148.943%22 cy=%221229.248%22 rx=%222.054%22 ry=%221.873%22 fill=%22%23fff%22 opacity=%22-.158%22%2F%3E%3Cellipse cx=%22894.454%22 cy=%221245.59%22 rx=%222.471%22 ry=%222.015%22 fill=%22%23fff%22 opacity=%22.259%22%2F%3E%3Cellipse cx=%221486.566%22 cy=%221259.36%22 rx=%222.479%22 ry=%222.115%22 fill=%22%23fff%22 opacity=%22.249%22%2F%3E%3Cellipse cx=%22609.105%22 cy=%221271.676%22 rx=%222.434%22 ry=%222.077%22 fill=%22%23fff%22 opacity=%22.866%22%2F%3E%3Cellipse cx=%22280.401%22 cy=%221278.313%22 rx=%221.55%22 ry=%221.38%22 fill=%22%23fff%22 opacity=%22-.256%22%2F%3E%3Cellipse cx=%22995.132%22 cy=%221298.17%22 rx=%223.062%22 ry=%222.637%22 fill=%22%23fff%22 opacity=%22.695%22%2F%3E%3Cellipse cx=%221784.449%22 cy=%221307.498%22 rx=%222.358%22 ry=%221.964%22 fill=%22%23fff%22 opacity=%22.205%22%2F%3E%3Cellipse cx=%22473.812%22 cy=%221325.186%22 rx=%221.934%22 ry=%221.474%22 fill=%22%23fff%22 opacity=%22.126%22%2F%3E%3Cellipse cx=%221490.129%22 cy=%221334.706%22 rx=%221.869%22 ry=%221.737%22 fill=%22%23fff%22 opacity=%22.724%22%2F%3E%3Cellipse cx=%221575.755%22 cy=%221349.372%22 rx=%222.106%22 ry=%221.714%22 fill=%22%23fff%22 opacity=%22.221%22%2F%3E%3Cellipse cx=%221910.5%22 cy=%221369.684%22 rx=%222.178%22 ry=%221.938%22 fill=%22%23fff%22 opacity=%22.611%22%2F%3E%3Cellipse cx=%221521.688%22 cy=%221375.28%22 rx=%222.151%22 ry=%222.087%22 fill=%22%23fff%22 opacity=%22.221%22%2F%3E%3Cellipse cx=%221725.505%22 cy=%221396.181%22 rx=%222.524%22 ry=%222.24%22 fill=%22%23fff%22 opacity=%22.692%22%2F%3E%3Cellipse cx=%221124.808%22 cy=%221402.433%22 rx=%222.832%22 ry=%222.649%22 fill=%22%23fff%22 opacity=%22.611%22%2F%3E%3Cg transform=%22rotate(-56.275 651.393 234.293)%22%3E%3Cdefs%3E%3ClinearGradient id=%22a%22 x1=%220%22 y1=%221%22 x2=%22275.5%22 y2=%221%22 gradientUnits=%22userSpaceOnUse%22%3E%3Cstop stop-color=%22%23fff23a%22%2F%3E%3Cstop offset=%22.3%22 stop-color=%22%23ff5e3a%22 stop-opacity=%22.1%22%2F%3E%3Cstop offset=%22.7%22 stop-color=%22%23ff5e3a%22 stop-opacity=%220%22%2F%3E%3C%2FlinearGradient%3E%3C%2Fdefs%3E%3Crect x=%22-13.775%22 y=%22-12.5%22 width=%22110.2%22 height=%2225%22 rx=%2225%22 ry=%2225%22 fill=%22url(%23a)%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22165.3%22 height=%228%22 rx=%228%22 ry=%228%22 fill=%22url(%23a)%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-57.484 614.103 -723.19)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-12.576%22 y=%22-12.5%22 width=%22100.606%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22150.909%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-57.305 906.296 -880.063)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-13.455%22 y=%22-12.5%22 width=%22107.639%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22161.459%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-59.958 1014.208 -1644.058)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-12.083%22 y=%22-12.5%22 width=%2296.667%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22145%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-58.995 812.833 22.665)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-13.061%22 y=%22-12.5%22 width=%22104.49%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22156.734%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-64.462 1432.592 -141.269)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-11.286%22 y=%22-12.5%22 width=%2290.287%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22135.43%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-61.785 1403.167 -650.465)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-12.092%22 y=%22-12.5%22 width=%2296.735%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22145.103%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3Cg transform=%22rotate(-58.814 1655.493 -1077.899)%22 fill=%22url(%23a)%22%3E%3Crect x=%22-17.119%22 y=%22-12.5%22 width=%22136.955%22 height=%2225%22 rx=%2225%22 ry=%2225%22 filter=%22url(%23b)%22 opacity=%22.4%22%2F%3E%3Crect width=%22205.433%22 height=%228%22 rx=%228%22 ry=%228%22%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E");


</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Charger le fichier css
with open("./style/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# En tête de l'application
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<h1 class="custom-text">Simplify your patent</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="custom-text2">Explainable Patent Learning for Artificial Intelligence</h1>', unsafe_allow_html=True)

