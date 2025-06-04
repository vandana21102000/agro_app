import streamlit as st
from streamlit_option_menu import option_menu
import base64
import pickle
import numpy as np
import os
import joblib
import pandas as pd
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="AGRO APP",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to add background image with optional blur (used only for Home page)
def add_bg_from_local(image_file, blur=True):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    blur_css = "backdrop-filter: blur(6px);" if blur else ""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .top-box, .left-box {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            {blur_css}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
crop_model_path = os.path.join(working_dir, 'RF_Crop.joblib')
rainfall_model_path = os.path.join(working_dir, 'Rainfall_xgboost.joblib')
crop_recom_model = joblib.load(crop_model_path)
rainfall_model = joblib.load(rainfall_model_path)

# Sidebar with title icon and extended options

with st.sidebar:
    selected = option_menu(
        menu_title="🌾 AGRO APP",
        options=[
            "Home", 
            "Crop Recommendation", 
            "Rainfall Prediction",
            "Crop Recommendation Model Creation",
            "Rainfall Prediction Model Creation",
            "Meet the Creator"
        ],
        icons=[ "house", "tree", "cloud-drizzle", "bar-chart-line", "bar-chart-line","person-circle"],
        default_index=0,
        orientation="vertical"
    )

# App Header with Logo
st.markdown(
    f"""
    <div style='display: flex; align-items: center; gap: 16px; padding: 1.5rem 2rem 1rem 2rem; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; background-color: rgba(255, 255, 255, 0.95); border-radius: 12px; margin: 1rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);'>
        <img src="data:image/png;base64,{base64.b64encode(open('agro_logo_img.jpg', 'rb').read()).decode()}" width="90" height="90" style="border-radius: 12px;" />
        <div>
            <h1 style='font-size: 44px; margin: 0; color: #117A65; font-weight: 800; font-family: "Georgia", serif; letter-spacing: 1px;'>Agro App</h1>
            <p style='font-size: 15px; margin: 0; color: #555; font-style: italic;'>Helping farmers make smarter decisions with Machine Learning</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# # Helper function to display notebooks (.ipynb)

def display_notebook(nb_path, height=1600):
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # Convert notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True
        html_exporter.exclude_output_prompt = True
        html_exporter.template_name = 'classic'

        (body, resources) = html_exporter.from_notebook_node(notebook)

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(body, 'html.parser')

        # Style input code cells
        for div in soup.find_all("div", class_="input"):
            div['style'] = (
                "background-color: #f0f0f0;"
                "border: 1px solid #d0d0d0;"
                "border-radius: 5px;"
                "padding: 12px;"
                "margin-bottom: 16px;"
                "box-shadow: 1px 1px 4px rgba(0,0,0,0.1);"
                "font-family: monospace;"
            )

        # Style output cells
        for div in soup.find_all("div", class_="output"):
            div['style'] = (
                "background-color: #e8f5e9;"    
                "border: 1px solid #a5d6a7;"
                "border-radius: 5px;"
                "padding: 12px;"
                "margin-bottom: 16px;"
                "box-shadow: 1px 1px 4px rgba(0,0,0,0.05);"
                "font-family: monospace;"
            )

        html_body = str(soup.find("body"))

        # Render in Streamlit
        components.html(html_body, height=height, scrolling=True)

    except FileNotFoundError:
        st.error(f"Notebook file '{nb_path}' not found!")
    except Exception as e:
        st.error(f"Error displaying notebook: {e}")


# Page Logic
if selected == "Home":
    add_bg_from_local("home_img.jpg", blur=False)
    st.markdown("""
    <div class='top-box'>
        <h2 style='color:#117A65;font-family: "Segoe UI";'> 🌱Welcome to Agro App </h2>
        <p style='font-size:16px;'>Agro App empowers farmers with cutting-edge technology—accurately predicting rainfall using our advanced XGBoost Regression model and providing intelligent crop recommendations through a powerful Random Forest algorithm.</p>
        <p style='font-size:16px;'>🌿 Make smarter, data-driven decisions for a more productive harvest.</p>
        <p><strong>🔗 GitHub:</strong> <a href='https://github.com/vandana21102000/agro_app.git' target='_blank'>https://github.com/vandana21102000/agro_app.git</a></p>
        <p style='color:#888;'>Made with 💖 by Vandana</p>
    </div>
    """, unsafe_allow_html=True)

elif selected == "Crop Recommendation":
    st.markdown("<div class='left-box'>", unsafe_allow_html=True)
    st.subheader("Crop Recommendation")

    N = st.number_input("Nitrogen (N)", min_value=0, value=0)
    P = st.number_input("Phosphorus (P)", min_value=0, value=0)
    K = st.number_input("Potassium (K)", min_value=0, value=0)
    pH = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)

    if st.button("Recommend Crop"):
        crop_input = np.array([[N, P, K, pH, temperature, humidity, rainfall]])
        if all(crop_input[0][:3]):
            prediction = crop_recom_model.predict(crop_input)
            st.success(f"🌾 Recommended Crop: **{prediction[0]}**")
        else:
            st.error("Please fill in all required nutrient values (N, P, K)")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Rainfall Prediction":
    st.markdown("<div class='left-box'>", unsafe_allow_html=True)
    st.subheader("Rainfall Prediction")

    subdivisions = ["ANDAMAN & NICOBAR ISLANDS", "ARUNACHAL PRADESH", "ASSAM & MEGHALAYA", "NAGA MANI MIZO TRIPURA", "SUB HIMALAYAN WEST BENGAL & SIKKIM", "GANGETIC WEST BENGAL", "ORISSA", "JHARKHAND", "BIHAR", "EAST UTTAR PRADESH", "WEST UTTAR PRADESH", "UTTARAKHAND", "HARYANA DELHI & CHANDIGARH", "PUNJAB", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "WEST RAJASTHAN", "EAST RAJASTHAN", "WEST MADHYA PRADESH", "EAST MADHYA PRADESH", "GUJARAT REGION", "SAURASHTRA & KUTCH", "KONKAN & GOA", "MADHYA MAHARASHTRA", "MATATHWADA", "VIDARBHA", "CHHATTISGARH", "COASTAL ANDHRA PRADESH", "TELANGANA", "RAYALSEEMA", "TAMIL NADU", "COASTAL KARNATAKA", "NORTH INTERIOR KARNATAKA", "SOUTH INTERIOR KARNATAKA", "KERALA", "LAKSHADWEEP"]

    subdivision = st.selectbox("Subdivision", subdivisions)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    monthly_rainfall = {month: st.number_input(f"{month} Rainfall (mm)", min_value=0.0, value=0.0) for month in months}

    seasonal = {
        'Jan-Feb': monthly_rainfall['JAN'] + monthly_rainfall['FEB'],
        'Mar-May': monthly_rainfall['MAR'] + monthly_rainfall['APR'] + monthly_rainfall['MAY'],
        'Jun-Sep': monthly_rainfall['JUN'] + monthly_rainfall['JUL'] + monthly_rainfall['AUG'] + monthly_rainfall['SEP'],
        'Oct-Dec': monthly_rainfall['OCT'] + monthly_rainfall['NOV'] + monthly_rainfall['DEC']
    }

    if st.button("Predict Rainfall"):
        input_data = {'SUBDIVISION': [subdivision], 'YEAR': [year]}
        input_data.update({m: [monthly_rainfall[m]] for m in months})
        input_data.update({s: [seasonal[s]] for s in seasonal})

        df_input = pd.DataFrame(input_data)
        prediction = rainfall_model.predict(df_input)
        st.success(f"☔ Predicted Rainfall: **{prediction[0]:.2f} mm**")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Crop Recommendation Model Creation":
    st.subheader("📘 Crop Recommendation Model Creation")

    # Dict of crop notebooks (6 ipynb)
    crop_notebooks = {
        "1. Data Preprocessing": "EDA_crop_recommendation.ipynb",
        "2. Model - Random Forest": "Random_Forest.ipynb",
        "3. Model - Naive Bayes": "Gaussian Navie Bayes.ipynb",
        "4. Model - Decision Tree": "Decision Tree.ipynb",
        "5. Model - SVM": "SVM.ipynb",
        "6. Model Comparision": "Model Comparison.ipynb"
    }
    crop_dataset_path = "Crop_recommendation.csv"  

    notebook_choice = st.selectbox("Select Crop Notebook to View:", list(crop_notebooks.keys()))
    display_notebook(crop_notebooks[notebook_choice])

    st.markdown("---")
    st.subheader("Crop Dataset Preview")
    try:
        df_crop = pd.read_csv(crop_dataset_path)
        st.write(df_crop.head(10))
    except FileNotFoundError:
        st.error(f"Dataset file '{crop_dataset_path}' not found!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

elif selected == "Rainfall Prediction Model Creation":
    st.subheader("📘 Rainfall Prediction Model Creation")

    rainfall_notebooks = {
        "1. Data Preprocessing":"EDA_Rainfall_Prediction.ipynb",
        "2. Model Creation": "Models.ipynb",
        "3. Model Selection": "XGboost_Regression.ipynb"
    }
    rainfall_dataset_path = "rainfall in india 1901-2015.csv"  

    notebook_choice = st.selectbox("Select Rainfall Notebook to View:", list(rainfall_notebooks.keys()))
    display_notebook(rainfall_notebooks[notebook_choice])

    st.markdown("---")
    st.subheader("Rainfall Dataset Preview")
    try:
        df_rainfall = pd.read_csv(rainfall_dataset_path)
        st.write(df_rainfall.head(10))
    except FileNotFoundError:
        st.error(f"Dataset file '{rainfall_dataset_path}' not found!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

elif selected == "Meet the Creator":
    st.markdown("""
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2E8B57;
        line-height: 1.6;
        max-width: 700px;
        margin: 2rem auto;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(46, 134, 193, 0.2);
    ">
        <h2 style="
            font-weight: 900;
            font-size: 32px;
            margin-bottom: 0.2rem;
            color: #117A65;
        ">👩‍💻 Vandana Udayakumar</h2>
        <p style="
            font-size: 18px;
            font-weight: 600;
            margin-top: 0;
            color: #145A32;
        ">Data Analyst </p>
        <p style="font-size: 16px; margin-top: 1rem;">
            Hello! I'm <strong>Vandana Udayakumar</strong>, a passionate Data Analyst dedicated to turning data into impactful insights.
        </p>
        <p style="font-size: 16px;">
            I love creating data-driven solutions that empower farmers and optimize agriculture for a sustainable future.
        </p>
        <p style="font-size: 16px;">
            This app is my project to help farmers with intelligent crop recommendations and accurate rainfall predictions using Machine Learning.
        </p>
        <p style="font-size: 16px; margin-top: 1.5rem;">
            Feel free to connect with me on:
        </p>
        <ul style="font-size: 16px;">
            <li><a href='https://www.linkedin.com/in/vandana-udayakumar/' target='_blank' style="color: #117A65; text-decoration:none;">LinkedIn</a></li>
            <li><a href='https://github.com/vandana21102000' target='_blank' style="color: #117A65; text-decoration:none;">GitHub</a></li>
            <li><a href='https://vandana21102000.github.io/portfolio_vandana/' target='_blank' style="color: #117A65; text-decoration:none;">Portfolio</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
