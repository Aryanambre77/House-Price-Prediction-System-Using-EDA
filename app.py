import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
import os

# ------------------------ Streamlit UI Theme Setup ------------------------
st.set_page_config(page_title="Sales Prediction", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    html, body, [class*="css"] {
        color: white;
        background-color: #121212;
    }
    label, .css-1v3fvcr, .css-10trblm, .css-1n76uvr {
        color: white !important;
    }
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div {
        background-color: #1e1e1e !important;
        color: white !important;
    }
    .stButton > button {
        background-color: #1db954;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-weight: bold;
        font-size: 16px;
        transition: 0.3s;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1ed760;
        transform: scale(1.05);
    }
    .stSlider > div > div > div {
        color: white !important;
    }
    header[data-testid="stHeader"] {
        background: #121212;
    }
    header[data-testid="stHeader"] .css-1dp5vir {
        color: white !important;
    }
    .stCheckbox > div > label,
    .css-1r6slb0,
    .stCheckbox label {
        color: red !important;
    }
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        border: none;
    }
    .stDownloadButton > button:hover {
        background-color: #eeeeee !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------ Load Assets ------------------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("linear_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    template = pd.read_csv("model_input_template.csv")
    train_df = pd.read_csv("train.csv")
    return model, scaler, template, train_df

model, scaler, template, train_df = load_assets()
template_columns = template.columns

valid_neighborhoods = sorted([col.replace("Neighborhood_", "") for col in template.columns if col.startswith("Neighborhood_")])
valid_housestyles = sorted([col.replace("HouseStyle_", "") for col in template.columns if col.startswith("HouseStyle_")])

# ------------------------ Navigation ------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ------------------------ HOME PAGE ------------------------
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center; color: white;'>🏠 Sales Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>Predict Ames Housing Prices with Machine Learning 🎯</p>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; padding-top: 50px;'>", unsafe_allow_html=True)
    if st.button("🚀 Go to Prediction Page"):
        st.session_state.page = "predict"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ PREDICTION PAGE ------------------------
elif st.session_state.page == "predict":
    st.markdown("<h2 style='color:#1db954;'>🏡 Ames House Price Prediction</h2>", unsafe_allow_html=True)

    # ---- Input Fields ----
    OverallQual = st.slider("Overall Quality (1=Very Poor, 10=Very Excellent)", 1, 10, 5)
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
    GarageCars = st.slider("Garage Size (Cars)", 0, 4, 2)
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    FullBath = st.selectbox("Full Bathrooms", [0, 1, 2, 3])
    YearBuilt = st.number_input("Year Built", 1872, 2025, 2000)
    Neighborhood = st.selectbox("Neighborhood", valid_neighborhoods)
    HouseStyle = st.selectbox("House Style", valid_housestyles)

    # ---- Input DataFrame ----
    input_df = pd.DataFrame(columns=template_columns)
    input_df.loc[0] = [0] * len(template_columns)
    input_df.at[0, 'OverallQual'] = OverallQual
    input_df.at[0, 'GrLivArea'] = GrLivArea
    input_df.at[0, 'GarageCars'] = GarageCars
    input_df.at[0, 'TotalBsmtSF'] = TotalBsmtSF
    input_df.at[0, 'FullBath'] = FullBath
    input_df.at[0, 'YearBuilt'] = YearBuilt

    nb_col = f"Neighborhood_{Neighborhood}"
    hs_col = f"HouseStyle_{HouseStyle}"
    if nb_col in input_df.columns:
        input_df.at[0, nb_col] = 1
    if hs_col in input_df.columns:
        input_df.at[0, hs_col] = 1

    scaled_input = scaler.transform(input_df)

    # ---- Predict Button ----
    if st.button("🔮 Predict House Price"):
        price = model.predict(scaled_input)[0]
        if price < 0:
            st.error("❌ Invalid prediction. Try different values.")
        else:
            st.markdown(
                f"""
                <div style='
                    background-color: #1db954;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    margin-top: 20px;
                '>
                    🏠 Estimated Sale Price: ${price:,.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- Show EDA Charts Toggle ----
    show_eda = st.checkbox("📊 Show EDA Charts ")
    if show_eda:
        st.subheader("🔎 Exploratory Data Analysis")

        # 1. Missing Values
        st.write("📌 Top Missing Value Features")
        missing = train_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        top_missing = missing.head(10)
        fig1, ax1 = plt.subplots()
        sns.barplot(x=top_missing.values, y=top_missing.index, palette="crest", ax=ax1)
        ax1.set_title("Top Features with Missing Values")
        ax1.set_xlabel("Count")
        st.pyplot(fig1)

        # 2. Correlation Heatmap
        st.write("📌 Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(train_df.corr(numeric_only=True), cmap="viridis", annot=False, ax=ax2)
        st.pyplot(fig2)

        # 3. Feature Importance
        st.write("📌 Feature Importance")
        if hasattr(model, "coef_"):
            coefs = model.coef_.flatten()
            top_feats = pd.Series(coefs, index=template_columns).sort_values(key=abs, ascending=False).head(10)
            fig3, ax3 = plt.subplots()
            sns.barplot(x=top_feats.values, y=top_feats.index, palette="magma", ax=ax3)
            ax3.set_title("Top 10 Important Features")
            st.pyplot(fig3)

        # ---- Export to PDF ----
        if st.button("📄 Export EDA to PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="EDA Charts Summary", ln=True, align="C")

            for fig in [fig1, fig2, fig3]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, format="png")
                    tmpfile.close()
                    pdf.image(tmpfile.name, w=180)
                    os.unlink(tmpfile.name)

            pdf_output = "EDA_Report.pdf"
            pdf.output(pdf_output)
            st.success("✅ PDF exported successfully!")
            with open(pdf_output, "rb") as f:
                st.download_button("📥 Download PDF", f, file_name="EDA_Report.pdf", mime="application/pdf")

    # ---- Back to Home ----
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()





# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64
# from fpdf import FPDF
#
# # Page config
# st.set_page_config(page_title="Sales Prediction", layout="centered")
#
# # Custom dark theme
# st.markdown("""
#     <style>
#     .stApp { background-color: #121212; color: white; font-family: 'Segoe UI', sans-serif; }
#     html, body, [class*="css"] { color: white; background-color: #121212; }
#     label, .css-1v3fvcr, .css-10trblm, .css-1n76uvr { color: white !important; }
#     .stTextInput input, .stNumberInput input, .stSelectbox div {
#         background-color: #1e1e1e !important; color: white !important;
#     }
#     .stButton > button {
#         background-color: #1db954; color: white; border-radius: 10px;
#         padding: 0.5em 2em; font-weight: bold; font-size: 16px;
#         transition: 0.3s; border: none;
#     }
#     .stButton > button:hover {
#         background-color: #1ed760; transform: scale(1.05);
#     }
#     .stSlider > div > div > div { color: white !important; }
#     header[data-testid="stHeader"] { background: #121212; }
#     header[data-testid="stHeader"] .css-1dp5vir { color: white !important; }
#
#     /* Fix checkbox label color */
# .stCheckbox > div > label {
#     color: white !important;
# }
#
#     </style>
# """, unsafe_allow_html=True)
#
# # Load model/scaler/template
# @st.cache_resource
# def load_assets():
#     model = pickle.load(open("linear_model.pkl", "rb"))
#     scaler = pickle.load(open("scaler.pkl", "rb"))
#     template = pd.read_csv("model_input_template.csv")
#     return model, scaler, template
#
# @st.cache_data
# def load_training_data():
#     return pd.read_csv("train.csv")
#
# model, scaler, template = load_assets()
# train_df = load_training_data()
# template_columns = template.columns
#
# valid_neighborhoods = sorted([col.replace("Neighborhood_", "") for col in template.columns if col.startswith("Neighborhood_")])
# valid_housestyles = sorted([col.replace("HouseStyle_", "") for col in template.columns if col.startswith("HouseStyle_")])
#
# if "page" not in st.session_state:
#     st.session_state.page = "home"
#
# # ----------- Home Page -----------
# if st.session_state.page == "home":
#     st.markdown("<h1 style='text-align: center;'>🏠 Sales Prediction App</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center; font-size:18px;'>Predict Ames Housing Prices with Machine Learning 🎯</p>", unsafe_allow_html=True)
#     st.markdown("<div style='text-align: center; padding-top: 50px;'>", unsafe_allow_html=True)
#     if st.button("🚀 Go to Prediction Page"):
#         st.session_state.page = "predict"
#         st.rerun()
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # ----------- Prediction Page -----------
# elif st.session_state.page == "predict":
#     st.markdown("<h2 style='color:#1db954;'>🏡 Ames House Price Prediction</h2>", unsafe_allow_html=True)
#
#     OverallQual = st.slider("Overall Quality (1=Very Poor, 10=Very Excellent)", 1, 10, 5)
#     GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
#     GarageCars = st.slider("Garage Size (Cars)", 0, 4, 2)
#     TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
#     FullBath = st.selectbox("Full Bathrooms", [0, 1, 2, 3])
#     YearBuilt = st.number_input("Year Built", 1872, 2025, 2000)
#     Neighborhood = st.selectbox("Neighborhood", valid_neighborhoods)
#     HouseStyle = st.selectbox("House Style", valid_housestyles)
#
#     input_df = pd.DataFrame(columns=template_columns)
#     input_df.loc[0] = [0] * len(template_columns)
#     input_df.at[0, 'OverallQual'] = OverallQual
#     input_df.at[0, 'GrLivArea'] = GrLivArea
#     input_df.at[0, 'GarageCars'] = GarageCars
#     input_df.at[0, 'TotalBsmtSF'] = TotalBsmtSF
#     input_df.at[0, 'FullBath'] = FullBath
#     input_df.at[0, 'YearBuilt'] = YearBuilt
#
#     nb_col = f"Neighborhood_{Neighborhood}"
#     hs_col = f"HouseStyle_{HouseStyle}"
#     if nb_col in input_df.columns: input_df.at[0, nb_col] = 1
#     if hs_col in input_df.columns: input_df.at[0, hs_col] = 1
#
#     scaled_input = scaler.transform(input_df)
#
#     if st.button("🔮 Predict House Price"):
#         price = model.predict(scaled_input)[0]
#         if price < 0:
#             st.error("❌ Invalid prediction. Try different values.")
#         else:
#             st.markdown(f"""
#                 <div style='background-color: #1db954; padding: 20px; border-radius: 12px;
#                             text-align: center; color: white; font-size: 24px; font-weight: bold; margin-top: 20px;'>
#                     🏠 Estimated Sale Price: ${price:,.2f}
#                 </div>""", unsafe_allow_html=True)
#
#     st.markdown("<br><br>", unsafe_allow_html=True)
#
#     # ---------- EDA Toggle ----------
#     if st.checkbox("📊 Show EDA Charts"):
#         st.subheader("📈 Multivariate EDA Visualizations")
#
#         # SalePrice Histogram colored by OverallQual
#         fig1 = plt.figure(figsize=(9, 4))
#         sns.histplot(data=train_df, x="SalePrice", hue="OverallQual", multiple="stack", palette="viridis")
#         plt.title("Sale Price Distribution by Overall Quality")
#         st.pyplot(fig1)
#
#         # Neighborhood boxplot
#         fig2 = plt.figure(figsize=(12, 5))
#         top_neigh = train_df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
#         sns.boxplot(data=train_df, x="Neighborhood", y="SalePrice", order=top_neigh, palette="Set3")
#         plt.xticks(rotation=45)
#         plt.title("Neighborhood-wise Sale Price Spread")
#         st.pyplot(fig2)
#
#         # GrLivArea vs SalePrice by OverallQual
#         fig3 = plt.figure(figsize=(9, 5))
#         sns.scatterplot(data=train_df, x="GrLivArea", y="SalePrice", hue="OverallQual", size="OverallQual",
#                         palette="Spectral", sizes=(20, 200))
#         plt.title("GrLivArea vs SalePrice Colored by Quality")
#         st.pyplot(fig3)
#
#         # Feature Importance
#         st.subheader("📌 Feature Importance (Top 10 Coefficients)")
#         try:
#             feature_importance = pd.Series(np.abs(model.coef_), index=input_df.columns).sort_values(ascending=False).head(10)
#             fig4 = plt.figure(figsize=(8, 4))
#             sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="coolwarm")
#             plt.title("Top 10 Most Influential Features")
#             st.pyplot(fig4)
#         except Exception as e:
#             st.warning(f"Couldn't load feature importance: {e}")
#
#         # EDA PDF Download
#         st.subheader("📄 Download EDA Summary Report")
#         pdf_bytes = FPDF()
#         pdf_bytes.add_page()
#         pdf_bytes.set_font("Arial", 'B', 16)
#         pdf_bytes.cell(0, 10, "EDA Report - House Price Prediction", ln=True, align="C")
#         pdf_bytes.set_font("Arial", '', 12)
#         pdf_bytes.ln(10)
#         pdf_bytes.multi_cell(0, 10,
#             "1. SalePrice is right-skewed with a long tail.\n"
#             "2. Higher OverallQual leads to higher prices.\n"
#             "3. GrLivArea is positively correlated with price.\n"
#             "4. Neighborhoods differ significantly in price.\n"
#             "5. Model feature importance confirms this trend.")
#         pdf_path = "eda_report.pdf"
#         pdf_bytes.output(pdf_path)
#         with open(pdf_path, "rb") as f:
#             b64 = base64.b64encode(f.read()).decode()
#             href = f'<a href="data:application/pdf;base64,{b64}" download="EDA_Report.pdf">📥 Download PDF Report</a>'
#             st.markdown(href, unsafe_allow_html=True)
#
#     st.markdown("<br><br>", unsafe_allow_html=True)
#     if st.button("⬅️ Back to Home"):
#         st.session_state.page = "home"
#         st.rerun()




