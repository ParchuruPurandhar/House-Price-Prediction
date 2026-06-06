import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("House.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

model, preprocessor = load_model()
df = load_data()

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "EDA",
        "Prediction",
        "Model Performance"
    ]
)

if page == "Home":

    st.title("🏠 House Price Prediction")

    st.write("""
    This application predicts house prices using
    Linear Regression.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", "0.653")
    col2.metric("MAE", "₹970K")
    col3.metric("RMSE", "₹1.32M")

    st.subheader("Dataset Sample")
    st.dataframe(df.head())

elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(
        [
            "Price Distribution",
            "Area vs Price",
            "Correlation"
        ]
    )

    with tab1:
        fig, ax = plt.subplots()
        sns.histplot(df["price"], kde=True, ax=ax)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="area",
            y="price",
            ax=ax
        )
        st.pyplot(fig)

    with tab3:

        numeric_df = df.select_dtypes(
            include=["int64", "float64"]
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

elif page == "Prediction":

    st.title("💰 Predict House Price")

    col1, col2 = st.columns(2)

    with col1:

        area = st.number_input(
            "Area",
            min_value=500,
            value=5000
        )

        bedrooms = st.selectbox(
            "Bedrooms",
            [1,2,3,4,5,6]
        )

        bathrooms = st.selectbox(
            "Bathrooms",
            [1,2,3,4]
        )

        stories = st.selectbox(
            "Stories",
            [1,2,3,4]
        )

        parking = st.selectbox(
            "Parking",
            [0,1,2,3]
        )

    with col2:

        mainroad = st.selectbox(
            "Main Road",
            ["yes","no"]
        )

        guestroom = st.selectbox(
            "Guest Room",
            ["yes","no"]
        )

        basement = st.selectbox(
            "Basement",
            ["yes","no"]
        )

        hotwaterheating = st.selectbox(
            "Hot Water Heating",
            ["yes","no"]
        )

        airconditioning = st.selectbox(
            "Air Conditioning",
            ["yes","no"]
        )

        prefarea = st.selectbox(
            "Preferred Area",
            ["yes","no"]
        )

        furnishingstatus = st.selectbox(
            "Furnishing Status",
            [
                "furnished",
                "semi-furnished",
                "unfurnished"
            ]
        )

    if st.button("Predict Price"):

        input_df = pd.DataFrame({

            "area":[area],
            "bedrooms":[bedrooms],
            "bathrooms":[bathrooms],
            "stories":[stories],
            "mainroad":[mainroad],
            "guestroom":[guestroom],
            "basement":[basement],
            "hotwaterheating":[hotwaterheating],
            "airconditioning":[airconditioning],
            "parking":[parking],
            "prefarea":[prefarea],
            "furnishingstatus":[furnishingstatus]

        })

        transformed = preprocessor.transform(input_df)

        prediction = model.predict(transformed)[0]

        st.success(
            f"🏠 Predicted House Price: ₹ {prediction:,.0f}"
        )

elif page == "Model Performance":

    st.title("📈 Model Performance")

    results = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ],

        "MAE":[
            970043,
            1017471,
            1054209
        ],

        "RMSE":[
            1324507,
            1399788,
            1440790
        ],

        "R² Score":[
            0.652924,
            0.612350,
            0.589307
        ]
    })

    st.dataframe(results)

    st.success(
        "Linear Regression selected as final model."
    )
