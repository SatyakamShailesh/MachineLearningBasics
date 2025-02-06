import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import streamlit as st
import base64

st.title("Error Clustering Application")
st.write("Upload your Excel file to categorize and cluster errors.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="clustered_data.csv">Download clustered data as CSV</a>'
    return href

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Display the first few rows of the dataset
    st.write("Dataset Preview:")
    st.write(df.head())

    columns = df.columns.to_list()
    # Select the columns to use for clustering
    st.write("Select the columns to use for clustering:")
    selected_columns = st.multiselect("Select columns", columns)

    submit_button = st.button(label="Submit")

    if selected_columns and submit_button:
        # Initialize the TF-IDF vectorizers for each feature
        tfid_vectorizer = TfidfVectorizer(stop_words='english')

        # Initialize KMeans clustering
        kmeans = KMeans(n_clusters=9, random_state=42)

        # Create a pipeline for clustering
        pipeline = Pipeline(steps=[
            ('features', ColumnTransformer(
                transformers=[
                    (col, tfid_vectorizer, col) for col in selected_columns
                ])),
            ('scaler', StandardScaler(with_mean=False)),  # Optional: Scale the features to normalize the variance
            ('svd', TruncatedSVD(n_components=50)),  # Use TruncatedSVD for dimensionality reduction
            ('kmeans', kmeans)  # Apply KMeans clustering
        ])

        # Fit the model and predict clusters
        df['Cluster'] = pipeline.fit_predict(df[selected_columns])

        # Print count of each cluster
        st.write("Count of each cluster:")
        st.write(df['Cluster'].value_counts())

        # Show a plot of the count of each cluster over time
        df['created_at'] = pd.to_datetime(df['created_at'])

        # Show a bar plot of the count of each cluster over created_at
        df.groupby(['created_at', 'Cluster']).size().unstack().plot(kind='bar', stacked=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # Download the clustered data as an Excel file
        st.write("Download the clustered data as an Excel file:")
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)