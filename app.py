import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="DATA_VISUALIZATION", layout="wide")
st.title("⚡ Insightify: Data Visualization App")

# ===================== Caching file load ===================== #
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# ===================== File Upload / Sample ===================== #
seaborn_datasets = ['iris', 'tips', 'diamonds', 'titanic', 'penguins']
sample = st.sidebar.selectbox("📚 Sample Dataset", ["None"] + seaborn_datasets)
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

df = None
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ CSV Uploaded!")
elif sample != "None":
    df = sns.load_dataset(sample)
    st.success(f"✅ Loaded Sample Dataset: {sample}")

# ===================== Preview + Filter ===================== #
if df is not None:
    st.subheader("🔍 Data Preview")
    row_limit = st.slider("How many rows to show?", 100, min(10000, len(df)), 500)
    st.dataframe(df.head(row_limit))

    st.subheader("🔎 Filter Data")
    col_selector = st.multiselect("Select Columns to Use", df.columns.tolist(), default=df.columns.tolist())

    filtered_df = df[col_selector].copy()

    for col in col_selector:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) < 100:  # avoid laggy dropdowns
                selected_vals = st.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
        elif pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            range_vals = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[filtered_df[col].between(*range_vals)]

    st.markdown("### 📄 Filtered Data (Paginated)")
    page = st.number_input("Page number", 1, max(1, len(filtered_df)//100 + 1))
    st.dataframe(filtered_df.iloc[(page-1)*100: page*100])
    st.download_button("⬇️ Download Filtered CSV", data=filtered_df.to_csv(index=False).encode(), file_name="filtered.csv")

    # ===================== Plotting Section ===================== #
    st.subheader("📈 Data Visualization")

    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

    chart = st.selectbox("Choose Chart", ["Line", "Bar", "Scatter", "Histogram", "Box", "Count", "Correlation Heatmap"])
    plot_df = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)  # sampling for speed

    fig, ax = plt.subplots(figsize=(10, 5))

    if chart == "Line" and numeric_cols:
        col = st.selectbox("Y-axis Column", numeric_cols)
        ax.plot(plot_df[col])
        ax.set_title(f"Line Plot of {col}")

    elif chart == "Bar" and numeric_cols and categorical_cols:
        cat = st.selectbox("X (Category)", categorical_cols)
        num = st.selectbox("Y (Numeric)", numeric_cols)
        grouped = plot_df.groupby(cat)[num].mean().reset_index()
        ax.bar(grouped[cat], grouped[num])
        ax.set_title(f"Avg {num} by {cat}")

    elif chart == "Scatter" and len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols, key="xscatter")
        y = st.selectbox("Y-axis", numeric_cols, key="yscatter")
        ax.scatter(plot_df[x], plot_df[y])
        ax.set_title(f"{y} vs {x}")

    elif chart == "Histogram" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        ax.hist(plot_df[col], bins=20)
        ax.set_title(f"Histogram of {col}")

    elif chart == "Box" and numeric_cols and categorical_cols:
        cat = st.selectbox("X (Category)", categorical_cols)
        num = st.selectbox("Y (Numeric)", numeric_cols)
        sns.boxplot(x=cat, y=num, data=plot_df, ax=ax)
        ax.set_title(f"Boxplot of {num} by {cat}")

    elif chart == "Count" and categorical_cols:
        col = st.selectbox("Column", categorical_cols)
        sns.countplot(x=col, data=plot_df, ax=ax)
        ax.set_title(f"Count Plot of {col}")
        plt.xticks(rotation=45)

    elif chart == "Correlation Heatmap" and len(numeric_cols) >= 2:
        corr = plot_df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.stop()

    ax.grid(True)
    st.pyplot(fig)

else:
    st.info("📂 Upload a CSV file or select a sample dataset to get started.")
