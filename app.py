import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="DATA_VISUALIZATION", layout="wide")
st.title("⚡ Insightify: Data Visualization App")

# ===================== Caching file load ===================== #
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# ===================== File Upload / Sample ===================== #
theme_selection = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark"])
plotly_template = "plotly_white" if theme_selection == "Light" else "plotly_dark"
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

    # ===================== Automatic Chart Suggestions ===================== #
    st.subheader("📊 Automatic Chart Suggestions")

    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        st.markdown("Here are some suggested charts based on your data:")

        # Suggestion 1: Bar chart (average of a numeric column by a categorical column)
        cat_col_1 = categorical_cols[0]
        num_col_1 = numeric_cols[0]
        st.markdown(f"#### Bar Chart: Average of `{num_col_1}` by `{cat_col_1}`")
        try:
            fig = px.bar(filtered_df, x=cat_col_1, y=num_col_1, title=f"Average of {num_col_1} by {cat_col_1}", color=cat_col_1, template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download Chart", data=fig.to_image(format="png"), file_name="bar_chart.png")
        except Exception as e:
            st.warning(f"Could not generate bar chart: {e}")

        # Suggestion 2: Box plot
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col_2 = categorical_cols[0]
            num_col_2 = numeric_cols[0]
            st.markdown(f"#### Box Plot: Distribution of `{num_col_2}` by `{cat_col_2}`")
            try:
                fig = px.box(filtered_df, x=cat_col_2, y=num_col_2, title=f"Distribution of {num_col_2} by {cat_col_2}", color=cat_col_2, template=plotly_template)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("⬇️ Download Chart", data=fig.to_image(format="png"), file_name="box_plot.png", key="download_box")
            except Exception as e:
                st.warning(f"Could not generate box plot: {e}")

    if len(numeric_cols) >= 2:
        # Suggestion 3: Scatter plot
        num_col_1 = numeric_cols[0]
        num_col_2 = numeric_cols[1]
        st.markdown(f"#### Scatter Plot: `{num_col_1}` vs `{num_col_2}`")
        try:
            fig = px.scatter(filtered_df, x=num_col_1, y=num_col_2, title=f"{num_col_1} vs {num_col_2}", template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download Chart", data=fig.to_image(format="png"), file_name="scatter_plot.png", key="download_scatter")
        except Exception as e:
            st.warning(f"Could not generate scatter plot: {e}")

    # ===================== Plotting Section ===================== #
    st.subheader("📈 Create Your Own Visualization")

    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### Plotly Charts")
    chart_type = st.selectbox("Choose Plotly Chart Type", 
                              ["None", "Scatter", "Line", "Bar", "Box", "Histogram", "Violin", "3D Scatter", "Pie"])

    plot_df = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)

    if chart_type != "None":
        try:
            if chart_type == "Scatter":
                x_axis = st.selectbox("X-axis", numeric_cols, key="px_scatter_x")
                y_axis = st.selectbox("Y-axis", numeric_cols, key="px_scatter_y")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols + numeric_cols, key="px_scatter_color")
                fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=None if color_dim == "None" else color_dim, title=f"{y_axis} vs. {x_axis}", template=plotly_template)
            
            elif chart_type == "Line":
                x_axis = st.selectbox("X-axis", numeric_cols, key="px_line_x")
                y_axis = st.selectbox("Y-axis", numeric_cols, key="px_line_y")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols, key="px_line_color")
                fig = px.line(plot_df, x=x_axis, y=y_axis, color=None if color_dim == "None" else color_dim, title=f"{y_axis} vs. {x_axis}", template=plotly_template)

            elif chart_type == "Bar":
                x_axis = st.selectbox("X-axis (Categorical)", categorical_cols, key="px_bar_x")
                y_axis = st.selectbox("Y-axis (Numeric)", numeric_cols, key="px_bar_y")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols, key="px_bar_color")
                fig = px.bar(plot_df, x=x_axis, y=y_axis, color=None if color_dim == "None" else color_dim, title=f"{y_axis} by {x_axis}", template=plotly_template)

            elif chart_type == "Box":
                x_axis = st.selectbox("X-axis (Categorical)", categorical_cols, key="px_box_x")
                y_axis = st.selectbox("Y-axis (Numeric)", numeric_cols, key="px_box_y")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols, key="px_box_color")
                fig = px.box(plot_df, x=x_axis, y=y_axis, color=None if color_dim == "None" else color_dim, title=f"Box plot of {y_axis} by {x_axis}", template=plotly_template)

            elif chart_type == "Histogram":
                x_axis = st.selectbox("X-axis", numeric_cols, key="px_hist_x")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols, key="px_hist_color")
                fig = px.histogram(plot_df, x=x_axis, color=None if color_dim == "None" else color_dim, title=f"Histogram of {x_axis}", template=plotly_template)

            elif chart_type == "Violin":
                x_axis = st.selectbox("X-axis (Categorical)", categorical_cols, key="px_violin_x")
                y_axis = st.selectbox("Y-axis (Numeric)", numeric_cols, key="px_violin_y")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols, key="px_violin_color")
                fig = px.violin(plot_df, x=x_axis, y=y_axis, color=None if color_dim == "None" else color_dim, title=f"Violin plot of {y_axis} by {x_axis}", template=plotly_template)

            elif chart_type == "3D Scatter" and len(numeric_cols) >= 3:
                x_axis = st.selectbox("X-axis", numeric_cols, key="px_3d_x")
                y_axis = st.selectbox("Y-axis", numeric_cols, key="px_3d_y")
                z_axis = st.selectbox("Z-axis", numeric_cols, key="px_3d_z")
                color_dim = st.selectbox("Color Dimension", ["None"] + categorical_cols + numeric_cols, key="px_3d_color")
                fig = px.scatter_3d(plot_df, x=x_axis, y=y_axis, z=z_axis, color=None if color_dim == "None" else color_dim, title="3D Scatter Plot", template=plotly_template)
            
            elif chart_type == "Pie":
                names_col = st.selectbox("Names (Categorical)", categorical_cols, key="px_pie_names")
                values_col = st.selectbox("Values (Numeric)", numeric_cols, key="px_pie_values")
                fig = px.pie(plot_df, names=names_col, values=values_col, title=f"Pie chart of {values_col} by {names_col}", template=plotly_template)

            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download Chart", data=fig.to_image(format="png"), file_name=f"{chart_type.lower()}_chart.png", key=f"download_{chart_type.lower()}")
        
        except Exception as e:
            st.error(f"An error occurred while creating the chart: {e}")

    # ===================== Seaborn/Matplotlib Plotting Section (optional) ===================== #
    st.markdown("### Seaborn/Matplotlib Charts")
    chart = st.selectbox("Choose Chart", ["None", "Line", "Bar", "Scatter", "Histogram", "Box", "Count", "Correlation Heatmap", "Violin", "Joint", "Pair"])
    plot_df_seaborn = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)  # sampling for speed

    if chart != "None":
        fig, ax = plt.subplots(figsize=(10, 5))

        if chart == "Line" and numeric_cols:
            col = st.selectbox("Y-axis Column", numeric_cols)
            ax.plot(plot_df_seaborn[col])
            ax.set_title(f"Line Plot of {col}")

        elif chart == "Bar" and numeric_cols and categorical_cols:
            cat = st.selectbox("X (Category)", categorical_cols)
            num = st.selectbox("Y (Numeric)", numeric_cols)
            grouped = plot_df_seaborn.groupby(cat)[num].mean().reset_index()
            ax.bar(grouped[cat], grouped[num])
            ax.set_title(f"Avg {num} by {cat}")

        elif chart == "Scatter" and len(numeric_cols) >= 2:
            x = st.selectbox("X-axis", numeric_cols, key="xscatter")
            y = st.selectbox("Y-axis", numeric_cols, key="yscatter")
            ax.scatter(plot_df_seaborn[x], plot_df_seaborn[y])
            ax.set_title(f"{y} vs {x}")

        elif chart == "Histogram" and numeric_cols:
            col = st.selectbox("Column", numeric_cols)
            ax.hist(plot_df_seaborn[col], bins=20)
            ax.set_title(f"Histogram of {col}")

        elif chart == "Box" and numeric_cols and categorical_cols:
            cat = st.selectbox("X (Category)", categorical_cols)
            num = st.selectbox("Y (Numeric)", numeric_cols)
            sns.boxplot(x=cat, y=num, data=plot_df_seaborn, ax=ax)
            ax.set_title(f"Boxplot of {num} by {cat}")

        elif chart == "Count" and categorical_cols:
            col = st.selectbox("Column", categorical_cols)
            sns.countplot(x=col, data=plot_df_seaborn, ax=ax)
            ax.set_title(f"Count Plot of {col}")
            plt.xticks(rotation=45)

        elif chart == "Correlation Heatmap" and len(numeric_cols) >= 2:
            corr = plot_df_seaborn[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.stop()

        elif chart == "Violin" and numeric_cols and categorical_cols:
            cat = st.selectbox("X (Category)", categorical_cols)
            num = st.selectbox("Y (Numeric)", numeric_cols)
            sns.violinplot(x=cat, y=num, data=plot_df_seaborn, ax=ax)
            ax.set_title(f"Violin Plot of {num} by {cat}")

        elif chart == "Joint" and len(numeric_cols) >= 2:
            x = st.selectbox("X-axis", numeric_cols, key="xjoint")
            y = st.selectbox("Y-axis", numeric_cols, key="yjoint")
            sns.jointplot(x=x, y=y, data=plot_df_seaborn, kind='scatter')
            st.pyplot(plt.gcf())
            st.stop()

        elif chart == "Pair" and len(numeric_cols) >= 2:
            st.warning("Pair plot can be slow on large datasets.")
            pair_cols = st.multiselect("Select columns for Pair Plot", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
            if len(pair_cols) >= 2:
                sns.pairplot(plot_df_seaborn[pair_cols])
                st.pyplot(plt.gcf())
                st.stop()
            else:
                st.warning("Please select at least 2 columns for the pair plot.")
                st.stop()

        ax.grid(True)
        st.pyplot(fig)

    # ===================== SHAP Section ===================== #
    st.subheader("🤖 Model Explanations (SHAP)")
    if len(numeric_cols) >= 2:
        target_var = st.selectbox("Select Target Variable (for Regression)", numeric_cols)
        
        shap_features = [col for col in numeric_cols if col != target_var]
        
        if st.button("📊 Generate SHAP Explanation"):
            with st.spinner("Training model and calculating SHAP values..."):
                model_df = filtered_df[numeric_cols].dropna()
                X = model_df[shap_features]
                y = model_df[target_var]

                if not X.empty and not y.empty:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    st.markdown("### SHAP Summary Plot")
                    st.info("This plot shows the most important features and their impact on the model's predictions.")
                    fig_shap, ax_shap = plt.subplots()
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    st.pyplot(fig_shap)
                else:
                    st.warning("Not enough data to train a model after dropping missing values.")
    else:
        st.warning("Not enough numeric columns to build a model for SHAP analysis.")

else:
    st.info("📂 Upload a CSV file or select a sample dataset to get started.")
