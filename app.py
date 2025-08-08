import streamlit as st
import pandas as pd
from predictor import tabpfn_predict

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, mean_squared_error, r2_score

st.set_page_config(page_title="TabPFN Predictor", layout="wide")
st.title("TabPFN Tabular Predictor")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df)

        # Task type: Classification or Regression
        task_type = st.radio("🔧 Task Type", options=["Classification", "Regression"])

        # Target column selection
        target_col = st.selectbox("🎯 Select the Target Column", df.columns)

        if st.button("🚀 Predict"):
            with st.spinner("Running TabPFN..."):
                output = tabpfn_predict(df, target_col, task_type)

                if output is None:
                    st.error("❌ TabPFN failed. Check your file and try again.")
                else:
                    df_result = output["result_df"]
                    time_taken = output["time_taken"]
                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.dataframe(df_result)

                    st.markdown(f"**⏱️ Inference Time:** {time_taken:.2f} seconds")

                    if task_type == "Classification":
                        accuracy = output["accuracy"]
                        class_mapping = output["class_mapping"]
                        y_proba = output.get("y_proba", None)

                        st.markdown(f"**🧠 Accuracy:** {accuracy * 100:.2f}%")
                        st.markdown("**🔢 Class Mapping (Label → Index):**")
                        st.code(str(class_mapping), language="json")

                        if len(df_result["True"].unique()) == 2 and y_proba is not None:
                            auc = roc_auc_score(df_result["True"], y_proba[:, 1])
                            st.markdown(f"**📈 ROC AUC:** {auc:.4f}")

                        # Confusion Matrix
                        st.subheader("📈 Visualizations")
                        cm_labels = df_result["True"].unique()
                        cm = confusion_matrix(df_result["True"], df_result["Predicted"], labels=cm_labels)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                    xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
                        ax.set_xlabel("Predicted", fontsize=10)
                        ax.set_ylabel("Actual", fontsize=10)
                        ax.set_title("Confusion Matrix", fontsize=12)
                        plt.xticks(rotation=45, fontsize=8)
                        plt.yticks(rotation=0, fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Confidence Distribution
                        fig2, ax2 = plt.subplots(figsize=(4, 3))
                        sns.histplot(df_result["Confidence"], bins=10, kde=True, ax=ax2, color="#4c72b0")
                        ax2.set_title("Confidence Distribution", fontsize=12)
                        ax2.set_xlabel("Model Confidence", fontsize=10)
                        ax2.set_ylabel("Frequency", fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig2)

                        # Class-wise Precision
                        st.subheader("📊 Class-wise Precision")
                        report = classification_report(df_result["True"], df_result["Predicted"], output_dict=True)
                        class_prec = {label: v["precision"] for label, v in report.items() if isinstance(v, dict)}

                        fig3, ax3 = plt.subplots(figsize=(5, 3))
                        sns.barplot(x=list(class_prec.keys()), y=list(class_prec.values()), ax=ax3, palette="crest")
                        ax3.set_ylabel("Precision", fontsize=10)
                        ax3.set_ylim(0, 1)
                        ax3.set_title("Class-wise Precision", fontsize=12)
                        plt.xticks(rotation=45, fontsize=8)
                        plt.yticks(fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig3)

                    elif task_type == "Regression":
                        mse = output["mse"]
                        r2 = output["r2"]
                        st.markdown(f"**📉 Mean Squared Error (MSE):** {mse:.4f}")
                        st.markdown(f"**📈 R² Score:** {r2:.4f}")

                        fig4, ax4 = plt.subplots(figsize=(5, 3))
                        sns.scatterplot(x=df_result["True"], y=df_result["Predicted"], ax=ax4)
                        ax4.set_title("True vs Predicted", fontsize=12)
                        ax4.set_xlabel("True Values", fontsize=10)
                        ax4.set_ylabel("Predicted Values", fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig4)

                    # Download
                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions", data=csv, file_name="tabpfn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")