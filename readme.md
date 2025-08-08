# TabPFN Tabular Predictor App

A lightweight, interactive **Streamlit** app that uses [**TabPFN**](https://github.com/PriorLabs/TabPFN) — a zero-shot pretrained transformer model — to perform **tabular classification and regression** without training.

> Powered by Meta AI’s TabPFN foundation model  
> Supports both classification and regression  
> Includes metrics, visualizations, and result export

---

## Features

- Upload any `.csv` with tabular data
- Choose target column and task type (classification or regression)
- One-shot prediction — no model training required!
- Metrics:
  - Classification: Accuracy, ROC AUC (if binary)
  - Regression: R² Score, MSE
- Compact, clean visualizations
- Downloadable result table

---

## Demo

<img src="https://user-images.githubusercontent.com/your-screenshot-placeholder" width="800"/>

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/RajashSankarBordoloi/tabpfn-predictor.git
   cd tabpfn-predictor
   ```

2. Install dependencies (or use `requirements.txt`):
   ```bash
   pip install streamlit pandas scikit-learn matplotlib seaborn tabpfn
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Sample CSVs

You can test the app using datasets like:

- Iris
- Breast Cancer
- Wine
- Penguins
- Titanic  
  _(Upload as `.csv` and select the target column.)_

---

## About TabPFN

**TabPFN** is a transformer-based foundation model for tabular data that performs inference by simulating thousands of training tasks. It eliminates the need for manual training or hyperparameter tuning for small tabular datasets.

> Paper: [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848)  
> Code: [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)

---

## Credits

- Model: [TabPFN by Meta AI](https://github.com/PriorLabs/TabPFN)
- UI: Built with [Streamlit](https://streamlit.io/)
- Developed by: [@RajashSankarBordoloi](https://github.com/RajashSankarBordoloi)

---

## License

This project is open source under the [MIT License](LICENSE).
