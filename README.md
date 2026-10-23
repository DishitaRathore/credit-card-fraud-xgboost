# Credit Card Fraud Detection with XGBoost

A machine learning project that identifies potentially fraudulent credit card transactions using XGBoost. The project focuses on the challenges that make fraud detection different from ordinary classification: a highly imbalanced target, choosing a useful decision threshold, and evaluating performance with measures that reflect the cost of missed fraud.

> **Project status:** The trained model artifact is stored in `models/credit_fraud_xgboost.joblib`. Add the Colab notebook to `notebooks/credit_fraud_xgboost.ipynb` when you save it to this repository. Test-set scores are intentionally not listed here until they are copied from a verified run of the notebook.

## Project goals

- Train an XGBoost binary classifier for legitimate (`Class = 0`) and fraudulent (`Class = 1`) transactions.
- Compare it with a simple baseline and a class-weighted logistic regression baseline.
- Account for class imbalance without changing the validation or test data.
- Choose a classification threshold on validation data, then report final results on the held-out test data.
- Save the trained model, input feature order, and chosen threshold together for reuse.

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from the Machine Learning Group at Université Libre de Bruxelles, distributed through Kaggle:

- Kaggle dataset: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- The commonly distributed dataset contains 284,807 transactions and 492 fraud cases. Check the CSV you downloaded because smaller or modified copies may contain fewer rows.
- `Class` is the target: `0` means legitimate and `1` means fraudulent.
- `Time`, `Amount`, and anonymized features `V1` through `V28` are the input columns.

The dataset is not included in this repository. Download it from Kaggle and keep the CSV in your Colab session or another local data directory. Do not commit the raw dataset to GitHub.

## Repository structure

```text
credit-card-fraud-xgboost/
├── README.md
├── notebooks/
│   └── credit_fraud_xgboost.ipynb   # Add your Colab notebook here
└── models/
    └── credit_fraud_xgboost.joblib  # Trained model bundle
```

## Tools used

- Python
- pandas and NumPy for data handling
- scikit-learn for data splitting, baselines, and evaluation metrics
- XGBoost for the fraud classifier
- Matplotlib and Seaborn for exploratory plots and evaluation visualizations
- Joblib for saving and loading the trained model bundle
- Google Colab for running the notebook

## Run the project in Google Colab

1. Download `creditcard.csv` from the Kaggle dataset page above.
2. Open the notebook in `notebooks/credit_fraud_xgboost.ipynb` (or upload your notebook to Colab if it is not in the repository yet).
3. Upload `creditcard.csv` to the Colab session when prompted, or update the notebook's data path to match your Drive location.
4. Run the notebook from top to bottom. It installs/imports the required libraries, checks the data, trains the models, selects a threshold, evaluates the final model, and saves the model bundle.

Install the main packages in a notebook cell if they are not already available:

```python
!pip install -q xgboost pandas numpy scikit-learn matplotlib seaborn joblib
```

The notebook may also need the `kaggle` package if you choose to download the dataset with the Kaggle API. Uploading the CSV manually does not require it.

## Modeling workflow

1. **Load and validate the data.** Check the shape, column names, missing values, target labels, and duplicate rows. Rows without a valid `Class` label cannot be used for supervised training; investigate them before removing them.
2. **Explore the class balance.** Count legitimate and fraudulent transactions. Accuracy alone can be misleading when fraud is rare.
3. **Separate features and target.** Use `Class` as `y` and the remaining model input columns as `X`.
4. **Split before fitting.** Create stratified training, validation, and test partitions (70% / 15% / 15%, with a fixed random seed). Stratification keeps the rare fraud class represented in each split. Keep validation and test data at their original class proportions.
5. **Set the class weight from training data only.** XGBoost's `scale_pos_weight` is calculated from the legitimate-to-fraud ratio in the training partition. This gives the rare fraud class more influence during fitting.
6. **Train and compare models.** Compare a majority-class `DummyClassifier`, a class-weighted logistic regression baseline, and an `XGBClassifier`. Tree models do not require feature standardization, so the XGBoost path does not scale the features.
7. **Select a decision threshold using validation data.** The default threshold of 0.5 is not automatically best for fraud detection. Choose a threshold using the validation probabilities and a stated metric, such as F2 when recall is weighted more heavily than precision.
8. **Evaluate once on the test partition.** Apply the selected threshold to the held-out test probabilities. Report precision, recall, F1, F2, average precision / PR-AUC, ROC-AUC, and the confusion matrix. Do not use test results to tune the model or threshold.
9. **Save the model bundle.** Store the fitted model, selected threshold, and expected feature names/order together so inference uses the same inputs as training.

## Results

Add the metrics printed by your final notebook run here. Use **test-set** values from the held-out test partition, not training or validation scores. Include the selected threshold and the confusion matrix so readers can see the false-positive and missed-fraud counts.

| Test metric | Result |
| --- | ---: |
| Selected threshold | Add verified value |
| Average precision (PR-AUC) | Add verified value |
| ROC-AUC | Add verified value |
| Precision | Add verified value |
| Recall | Add verified value |
| F1 | Add verified value |
| F2 | Add verified value |
| False positives | Add verified count |
| False negatives | Add verified count |

Do not compare this result with other projects unless they use the same data version, split strategy, and evaluation method.

## Load the saved model

The model bundle is expected to contain the fitted estimator under `model`, the validation-selected decision threshold under `threshold`, and the input column list under `features`.

```python
import joblib
import pandas as pd

artifact = joblib.load("models/credit_fraud_xgboost.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# One row per transaction, with the same feature columns used during training.
transaction = pd.DataFrame([{
    "Time": 0.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "Amount": 0.0,
}])

# Enforce the exact input column order stored with the model.
transaction = transaction[features]
fraud_probability = model.predict_proba(transaction)[:, 1][0]
predicted_class = int(fraud_probability >= threshold)

print(f"Fraud probability: {fraud_probability:.4f}")
print("Prediction:", "potential fraud" if predicted_class else "likely legitimate")
```

The example values above are placeholders to demonstrate the expected input shape; they are not a real transaction and should not be used to make financial decisions. The feature values must follow the dataset's definitions and preprocessing used in the notebook.

## Limitations

- This is an educational portfolio project using a historical, anonymized dataset. It is not a production fraud prevention system.
- Fraud patterns and the costs of false positives and missed fraud change over time. A model needs monitoring, fresh representative data, and periodic evaluation before operational use.
- The selected threshold reflects the notebook's validation data and metric choice; another institution may need a different threshold based on its review capacity and business costs.
- The anonymized features limit the ability to explain predictions in terms of real-world transaction characteristics.

## Resume description

After verifying your final test results, you can adapt this project description for your resume:

> Built a credit card fraud detection pipeline with XGBoost on a highly imbalanced transaction dataset; used stratified train/validation/test splits, class weighting, validation-based threshold selection, and held-out evaluation with precision, recall, F-scores, and PR-AUC.

Add a measured result only after confirming it from the final test evaluation, for example: “achieved **[verified recall]** recall at **[verified precision]** precision on a held-out test set.”
