# Credit Risk Modeling in Python

A walkthrough of building and evaluating a **probability-of-default (PD)** model on loan data, and turning model output into a **loan approval strategy** (acceptance rate, bad rate, expected loss).

> **Note:** This is a learning project. It follows the exercises of DataCamp's
> [Credit Risk Modeling in Python](https://www.datacamp.com/courses/credit-risk-modeling-in-python)
> course, and the datasets are provided by that course. The notebook is my own
> working through the course material, not an independent case study.

## Business question

A lender wants to approve as many good loans as possible while limiting losses from defaults.
This project asks:

1. How likely is each applicant to default?
2. How should the approval threshold be chosen, and what does it cost or earn?

## Dataset

Loan records from the DataCamp course (`datasets/`):

- About 32,500 loans in the raw file, 12 columns, with roughly 22% labelled as default (`loan_status = 1`)
- Features include applicant age, income, home ownership, employment length, loan intent, loan grade, loan amount, interest rate, loan-to-income ratio, and credit history length
- Intermediate files (`cr_loan_no_out.csv`, `cr_loan_nout_nmiss.csv`, `cr_loan_w2.csv`, ...) are the course's cleaned versions of the same data at different stages

## What the notebook covers

`script.ipynb` is organised in the same order as the workflow:

| Step | What I did |
|---|---|
| **1. Explore and clean** | Crosstabs and plots; removed outliers (e.g. age > 100, employment length > 60 years); handled missing values (median imputation, dropping rows, filling a category as "Other"); one-hot encoded categorical columns |
| **2. Logistic Regression** | Single and multivariate models, train/test split, feature scaling, predicted probability of default |
| **3. Evaluate** | Classification report (precision/recall), ROC curve and AUC, confusion matrices |
| **4. Threshold analysis** | Compared thresholds (0.5 vs 0.4 and a sweep from 0.2 to 0.65) to see the trade-off between default recall and accuracy |
| **5. Gradient boosting (XGBoost)** | Trained XGBoost, checked feature importance, compared column subsets, cross-validated with AUC |
| **6. Class imbalance** | Undersampled the majority class and compared results before and after |
| **7. Compare models** | Logistic Regression vs XGBoost using ROC/AUC and calibration curves |
| **8. Approval strategy** | Chose thresholds from target acceptance rates, built a strategy table (acceptance rate, bad rate, estimated value), and computed expected loss = PD x LGD x loan amount |

## Key takeaways

- Accuracy alone is misleading when defaults are the minority class. Recall on defaults and AUC matter more.
- The probability threshold is a business decision: lowering it catches more defaults but rejects more good borrowers.
- A good model is not only about prediction quality. Linking predictions to acceptance rate, bad rate, and expected loss is what makes the output usable by a lender.

## Tools

Python, pandas, matplotlib, scikit-learn, XGBoost (dependencies managed with Poetry).

## Project structure

```
.
├── script.ipynb        # main notebook
├── datasets/           # course datasets (raw and cleaned stages)
├── pyproject.toml      # dependencies
└── poetry.lock
```

## Run it

```bash
poetry install
poetry run jupyter lab   # or open script.ipynb in VS Code
```

## Limitations

- Built from course exercises, so data cleaning and feature choices follow the course rather than my own investigation
- Thresholds and strategy tables are evaluated on the same test set used for model comparison, with no separate validation set
- Loss given default is assumed (20% or 100% depending on the exercise), not estimated from data
- No hyperparameter tuning beyond the course examples
