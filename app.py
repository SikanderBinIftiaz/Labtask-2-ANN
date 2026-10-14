import gradio as gr
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. LOAD AND VALIDATE DATASET
# =========================================================

df = pd.read_csv("dataset.csv")
df = df.drop_duplicates().dropna().reset_index(drop=True)

target_column = "loan_approved"

if target_column not in df.columns:
    raise ValueError(
        f"Dataset must contain a '{target_column}' column. "
        f"Available columns: {list(df.columns)}"
    )

feature_columns = [
    "age",
    "annual_income",
    "credit_score",
    "employment_years",
    "existing_debt",
    "savings"
]

missing_columns = [
    col for col in feature_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

X = df[feature_columns]
y = df[target_column]

if y.nunique() != 2:
    raise ValueError(
        "The loan_approved column must contain exactly two classes."
    )

# Support datasets where approval labels are 0/1 or Boolean.
# The positive class is identified as 1 or True.
classes = list(y.unique())

if 1 in classes:
    positive_class = 1
elif True in classes:
    positive_class = True
else:
    raise ValueError(
        "The loan_approved column must use 1 for approved "
        "and 0 for not approved."
    )

if len(df) < 10 or y.value_counts().min() < 2:
    raise ValueError(
        "Dataset is too small or one class has too few examples "
        "for a stratified train/test split."
    )

# =========================================================
# 2. TRAIN LOGISTIC REGRESSION MODEL
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(max_iter=1000, random_state=42)
    )
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

class_labels = list(model.named_steps["classifier"].classes_)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

report = classification_report(
    y_test,
    y_pred,
    labels=class_labels,
    target_names=[
        f"Not Approved ({class_labels[0]})",
        f"Approved ({class_labels[1]})"
    ],
    zero_division=0
)

# =========================================================
# 3. PREDICTION FUNCTION
# =========================================================

def predict_loan(
    age,
    annual_income,
    credit_score,
    employment_years,
    existing_debt,
    savings
):
    values = [
        age,
        annual_income,
        credit_score,
        employment_years,
        existing_debt,
        savings
    ]

    if any(value is None for value in values):
        return (
            "Please enter all applicant details.",
            "—",
            f"{accuracy:.2f}%",
            pd.DataFrame(),
            report
        )

    if any(value < 0 for value in values):
        return (
            "Values cannot be negative.",
            "—",
            f"{accuracy:.2f}%",
            pd.DataFrame(),
            report
        )

    if not 18 <= age <= 100:
        return (
            "Age must be between 18 and 100.",
            "—",
            f"{accuracy:.2f}%",
            pd.DataFrame(),
            report
        )

    if not 300 <= credit_score <= 850:
        return (
            "Credit score must be between 300 and 850.",
            "—",
            f"{accuracy:.2f}%",
            pd.DataFrame(),
            report
        )

    new_data = pd.DataFrame(
        [[
            age,
            annual_income,
            credit_score,
            employment_years,
            existing_debt,
            savings
        ]],
        columns=feature_columns
    )

    prediction = model.predict(new_data)[0]

    probabilities = model.predict_proba(new_data)[0]

    approval_index = class_labels.index(positive_class)

    approval_probability = probabilities[approval_index] * 100

    decision = (
        "LOAN APPROVED"
        if prediction == positive_class
        else "LOAN NOT APPROVED"
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {label}"
            for label in class_labels
        ],
        columns=[
            f"Predicted {label}"
            for label in class_labels
        ]
    )

    return (
        decision,
        f"{approval_probability:.2f}%",
        f"{accuracy:.2f}%",
        confusion_df,
        report
    )


# =========================================================
# 4. PROFESSIONAL GRADIO INTERFACE
# =========================================================

css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 20px;
}

.result-box {
    border-radius: 12px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=css,
    title="AI Loan Approval System"
) as demo:

    gr.Markdown("""
    # 🏦 AI Loan Approval Prediction System

    ### Logistic Regression · Machine Learning Dashboard

    Enter an applicant's financial details to generate a
    prediction from your trained machine-learning model.
    """)

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 👤 Applicant Information")

            age = gr.Slider(
                minimum=18,
                maximum=100,
                value=30,
                step=1,
                label="Age (years)"
            )

            annual_income = gr.Number(
                value=70000,
                minimum=0,
                label="Annual Income"
            )

            credit_score = gr.Slider(
                minimum=300,
                maximum=850,
                value=750,
                step=1,
                label="Credit Score"
            )

            employment_years = gr.Slider(
                minimum=0,
                maximum=50,
                value=5,
                step=0.5,
                label="Employment Experience (years)"
            )

            existing_debt = gr.Number(
                value=5000,
                minimum=0,
                label="Existing Debt"
            )

            savings = gr.Number(
                value=20000,
                minimum=0,
                label="Savings"
            )

            predict_btn = gr.Button(
                "🔍 Predict Loan Approval",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Results")

            prediction_output = gr.Textbox(
                label="Loan Decision",
                interactive=False,
                elem_classes=["result-box"]
            )

            probability_output = gr.Textbox(
                label="Probability of Approval",
                interactive=False
            )

            accuracy_output = gr.Textbox(
                label="Model Test Accuracy",
                interactive=False
            )

            confusion_output = gr.Dataframe(
                label="Confusion Matrix",
                interactive=False
            )

    with gr.Accordion(
        "Classification Report — Precision, Recall and F1-Score",
        open=False
    ):
        # FIX: Use Textbox instead of Code(language="text").
        report_output = gr.Textbox(
            label="Classification Report",
            lines=10,
            max_lines=20,
            interactive=False
        )

    predict_btn.click(
        fn=predict_loan,
        inputs=[
            age,
            annual_income,
            credit_score,
            employment_years,
            existing_debt,
            savings
        ],
        outputs=[
            prediction_output,
            probability_output,
            accuracy_output,
            confusion_output,
            report_output
        ]
    )

    gr.Markdown("""
    ---

    **Disclaimer:** This application is an educational demonstration.
    Model predictions are not suitable for real financial decisions.
    The displayed accuracy measures performance on the test split;
    it does not guarantee correct predictions for individual applicants.
    """)


# =========================================================
# 5. LAUNCH APPLICATION
# =========================================================

if __name__ == "__main__":
    print("Starting AI Loan Approval System...")
    print(f"Dataset records: {len(df)}")
    print(f"Test accuracy: {accuracy:.2f}%")

    demo.launch() 