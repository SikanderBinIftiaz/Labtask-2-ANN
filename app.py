import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("framingham.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.drop_duplicates().dropna().reset_index(drop=True)

target = "TenYearCHD"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

X = df.drop(columns=[target, "id"], errors="ignore")
y = df[target].astype(int)
X = pd.get_dummies(X, drop_first=False)

if y.nunique() != 2:
    raise ValueError("TenYearCHD must contain both 0 and 1 classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
report = classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["No CHD", "CHD"], zero_division=0
)

feature_columns = X.columns.tolist()
original_features = df.drop(columns=[target, "id"], errors="ignore")
numeric_features = original_features.select_dtypes(include="number").columns.tolist()
categorical_features = original_features.select_dtypes(exclude="number").columns.tolist()

def predict_chd(*values):
    applicant = pd.DataFrame(
        [values],
        columns=original_features.columns
    )
    applicant = pd.get_dummies(applicant, drop_first=False)
    applicant = applicant.reindex(columns=feature_columns, fill_value=0)
    applicant = applicant.astype(float)

    prediction = int(model.predict(applicant)[0])
    probability = model.predict_proba(applicant)[0][1] * 100

    decision = "Higher predicted CHD risk" if prediction == 1 else "Lower predicted CHD risk"

    return (
        decision,
        f"{probability:.2f}%",
        f"{accuracy:.2f}%",
        pd.DataFrame(
            cm,
            index=["Actual No CHD", "Actual CHD"],
            columns=["Predicted No CHD", "Predicted CHD"]
        ),
        report
    )

css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=css,
    title="Framingham CHD Risk Prediction"
) as demo:

    gr.Markdown("""
    # ❤️ 10-Year CHD Risk Prediction
    ### Logistic Regression · Framingham Dataset

    Enter the applicant's health and demographic information to obtain a model prediction.
    """)

    inputs = []

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Applicant Information")

            for col in original_features.columns:
                if col in numeric_features:
                    default_value = float(original_features[col].median())
                    minimum = float(original_features[col].min())
                    maximum = float(original_features[col].max())

                    if minimum == maximum:
                        maximum = minimum + 1

                    component = gr.Number(
                        label=col,
                        value=default_value
                    )
                else:
                    options = sorted(original_features[col].astype(str).unique().tolist())
                    component = gr.Dropdown(
                        choices=options,
                        value=options[0] if options else None,
                        label=col
                    )
                inputs.append(component)

            predict_btn = gr.Button(
                "Predict 10-Year CHD Risk",
                variant="primary",
                size="lg"
            )

        with gr.Column():
            gr.Markdown("### Prediction Results")

            prediction_output = gr.Textbox(
                label="Predicted Risk Category",
                interactive=False
            )

            probability_output = gr.Textbox(
                label="Model-Estimated CHD Probability",
                interactive=False
            )

            accuracy_output = gr.Textbox(
                label="Test Set Accuracy",
                interactive=False
            )

            confusion_output = gr.Dataframe(
                label="Confusion Matrix",
                interactive=False
            )

    with gr.Accordion("Classification Report", open=False):
        report_output = gr.Textbox(
            label="Precision, Recall and F1-Score",
            lines=10,
            interactive=False
        )

    predict_btn.click(
        fn=predict_chd,
        inputs=inputs,
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
    **Disclaimer:** This application is for educational purposes only.
    It is not a medical diagnosis or a substitute for professional medical advice.
    Model-estimated probabilities may not reflect an individual's actual risk.
    """)

if __name__ == "__main__":
    print("Dataset records:", len(df))
    print("Features:", len(feature_columns))
    print(f"Test accuracy: {accuracy:.2f}%")
    demo.launch()