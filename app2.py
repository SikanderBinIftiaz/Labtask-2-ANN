import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("student_mental_health_burnout.csv")
df = df.drop_duplicates().dropna().reset_index(drop=True)

target_column = "burnout_level" if "burnout_level" in df.columns else df.columns[-1]
X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=False)
encoder = LabelEncoder()
y = encoder.fit_transform(df[target_column].astype(str))

if len(encoder.classes_) < 2:
    raise ValueError("The target column must contain at least two classes.")

counts = pd.Series(y).value_counts()
stratify = y if counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred, labels=range(len(encoder.classes_)))
report = classification_report(
    y_test, y_pred, labels=range(len(encoder.classes_)),
    target_names=encoder.classes_, zero_division=0
)

numeric_columns = df.drop(columns=[target_column]).select_dtypes(include="number").columns.tolist()
categorical_columns = df.drop(columns=[target_column]).select_dtypes(exclude="number").columns.tolist()

def predict_burnout(*values):
    row = {}
    for col, value in zip(df.drop(columns=[target_column]).columns, values):
        row[col] = value
    applicant = pd.DataFrame([row])
    applicant = pd.get_dummies(applicant, drop_first=False)
    applicant = applicant.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(applicant)[0]
    probabilities = model.predict_proba(applicant)[0]
    confidence = probabilities[prediction] * 100
    decision = encoder.inverse_transform([prediction])[0]
    return str(decision), f"{confidence:.2f}%", f"{accuracy:.2f}%", pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in encoder.classes_],
        columns=[f"Predicted {c}" for c in encoder.classes_]
    ), report

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
    title="Student Mental Health & Burnout AI"
) as demo:
    gr.Markdown("""
    # 🧠 Student Mental Health & Burnout Prediction
    ### Logistic Regression · Machine Learning Dashboard
    Enter student information to predict the dataset's target category.
    """)

    inputs = []

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Student Information")
            for col in df.drop(columns=[target_column]).columns:
                if col in numeric_columns:
                    component = gr.Number(
                        label=col.replace("_", " ").title(),
                        value=float(df[col].median())
                    )
                else:
                    options = sorted(df[col].astype(str).unique().tolist())
                    component = gr.Dropdown(
                        choices=options,
                        value=options[0] if options else None,
                        label=col.replace("_", " ").title()
                    )
                inputs.append(component)

            predict_btn = gr.Button(
                "Predict Burnout Category",
                variant="primary",
                size="lg"
            )

        with gr.Column():
            gr.Markdown("### Prediction Results")
            prediction_output = gr.Textbox(label="Predicted Category", interactive=False)
            confidence_output = gr.Textbox(label="Prediction Confidence", interactive=False)
            accuracy_output = gr.Textbox(label="Test Accuracy", interactive=False)
            confusion_output = gr.Dataframe(label="Confusion Matrix", interactive=False)

    with gr.Accordion("Classification Report", open=False):
        report_output = gr.Textbox(label="Precision, Recall and F1-Score", lines=12, interactive=False)

    predict_btn.click(
        fn=predict_burnout,
        inputs=inputs,
        outputs=[
            prediction_output,
            confidence_output,
            accuracy_output,
            confusion_output,
            report_output
        ]
    )

    gr.Markdown("""
    ---
    **Disclaimer:** This is an educational machine-learning demonstration.
    Predictions are not a medical diagnosis or a substitute for professional support.
    """)

if __name__ == "__main__":
    print("Dataset:", len(df), "records")
    print("Target column:", target_column)
    print("Classes:", list(encoder.classes_))
    print(f"Test accuracy: {accuracy:.2f}%")
    demo.launch()