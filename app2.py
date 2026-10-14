
from pathlib import Path

import gradio as gr
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "MAGIC Gamma Telescope.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}\n"
        "Place MAGIC Gamma Telescope.csv in the same folder as app.py."
    )

df = pd.read_csv(CSV_PATH, sep=None, engine="python")

# Clean column names and remove an unnecessary saved index column.
df.columns = df.columns.astype(str).str.strip()
df = df.loc[
    :,
    ~df.columns.str.match(r"^Unnamed", case=False)
]

# Find the target column, ignoring capitalization.
target_matches = [
    column for column in df.columns
    if column.strip().lower() == "class"
]

if not target_matches:
    raise ValueError(
        "The CSV must contain a target column named 'class'. "
        "Check the column names in your CSV file."
    )

TARGET = target_matches[0]

# Remove empty rows and duplicate rows.
df = df.dropna().drop_duplicates().reset_index(drop=True)

# Prepare features and target.
X = df.drop(columns=[TARGET]).copy()
y_text = df[TARGET].astype(str).str.strip().str.lower()

# Convert all feature columns to numeric values.
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows containing invalid feature values.
valid_rows = X.notna().all(axis=1) & y_text.notna()
X = X.loc[valid_rows].reset_index(drop=True)
y_text = y_text.loc[valid_rows].reset_index(drop=True)

if X.empty:
    raise ValueError("No usable rows were found in the CSV file.")

if y_text.nunique() < 2:
    raise ValueError("The class column must contain at least two classes.")

# Encode labels, such as g and h.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

if len(label_encoder.classes_) != 2:
    raise ValueError(
        "This app expects two classes, normally 'g' and 'h'. "
        f"Found: {list(label_encoder.classes_)}"
    )

if pd.Series(y).value_counts().min() < 2:
    raise ValueError(
        "Each class needs at least two rows to create a stratified train/test split."
    )

# --------------------------------------------------
# 2. SPLIT DATA AND TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)

# --------------------------------------------------
# 3. EVALUATE THE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
class_names = list(label_encoder.classes_)

def friendly_class(label):
    """Convert dataset labels into readable class names."""
    label = str(label).lower()

    if label == "g":
        return "Gamma (g)"
    if label == "h":
        return "Hadron (h)"

    return label.upper()

display_names = [friendly_class(name) for name in class_names]

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
)

matrix_df = pd.DataFrame(
    matrix,
    index=[f"Actual {name}" for name in display_names],
    columns=[f"Predicted {name}" for name in display_names],
)

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=display_names,
    zero_division=0,
)

metrics_text = (
    f"LOGISTIC REGRESSION MODEL RESULTS\n"
    f"{'=' * 38}\n"
    f"Dataset: MAGIC Gamma Telescope.csv\n"
    f"Usable rows: {len(X)}\n"
    f"Features: {X.shape[1]}\n"
    f"Training samples: {len(X_train)}\n"
    f"Testing samples: {len(X_test)}\n"
    f"Test accuracy: {accuracy:.2f}%\n\n"
    f"CONFUSION MATRIX\n"
    f"{matrix_df.to_string()}\n\n"
    f"CLASSIFICATION REPORT\n"
    f"{report}"
)

# --------------------------------------------------
# 4. CREATE GRADIO PREDICTION FUNCTION
# --------------------------------------------------

# Use training-set medians as sensible starting values.
defaults = X_train.median()

def predict_telescope_class(*values):
    try:
        if any(value is None for value in values):
            return (
                "Please enter a value for every feature.",
                "",
            )

        observation = pd.DataFrame(
            [values],
            columns=X.columns,
        )

        prediction_encoded = int(model.predict(observation)[0])
        probabilities = model.predict_proba(observation)[0]

        predicted_label = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        predicted_name = friendly_class(predicted_label)

        probability_by_label = {
            label_encoder.inverse_transform([index])[0]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        gamma_probability = probability_by_label.get("g", 0.0) * 100
        hadron_probability = probability_by_label.get("h", 0.0) * 100

        result = (
            f"Predicted Class: {predicted_name}\n\n"
            f"Gamma (g) probability: {gamma_probability:.2f}%\n"
            f"Hadron (h) probability: {hadron_probability:.2f}%\n\n"
            "Probabilities are estimates from the trained model."
        )

        return result, metrics_text

    except Exception as error:
        return f"Prediction error: {error}", metrics_text

# --------------------------------------------------
# 5. BUILD GRADIO USER INTERFACE
# --------------------------------------------------

input_components = []

for column in X.columns:
    default_value = float(defaults[column])

    input_components.append(
        gr.Number(
            label=column,
            value=default_value,
            precision=6,
        )
    )

with gr.Blocks(theme=gr.themes.Soft(), title="MAGIC Telescope AI Classifier") as demo:

    gr.Markdown(
        """
        # MAGIC Gamma Telescope Classifier
        ### Supervised Machine Learning — Logistic Regression

        Enter the telescope measurement values to classify an observation
        as **Gamma (g)** or **Hadron (h)**.

        The initial input values are feature medians from the training data.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Telescope Measurements")

            inputs_per_row = 2
            for start in range(0, len(input_components), inputs_per_row):
                with gr.Row():
                    for component in input_components[
                        start:start + inputs_per_row
                    ]:
                        component.render()

            predict_button = gr.Button(
                "Predict Class",
                variant="primary",
                size="lg",
            )

            clear_button = gr.Button("Reset to Default Values")

        with gr.Column(scale=1):
            gr.Markdown("### Prediction Result")

            prediction_output = gr.Textbox(
                label="Classification",
                lines=7,
                interactive=False,
            )

            gr.Markdown("### Model Evaluation")

            evaluation_output = gr.Textbox(
                label="Accuracy, Confusion Matrix and Classification Report",
                value=metrics_text,
                lines=24,
                interactive=False,
            )

    predict_button.click(
        fn=predict_telescope_class,
        inputs=input_components,
        outputs=[prediction_output, evaluation_output],
    )

    clear_button.click(
        fn=lambda: [float(defaults[column]) for column in X.columns],
        inputs=[],
        outputs=input_components,
    )

    gr.Markdown(
        """
        ---
        **Note:** This is an educational machine-learning demonstration.
        Predicted probabilities are model estimates, not guaranteed certainty.
        """
    )

# --------------------------------------------------
# 6. LAUNCH APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Dataset loaded: {CSV_PATH.name}")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Test accuracy: {accuracy:.2f}%")
    demo.launch()
