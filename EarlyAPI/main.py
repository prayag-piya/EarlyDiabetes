from fastapi import FastAPI, HTTPException, BackgroundTasks
import requests
import base64
import uuid
import json
import tensorflow as tf
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO


from MultiHead.MultiHeadAttention import MultiHeadAttentionFCN
from MultiHead.TransferLearning import TransferAttention

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

smote = SMOTE(random_state=42)


# Define Pydantic models
class LimeExplanation(BaseModel):
    feature: str
    importance: float

class PredictionRequest(BaseModel):
    Age: int
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str


class DiabetesExplanationRequest(BaseModel):
    lime_explanation: List[LimeExplanation]


def sample_dataset(X, y):
  X_resampled, y_resampled = smote.fit_resample(X, y)
  return X_resampled, y_resampled

def categorical_age(age: int) -> int:
    age_ranges = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
                  (45, 49), (50, 54), (55, 59), (60, 64), (65, 69),
                  (70, 74), (75, 79), (80, float("inf"))]

    for idx, (start, end) in enumerate(age_ranges, start=1):
        if start <= age <= end:
            return idx
    return 0


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("models/Early_Diabetes_symptom.keras", custom_objects={"TransferAttention": TransferAttention, "MultiHeadAttention": MultiHeadAttentionFCN})
print(model.summary())

symptom = pd.read_csv("dataset/diabetes_data_upload.csv")
std = StandardScaler()
le = LabelEncoder()

symptom["class"] = symptom["class"].map({"Positive": 1, "Negative": 0})
symptom["Age"] = symptom["Age"].apply(categorical_age)
x3, y3 = symptom.drop("class", axis=1), symptom["class"]
for i in x3.columns:
    x3[i] = le.fit_transform(x3[i])
x3, y3 = sample_dataset(x3, y3)
x3 = std.fit_transform(x3)
X_train3, X_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.2, stratify=y3, random_state=42)

explainer = LimeTabularExplainer(
    training_data=X_train3[0:200],
    feature_names=list(symptom.columns[:-1]),
    class_names=["No Diabetes", "Diabetes"],
    mode="classification",
    discretize_continuous=True
)

def predict_fn(X):
    preds = model.predict(X)
    preds = np.hstack([1 - preds, preds])
    return preds


class OllamaAgent:
    def __init__(self, model="llama2"):
        
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.context = "" 

    def chat(self, user_input):
        """Sends a message to Ollama and returns a response."""
        self.context += f"\nUser: {user_input}\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": self.context,
            "stream": False
        }

        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            ai_reply = response.json().get("response", "")
            self.context += f" {ai_reply}\n"
            return ai_reply
        else:
            return f"Error: {response.status_code}, {response.text}"


class DiabetesExplanationAgent(OllamaAgent):
    def __init__(self, model="llama2"):
        super().__init__(model)
        self.explanation_context = "You are an expert doctor who analyzes patient health conditions based on model explanations."

    def explain_diabetes_prediction(self, lime_explanation, prediction_output):
        """Sends the LIME explanation to Ollama for analysis."""
        explanation_text = self.format_lime_explanation(lime_explanation)
        user_input = (
            f"Here is the patient's diabetes prediction explanation: {explanation_text}.\n\n"
            f"The model predicted: {prediction_output}.\n\n"
            "Can you analyze these results and explain why the model might have made this decision?"
        )

        self.context = self.explanation_context
        response = self.chat(user_input)
        return response


    def format_lime_explanation(self, lime_explanation):
        """Formats the LIME explanation into a human-readable paragraph."""
        explanation = "Based on the model's analysis, several factors contribute to the patient's diabetes prediction:\n"

        for item in lime_explanation:
            feature, importance = item

            if "<=" in feature:
                feature_name, condition = feature.split(" <= ")
            elif ">=" in feature:
                feature_name, condition = feature.split(" >= ")
            else:
                feature_name = feature
                condition = feature

            explanation += f"- The presence of {feature_name} ({condition}) was found to be {'important' if importance > 0 else 'less important'} with a score of {abs(importance):.2f}.\n"

        explanation += "\nBased on these factors, the patient is at a higher risk of having diabetes."
        return explanation


results_cache = {}

@app.post("/predict/")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        task_id = str(uuid.uuid4())

        input_data = np.array([
            categorical_age(request.Age),
            1 if request.Gender.lower() == "male" else 0,
            1 if request.Polyuria.lower() == "yes" else 0,
            1 if request.Polydipsia.lower() == "yes" else 0,
            1 if request.sudden_weight_loss.lower() == "yes" else 0,
            1 if request.weakness.lower() == "yes" else 0,
            1 if request.Polyphagia.lower() == "yes" else 0,
            1 if request.Genital_thrush.lower() == "yes" else 0,
            1 if request.visual_blurring.lower() == "yes" else 0,
            1 if request.Itching.lower() == "yes" else 0,
            1 if request.Irritability.lower() == "yes" else 0,
            1 if request.delayed_healing.lower() == "yes" else 0,
            1 if request.partial_paresis.lower() == "yes" else 0,
            1 if request.muscle_stiffness.lower() == "yes" else 0,
            1 if request.Alopecia.lower() == "yes" else 0,
            1 if request.Obesity.lower() == "yes" else 0
        ])
        input_data = pd.DataFrame(input_data)
        normalized_input_data = std.fit_transform(input_data)
        prediction = model.predict(normalized_input_data.reshape(1, -1))
        print(prediction)
        predicted_class = "Diabetes" if prediction[0][0] > 0.5 else "No Diabetic"
        print(predicted_class)
        lime_explanation = explainer.explain_instance(input_data[0], predict_fn, num_features=20)

        background_tasks.add_task(generate_llm_explanation, task_id, lime_explanation, predicted_class)

        return {"prediction": predicted_class, "task_id": task_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_llm_explanation(task_id, lime_explanation, prediction):
    """Handle LLM explanation generation in the background."""
    agent = DiabetesExplanationAgent(model="llama2")
    explanation = agent.explain_diabetes_prediction(lime_explanation.as_list(), prediction)
    fig = lime_explanation.as_pyplot_figure()
    fig.set_size_inches(8, 6)
    plt.tight_layout()
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
    results_cache[task_id] = {
        "explanation": explanation,
        "explanation_image": img_base64
    }

def get_task_result(task_id: str):
    """Fetch the result for a given task ID."""
    return results_cache.get(task_id)


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Fetch the result of the background task using the task ID."""
    result = get_task_result(task_id)
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail="Task not found or still processing")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)