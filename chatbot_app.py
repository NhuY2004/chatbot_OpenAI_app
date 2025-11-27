import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv("openAI.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if (OPENAI_API_KEY is not None):
    client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_PATH = os.path.join("models", "model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
DATA_PATH = os.path.join("data", "titanic.csv")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load dataset for analysis
data_df = pd.read_csv(DATA_PATH)

app = Flask(__name__)

#TASK 3 – FUNCTION CALLING
def predict_survival(age: float, sex: str, pclass: int, fare: float) -> str:
    sex_val = 0 if sex.lower() == "male" else 1
    X = np.array([[pclass, sex_val, age, fare]], dtype=float)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    if pred == 1:
        msg = f"The model predicts this passenger would survive with probability {prob:.2f}."
    else:
        msg = f"The model predicts this passenger would not survive with probability {1 - prob:.2f}."

    return msg

def simple_data_analysis() -> str:
    desc = data_df[["Age", "Fare"]].describe().to_dict()
    survived_rate = data_df["Survived"].mean()

    text = (
        f"In this Titanic dataset, the overall survival rate is about "
        f"{survived_rate:.2f}. "
        f"The average passenger age is {desc['Age']['mean']:.2f} years old "
        f"and the average ticket fare is {desc['Fare']['mean']:.2f}. "
        f"Pclass and Sex are strong indicators for survival."
    )
    return text

#NLU LAYER — Detect if user wants ML function

def handle_function_calling(user_message: str) -> str | None:
    text = user_message.lower()

    # EXACT KEYWORDS YOU REQUESTED
    if "dataset summary" in text or "return the dataset summary" in text:
        return simple_data_analysis()

    # BUSINESS QUESTION SAMPLE:
    if "strategic planning" in text:
        return "Strategic planning is the long term process where an organization defines its goals, direction, and resource allocation to achieve competitive advantage."

    if "differentiation strategy" in text:
        return "Differentiation strategy is when a company offers unique features or value that makes its product or service stand out from competitors."

    # Generic triggers
    if "data analysis" in text or "summary of titanic" in text:
        return simple_data_analysis()

    if "predict survival" in text:
        try:
            tokens = text.split()
            age = float(tokens[tokens.index("age") + 1])
            sex = tokens[tokens.index("sex") + 1]
            pclass = int(tokens[tokens.index("pclass") + 1])
            fare = float(tokens[tokens.index("fare") + 1])
            return predict_survival(age, sex, pclass, fare)
        except:
            return (
                "Prediction failed. Use format: "
                "predict survival age 30 sex female pclass 1 fare 80"
            )

    return None  # not ML request


#GPT LAYER — Answer ANY question
def llm_chat(user_message: str) -> str:
    """
    Online mode using OpenAI API
    """
    global client
    if client is None:
        return "OpenAI API key not found. Cannot use GPT mode."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant for managers. Answer clearly and professionally."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling OpenAI API: {e}"
#FLASK API

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_message = data.get("message", "")

    # 1. check ML tools
    func_answer = handle_function_calling(user_message)
    if func_answer is not None:
        return jsonify({"reply": func_answer})

    # 2. else ask GPT
    reply = llm_chat(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
