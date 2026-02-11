import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 10  # safe fixed length

# Color dictionary
colors = {
    "red": "लाल",
    "blue": "निळा",
    "green": "हिरवा",
    "yellow": "पिवळा",
    "black": "काळा",
    "white": "पांढरा",
    "pink": "गुलाबी",
    "orange": "नारिंगी"
}

label_map = ["favourite_color", "goodbye", "greeting"]

def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    return label_map[np.argmax(pred)]

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"].lower()
    intent = predict_intent(user_input)

    if intent == "greeting":
        return jsonify({"response": "Hello! How can I help you?"})

    elif intent == "favourite_color":
        detected_color = None
        for color in colors:
            if color in user_input:
                detected_color = color
                break

        if detected_color:
            marathi_sentence = f"माझा आवडता रंग {colors[detected_color]} आहे"
            reply = f"Wow that's nice to hear!\nIn Marathi you can say:\n{marathi_sentence}"
        else:
            reply = "That's great! Which color do you like?"

        return jsonify({"response": reply})

    elif intent == "goodbye":
        return jsonify({"response": "Goodbye! Have a nice day!"})

    return jsonify({"response": "I didn't understand."})

if __name__ == "__main__":
    app.run(debug=True)
