from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load(os.path.join("..", "model", "toxic_model.pkl"))
vectorizer = joblib.load(os.path.join("..", "model", "vectorizer.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    comment = ""
    
    if request.method == "POST":
        comment = request.form["comment"]
        vect_comment = vectorizer.transform([comment])
        result = model.predict(vect_comment)[0]
        prediction = "🔥 TOXIC" if result == 1 else "✅ SAFE"
        
    return render_template("index.html", prediction=prediction, comment=comment)

if __name__ == "__main__":
    app.run(debug=True)
