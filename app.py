from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(_name_)

@app.route("/")
def home():
    return "Loan Approval Prediction System"

if _name_ == "_main_":
    app.run(debug=True)
