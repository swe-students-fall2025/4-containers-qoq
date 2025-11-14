"""Basic Flask web app Setup"""
from flask import Flask, render_template

app = Flask(__name__)

def index():
    """index page."""
    return render_template('index.html')

def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')
