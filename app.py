from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
import os

# Facultatif : pour télécharger les fichiers .dat si manquants
try:
    import gdown
except ImportError:
    gdown = None  # Ne bloque pas l'exécution si absent

app = Flask(__name__)

def download_model_files():
    if not gdown:
        print("gdown n'est pas installé. Téléchargement ignoré.")
        return

    if not os.path.exists('models'):
        os.makedirs('models')

    model_files = {
        "shape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=1D7WzzJ1c9S6pUay6cl3JHYvR-yrMVv8R",  # Exemple d'ID
        "dlib_face_recognition_resnet_model_v1.dat": "https://drive.google.com/uc?id=1yA1vVVW_km7S5yo7p0vHyvVxHRhzUHQe"  # Exemple d'ID
    }

    for filename, url in model_files.items():
        filepath = os.path.join('models', filename)
        if not os.path.exists(filepath):
            print(f"Téléchargement de {filename}...")
            gdown.download(url, filepath, quiet=False)

def get_attendance_by_date(date_str):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (date_str,))
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    if not selected_date:
        return render_template('index.html', selected_date='', no_data=True)

    try:
        selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
        formatted_date = selected_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return render_template('index.html', selected_date=selected_date, no_data=True)

    attendance_data = get_attendance_by_date(formatted_date)

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)

    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

if __name__ == '__main__':
    # Décommenter cette ligne si tu veux télécharger automatiquement les modèles .dat
    # download_model_files()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
