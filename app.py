# app.py (Complete and Corrected)

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_required, current_user, login_user, logout_user

# --- Import Custom Modules ---
import ai_core
import rag_core
import database

# --- App Initialization ---
app = Flask(__name__)
app.secret_key = 'a_very_secret_key'

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dermascan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Initialize Database and Login Manager ---
database.init_app(app)

# --- Import models and objects from the database module ---
User = database.User
Scan = database.Scan
db = database.db
login_manager = database.login_manager
CLASS_FULL_NAMES = database.CLASS_FULL_NAMES # Import for use in show_results


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/scan')
@login_required
def scan():
    return render_template('scan.html')

# In app.py, replace the analyze_image function with this:

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = os.path.join('static/uploads', file.filename)
        file.save(filename)
        
        # --- Normalize path to use forward slashes for web URLs ---
        filename = filename.replace('\\', '/')
        
        # --- Call AI Core ---
        risk_class, confidence, gradcam_paths, full_class_name, simple_risk = ai_core.predict_and_explain(filename)

        if risk_class is None:
            return jsonify({'error': 'Analysis failed. Please try again.'}), 500

        # --- FIX: Join the list of Grad-CAM paths into a single string ---
        gradcam_path_str = ','.join(gradcam_paths)
        
        # --- FIX: Also normalize the individual paths in the list before joining ---
        normalized_paths = [p.replace('\\', '/') for p in gradcam_paths]
        gradcam_path_str = ','.join(normalized_paths)

        # --- Save to Database ---
        new_scan = Scan(
            user_id=current_user.id,
            image_path=filename,
            gradcam_path=gradcam_path_str, # Save the corrected string
            risk_level=risk_class, 
            full_class_name=full_class_name,
            simple_risk=simple_risk,
            confidence=confidence
        )
        db.session.add(new_scan)
        db.session.commit()
        
        print(f"[DEBUG] Saved to database. Scan ID: {new_scan.id}, Image Path: {new_scan.image_path}")

        return redirect(url_for('show_results', scan_id=new_scan.id))
    
@app.route('/results/<int:scan_id>')
@login_required
def show_results(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        flash('You are not authorized to view this scan.')
        return redirect(url_for('index'))
        
    # --- Get RAG Recommendation ---
    recommendation = rag_core.get_recommendation(scan.simple_risk)

    # Pass the scan object which now contains all the necessary data
    return render_template('results.html', scan=scan, recommendation=recommendation)

@app.route('/history')
@login_required
def history():
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.id.desc()).all()
    return render_template('history.html', scans=user_scans)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'Query is required'}), 400
        
    ai_response = rag_core.chat_with_ai(user_query)
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    app.run(debug=True)