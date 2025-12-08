# app.py (Complete and Corrected)

# Load environment variables first
from dotenv import load_dotenv
import os

load_dotenv()

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OpenAI API Key loaded: {api_key[:10]}...")
else:
    print("❌ OpenAI API Key NOT loaded! Check your .env file")

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from openai import OpenAI

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
CLASS_FULL_NAMES = database.CLASS_FULL_NAMES

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# --- Routes ---
    
@app.route('/')
@login_required
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

        # --- Normalize the individual paths in the list before joining ---
        normalized_paths = [p.replace('\\', '/') for p in gradcam_paths]
        gradcam_path_str = ','.join(normalized_paths)

        # --- Save to Database ---
        new_scan = Scan(
            user_id=current_user.id,
            image_path=filename,
            gradcam_path=gradcam_path_str,
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

# --- OpenAI Chat Route ---
@app.route('/api/chat', methods=['POST'])
def openai_chat():
    """
    Handle chat messages and communicate with OpenAI API
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('conversation_history', [])
        scan_id = data.get('result_id')  # This comes as 'result_id' from frontend

        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400

        # Add user message to conversation history
        conversation_history.append({
            'role': 'user',
            'content': user_message
        })

        # Call OpenAI API
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',  # Use 'gpt-4' for better quality (more expensive)
            messages=conversation_history,
            temperature=0.7,
            max_tokens=500
        )

        # Extract AI response
        ai_message = response.choices[0].message.content

        # Add AI response to conversation history
        conversation_history.append({
            'role': 'assistant',
            'content': ai_message
        })

        # Optional: Save chat to database
        if current_user.is_authenticated and scan_id:
            save_chat_to_database(current_user.id, scan_id, user_message, ai_message)

        return jsonify({
            'success': True,
            'response': ai_message,
            'conversation_history': conversation_history
        })

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def save_chat_to_database(user_id, scan_id, user_message, ai_message):
    """
    Optional: Save chat history to database
    Create a ChatMessage model if you want to persist chat history
    """
    try:
        # Example if you want to add a ChatMessage model:
        # chat_message = ChatMessage(
        #     user_id=user_id,
        #     scan_id=scan_id,
        #     user_message=user_message,
        #     ai_response=ai_message,
        #     timestamp=datetime.utcnow()
        # )
        # db.session.add(chat_message)
        # db.session.commit()
        pass
    except Exception as e:
        print(f"Error saving chat: {str(e)}")


@app.route('/download-report/<int:scan_id>')
@login_required
def download_report(scan_id):
    """
    Generate and download PDF report
    """
    scan = Scan.query.get_or_404(scan_id)
    
    # Check if user owns this scan
    if scan.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('index'))
    
    # TODO: Implement PDF generation
    # You can use libraries like ReportLab or WeasyPrint
    flash('PDF generation coming soon!', 'info')
    return redirect(url_for('show_results', scan_id=scan_id))

@app.route('/delete-scan/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    
    # Check if user owns this scan
    if scan.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('history'))
    
    # Delete associated files
    try:
        if scan.image_path and os.path.exists(scan.image_path):
            os.remove(scan.image_path)
        
        if scan.gradcam_path:
            for gradcam in scan.gradcam_path.split(','):
                if os.path.exists(gradcam):
                    os.remove(gradcam)
    except Exception as e:
        print(f"Error deleting files: {e}")
    
    # Delete from database
    db.session.delete(scan)
    db.session.commit()
    
    flash('Scan deleted successfully', 'success')
    return redirect(url_for('history'))


if __name__ == '__main__':
    app.run(debug=True)