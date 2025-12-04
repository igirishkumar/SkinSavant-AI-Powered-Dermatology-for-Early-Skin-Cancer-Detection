# database.py

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager

# --- Constants from your model loader ---
CLASS_NAMES = [
    'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
]
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease', 'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis', 'df': 'Dermatofibroma',
    'mel': 'Melanoma', 'nv': 'Melanocytic Nevi (Mole)', 'vasc': 'Vascular Lesions'
}

# --- Initialize Extensions ---
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'login'

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True)
    def __repr__(self): return f'<User {self.username}>'

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    gradcam_path = db.Column(db.Text, nullable=True)
    risk_level = db.Column(db.String(50), nullable=False) # e.g., 'mel'
    full_class_name = db.Column(db.String(200), nullable=False) # e.g., 'Melanoma'
    simple_risk = db.Column(db.String(50), nullable=False) # e.g., 'high'
    confidence = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    def __repr__(self): return f'<Scan {self.id} for User {self.user_id}>'

# --- User Loader for Flask-Login ---
@login_manager.user_loader
def load_user(user_id): return User.query.get(int(user_id))

# --- App Initialization Function ---
def init_app(app):
    db.init_app(app)
    login_manager.init_app(app)
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='testuser').first():
            dummy_user = User(username='testuser', password='password123')
            db.session.add(dummy_user)
            db.session.commit()
            print("Created dummy user: 'testuser' with password 'password123'")