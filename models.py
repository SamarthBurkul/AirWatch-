from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)  # ✅ FIXED: Increased from 128 to 255
    preferred_city = db.Column(db.String(100), default='Delhi')

    def set_password(self, password):
        """Hash and store password securely."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify password against stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'


class Tip(db.Model):
    __tablename__ = 'tip'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), index=True)  # Added index for faster queries
    difficulty = db.Column(db.String(50))
    impact = db.Column(db.String(50))
    pollutants_targeted = db.Column(db.String(200))
    related_diseases = db.Column(db.String(200))

    def __repr__(self):
        return f'<Tip {self.title}>'


class Favorite(db.Model):
    __tablename__ = 'favorite'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    city = db.Column(db.String(100), nullable=False)
    
    # Add unique constraint to prevent duplicate favorites
    __table_args__ = (
        db.UniqueConstraint('user_id', 'city', name='unique_user_city'),
    )

    def __repr__(self):
        return f'<Favorite {self.city} for User {self.user_id}>'