import sqlite3
import bcrypt
from src.v1.config import DB_FILE

# init database
def init_db():
    """Run this once when your application starts up."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                hashed_password TEXT NOT NULL
            )
        """)

# logic for registering the user in the database
def register_user(username, password):
    # saving the user, pass only after encoding them
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (username, hashed)
            )
        return True
    except sqlite3.IntegrityError:
        return False

# verify the user is in database
def verify_user(username, password):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
    if row:
        stored_hash = row[0].encode('utf-8')
        password_bytes = password.encode('utf-8')
        
        return bcrypt.checkpw(password_bytes, stored_hash)
    
    return False