import bcrypt
import jwt
import mysql.connector
import streamlit as st
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth, credentials, initialize_app
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase
@st.cache_resource

def init_firebase():
    load_dotenv()  # Load environment variables from .env file
    
    # Check if app is already initialized
    if not firebase_admin._apps:
        # Use environment variables instead of the JSON file
        cred = credentials.Certificate({
            "type": os.getenv("FIREBASE_TYPE"),
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
            "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
            "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
        })
        firebase_admin.initialize_app(cred)

# MySQL connection with error handling
def get_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="rounak",
            password="rounakbag24",
            database="RounakDB"
        )
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        st.error(f"Database connection error: {err}")
        return None

# Check if the users_auth table exists, create if not
def ensure_users_table():
    conn = get_db()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'users_auth'")
        if not cursor.fetchone():
            # Create users_auth table
            cursor.execute("""
                CREATE TABLE users_auth (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(255) NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("users_auth table created successfully")
        return True
    except mysql.connector.Error as err:
        logger.error(f"Error creating users_auth table: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

# Local user registration with detailed error handling
def register_user(email, username, password):
    # Ensure users_auth table exists
    if not ensure_users_table():
        return False
    
    conn = get_db()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        # Check if user already exists
        cursor.execute("SELECT id FROM users_auth WHERE email = %s", (email,))
        if cursor.fetchone():
            logger.warning(f"User with email {email} already exists")
            st.warning("A user with this email already exists")
            return False
        
        # Hash password
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users_auth (email, username, password_hash) VALUES (%s, %s, %s)",
            (email, username, hashed)
        )
        conn.commit()
        logger.info(f"User {email} registered successfully")
        return True
    except mysql.connector.Error as err:
        logger.error(f"Database error during registration: {err}")
        st.error(f"Registration failed: {err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        st.error(f"Registration failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# Local login with improved error handling
def verify_user(email, password):
    conn = get_db()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash FROM users_auth WHERE email = %s", (email,))
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"Login attempt for non-existent user: {email}")
            return False
        
        if bcrypt.checkpw(password.encode(), result[0].encode()):
            logger.info(f"User {email} logged in successfully")
            return True
        else:
            logger.warning(f"Failed login attempt for user: {email}")
            return False
    except mysql.connector.Error as err:
        logger.error(f"Database error during login: {err}")
        st.error(f"Login failed: {err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        st.error(f"Login failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# Firebase login with improved error handling
def firebase_google_login(firebase_token):
    if not firebase_token:
        logger.warning("Empty Firebase token provided")
        return None
    
    try:
        decoded_token = auth.verify_id_token(firebase_token)
        email = decoded_token.get('email')
        if email:
            logger.info(f"Firebase user {email} logged in successfully")
            return email
        else:
            logger.warning("Firebase token did not contain email")
            return None
    except Exception as e:
        logger.error(f"Firebase authentication error: {e}")
        st.error(f"Google login failed: {e}")
        return None
