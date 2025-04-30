import bcrypt
import mysql.connector
from mysql.connector import Error
import streamlit as st
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL connection with error handling
def get_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="rounak",
            password="rounakbag24",
            database="RounakDB"
        )
        if conn.is_connected():
            return conn
    except Error as err:
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
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("users_auth table created successfully")

        # Create user_preferences table if it doesn't exist
        cursor.execute("SHOW TABLES LIKE 'user_preferences'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE user_preferences (
                    user_id INT PRIMARY KEY,
                    preferred_region VARCHAR(50) DEFAULT 'India',
                    preferred_models VARCHAR(255) DEFAULT 'RainForest',
                    prediction_duration VARCHAR(50) DEFAULT '1 Year',
                    dark_mode BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (user_id) REFERENCES users_auth(id)
                )
            """)
            conn.commit()
            logger.info("user_preferences table created successfully")

        # Create user_watchlist table if it doesn't exist
        cursor.execute("SHOW TABLES LIKE 'user_watchlist'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE user_watchlist (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    region VARCHAR(50) NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users_auth(id)
                )
            """)
            conn.commit()
            logger.info("user_watchlist table created successfully")

        # Create prediction_history table if it doesn't exist
        cursor.execute("SHOW TABLES LIKE 'prediction_history'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE prediction_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    region VARCHAR(50) NOT NULL,
                    models_used VARCHAR(255) NOT NULL,
                    forecast_json LONGTEXT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users_auth(id)
                )
            """)
            conn.commit()
            logger.info("prediction_history table created successfully")

        return True
    except mysql.connector.Error as err:
        logger.error(f"Error creating tables: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# Local user registration with detailed error handling
def register_user(email, username, password):
    # Ensure users_auth table exists
    if not ensure_users_table():
        st.error("Database setup failed. Cannot register user.")
        return False

    conn = get_db()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        # Check if user already exists by email
        cursor.execute("SELECT id FROM users_auth WHERE email = %s", (email,))
        if cursor.fetchone():
            logger.warning(f"User with email {email} already exists")
            st.warning("A user with this email already exists")
            return False

        # Check if user already exists by username
        cursor.execute("SELECT id FROM users_auth WHERE username = %s", (username,))
        if cursor.fetchone():
            logger.warning(f"User with username {username} already exists")
            st.warning("A user with this username already exists")
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
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# Local login with improved error handling
def verify_user(username, password):
    conn = get_db()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash FROM users_auth WHERE username = %s", (username,))
        result = cursor.fetchone()

        if not result:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return False

        if bcrypt.checkpw(password.encode(), result[0].encode()):
            logger.info(f"User {username} logged in successfully")
            return True
        else:
            logger.warning(f"Failed login attempt for user: {username}")
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
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# --- auth_utils.py addition ---
def get_user_email(username):
    conn = get_db(); cursor = conn.cursor()
    try:
        cursor.execute("SELECT email FROM users_auth WHERE username=%s", (username,))
        row = cursor.fetchone()
        return row[0] if row else None
    except Error as err:
        logger.error(f"Error getting user email for {username}: {err}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()

def get_user_id_by_username(username):
    conn = get_db(); cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users_auth WHERE username=%s", (username,))
        row = cursor.fetchone()
        return row[0] if row else None
    except Error as err:
        logger.error(f"Error getting user ID for {username}: {err}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()

def get_user_id(email):
    conn = get_db(); cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users_auth WHERE email=%s", (email,))
        row = cursor.fetchone()
        return row[0] if row else None
    except Error as err:
        logger.error(f"Error getting user ID for {email}: {err}")
        return None
    finally:
         if conn and conn.is_connected():
            cursor.close(); conn.close()

def get_user_preferences(user_id):
    conn = get_db();
    if not conn: return {}
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM user_preferences WHERE user_id=%s", (user_id,))
        pref = cursor.fetchone()
        return pref or {}
    except Error as err:
        logger.error(f"Error getting preferences for user {user_id}: {err}")
        return {}
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def set_user_preferences(user_id, region, models, duration, dark_mode):
    conn = get_db()
    if not conn: return False
    cursor = conn.cursor()
    try:
        # Ensure models is stored correctly (e.g., as a comma-separated string)
        models_str = ",".join(models) if isinstance(models, list) else models

        # Upsert logic
        cursor.execute("""
          INSERT INTO user_preferences
            (user_id, preferred_region, preferred_models, prediction_duration, dark_mode)
          VALUES (%s, %s, %s, %s, %s)
          ON DUPLICATE KEY UPDATE
            preferred_region=VALUES(preferred_region),
            preferred_models=VALUES(preferred_models),
            prediction_duration=VALUES(prediction_duration),
            dark_mode=VALUES(dark_mode)
        """, (user_id, region, models_str, duration, bool(dark_mode))) # Ensure boolean conversion
        conn.commit()
        logger.info(f"Preferences updated for user {user_id}")
        return True
    except Error as err:
        logger.error(f"Error setting preferences for user {user_id}: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def add_to_watchlist(user_id, symbol, region):
    conn = get_db()
    if not conn: return False
    cursor = conn.cursor()
    try:
        # Check if item already exists to prevent duplicates
        cursor.execute(
            "SELECT id FROM user_watchlist WHERE user_id=%s AND symbol=%s AND region=%s",
            (user_id, symbol, region)
        )
        if cursor.fetchone():
            logger.warning(f"Watchlist item {symbol} ({region}) already exists for user {user_id}")
            return True # Or False depending on desired behavior for duplicates

        # Insert new item
        cursor.execute(
          "INSERT INTO user_watchlist(user_id, symbol, region) VALUES(%s, %s, %s)",
          (user_id, symbol, region)
        )
        conn.commit()
        logger.info(f"Added {symbol} ({region}) to watchlist for user {user_id}")
        return True
    except Error as err:
        logger.error(f"Error adding {symbol} to watchlist for user {user_id}: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def get_watchlist(user_id):
    conn = get_db()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM user_watchlist WHERE user_id=%s ORDER BY added_at DESC", (user_id,))
        lst = cursor.fetchall()
        return lst
    except Error as err:
        logger.error(f"Error fetching watchlist for user {user_id}: {err}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def add_prediction_history(user_id, symbol, region, models_used, forecast_json):
    conn = get_db()
    if not conn: return False
    cursor = conn.cursor()
    try:
        # Ensure models_used is stored correctly (e.g., comma-separated string)
        models_str = ",".join(models_used) if isinstance(models_used, list) else models_used

        cursor.execute("""
          INSERT INTO prediction_history
            (user_id, symbol, region, models_used, forecast_json, prediction_date)
          VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, symbol, region, models_str, forecast_json, datetime.now())) # Use current time
        conn.commit()
        logger.info(f"Added prediction history for {symbol} ({region}) for user {user_id}")
        return True
    except Error as err:
         logger.error(f"Error adding prediction history for user {user_id}, symbol {symbol}: {err}")
         return False
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()

# --- MODIFIED get_prediction_history function ---
def get_prediction_history(user_id, start_date=None, end_date=None):
    """
    Fetches prediction history for a user, optionally filtering by date range.
    Dates should be datetime.date objects or None.
    """
    conn = get_db()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT * FROM prediction_history WHERE user_id = %s"
        params = [user_id]

        # Add date filtering if dates are provided
        if start_date:
            # Assuming prediction_date is stored as TIMESTAMP/DATETIME
            # Convert date to datetime start of day
            start_datetime = datetime.combine(start_date, datetime.min.time())
            query += " AND prediction_date >= %s"
            params.append(start_datetime)
        if end_date:
            # Convert date to datetime end of day
            end_datetime = datetime.combine(end_date, datetime.max.time())
            query += " AND prediction_date <= %s"
            params.append(end_datetime)

        query += " ORDER BY prediction_date DESC" # Keep ordering by most recent first

        cursor.execute(query, tuple(params))
        hist = cursor.fetchall()
        return hist
    except Error as err:
        logger.error(f"Error fetching prediction history for user {user_id}: {err}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()
# --- END OF MODIFIED FUNCTION ---


def update_username(user_id, new_username):
    conn = get_db()
    if not conn: return False
    cursor = conn.cursor()
    try:
        # Check if new username is already taken by another user
        cursor.execute("SELECT id FROM users_auth WHERE username = %s AND id != %s", (new_username, user_id))
        if cursor.fetchone():
            logger.warning(f"Attempt to update username to an existing one: {new_username}")
            st.error("Username already taken.")
            return False

        # Update the username
        cursor.execute("UPDATE users_auth SET username=%s WHERE id=%s", (new_username, user_id))
        conn.commit()
        logger.info(f"Username updated to {new_username} for user {user_id}")
        return True
    except Error as err:
        logger.error(f"Error updating username for user {user_id}: {err}")
        st.error("Failed to update username.")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def get_user_profile(user_id):
    conn = get_db()
    if not conn: return None
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT email, username, created_at FROM users_auth WHERE id=%s", (user_id,))
        profile = cursor.fetchone()
        return profile
    except Error as err:
        logger.error(f"Error fetching profile for user {user_id}: {err}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def remove_from_watchlist(user_id, item_id):
    conn = get_db()
    if not conn: return False
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM user_watchlist WHERE id=%s AND user_id=%s", (item_id, user_id))
        conn.commit()
        deleted_rows = cursor.rowcount
        if deleted_rows > 0:
            logger.info(f"Removed watchlist item {item_id} for user {user_id}")
            return True
        else:
            logger.warning(f"Attempt to remove non-existent/unauthorized watchlist item {item_id} for user {user_id}")
            return False
    except Error as err:
        logger.error(f"Error removing watchlist item {item_id} for user {user_id}: {err}")
        st.error(f"Failed to remove watchlist item: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()


def delete_prediction_history_item(user_id, history_item_id):
    """Deletes a specific prediction history item for a user."""
    conn = get_db()
    if not conn:
        return False
    cursor = conn.cursor()
    try:
        # Ensure the item belongs to the user before deleting
        cursor.execute(
            "DELETE FROM prediction_history WHERE id = %s AND user_id = %s",
            (history_item_id, user_id)
        )
        conn.commit()
        deleted_rows = cursor.rowcount
        cursor.close()
        conn.close()
        if deleted_rows > 0:
            logger.info(f"Deleted prediction history item {history_item_id} for user {user_id}")
            return True
        else:
            logger.warning(f"Attempt to delete non-existent or unauthorized history item {history_item_id} for user {user_id}")
            return False # Item not found or didn't belong to user
    except mysql.connector.Error as err:
        logger.error(f"Database error deleting prediction history item {history_item_id} for user {user_id}: {err}")
        st.error(f"Failed to delete history item: {err}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def clear_all_prediction_history(user_id):
    """Deletes all prediction history for a user."""
    conn = get_db()
    if not conn:
        return False
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM prediction_history WHERE user_id = %s",
            (user_id,)
        )
        conn.commit()
        deleted_rows = cursor.rowcount
        cursor.close()
        conn.close()
        logger.info(f"Cleared {deleted_rows} prediction history items for user {user_id}")
        return True
    except mysql.connector.Error as err:
        logger.error(f"Database error clearing prediction history for user {user_id}: {err}")
        st.error(f"Failed to clear history: {err}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- Function to get the oldest prediction ID for a specific stock/region ---
def get_oldest_prediction_id_for_stock(user_id, symbol, region):
    """Finds the ID of the oldest prediction entry for a given stock and region."""
    conn = get_db()
    if not conn: return None
    cursor = conn.cursor()
    try:
        # Query for the oldest entry (minimum prediction_date) for the specific user/symbol/region
        cursor.execute(
            """
            SELECT id
            FROM prediction_history
            WHERE user_id = %s AND symbol = %s AND region = %s
            ORDER BY prediction_date ASC
            LIMIT 1
            """,
            (user_id, symbol, region)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    except Error as err:
        logger.error(f"Error finding oldest prediction for user {user_id}, stock {symbol} ({region}): {err}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close(); conn.close()