import sqlite3
import os

def get_connection():
    return sqlite3.connect("database/transaction.db")

def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS plots (
            plot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            size_sqm REAL,
            soil_type TEXT,
            sun_exposure TEXT,
            monthly_fee REAL
        );

        CREATE TABLE IF NOT EXISTS crops (
            crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            ideal_soil TEXT,
            sun_requirements TEXT
        );

        CREATE TABLE IF NOT EXISTS plantings (
            plot_id INTEGER PRIMARY KEY,
            crop_id INTEGER,
            renter_id INTEGER,
            planted_date DATE,
            harvest_date DATE,
            FOREIGN KEY (plot_id) REFERENCES plots(plot_id),
            FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
            FOREIGN KEY (renter_id) REFERENCES users(user_id)
        );
    """)

    conn.commit()
    conn.close()

def seed_database():
    conn = get_connection()
    cursor = conn.cursor()

    #need to see if database is already seeded
    cursor.execute("SELECT COUNT(*) FROM plots;")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    sample_users = [
        ("Alice",),
        ("Bob",),
    ]

    sample_crops = [
        ("Tomato", "Loam", "Full Sun"),
        ("Carrot", "Sandy", "Partial Shade"),
        ("Lettuce", "Loam", "Partial Shade"),
        ("Potato", "Clay", "Full Sun"),
        ("Spinach", "Clay", "Partial Shade"),
        ("Mint", "Loam", "Shade"),
        ("Cabbage", "Clay", "Full Sun"),
        ("Onion", "Loam", "Full Sun"),
        ("Radish", "Loam", "Partial Shade"),
        ("Green Bean", "Loam", "Full Sun"),
        ("Swiss Chard", "Clay", "Partial Shade"),
        ("Peas", "Loam", "Partial Shade"),
        ("Kale", "Loam", "Full Sun"),
        ("Zucchini", "Loam", "Full Sun"),
        ("Cucumber", "Loam", "Full Sun"),
        ("Pumpkin", "Loam", "Full Sun"),
        ("Broccoli", "Clay", "Full Sun"),
        ("Cauliflower", "Clay", "Full Sun"),
        ("Brussels Sprouts", "Clay", "Full Sun"),
        ("Asparagus", "Sandy", "Full Sun"),
    ]

    sample_plots = [
        (50, "Clay", "Partial Shade", 25.0),
        (100, "Loam", "Shade", 50.0),
        (100, "Clay", "Partial Shade", 50.0),
        (100, "Sandy", "Partial Shade", 50.0),
        (150, "Loam", "Partial Shade", 75.0),
        (150, "Loam", "Full Sun", 75.0),
        (150, "Clay", "Full Sun", 75.0),
        (150, "Loam", "Full Sun", 75.0),
        (200, "Clay", "Partial Shade", 100.0),
        (200, "Clay", "Full Sun", 100.0),
        (250, "Loam", "Full Sun", 125.0),
        (250, "Loam", "Partial Shade", 125.0),
        (300, "Sandy", "Full Sun", 150.0),
        (300, "Loam", "Full Sun", 150.0),
        (350, "Clay", "Partial Shade", 175.0),
        (350, "Loam", "Full Sun", 175.0),
        (400, "Loam", "Partial Shade", 200.0),
        (400, "Clay", "Full Sun", 200.0),
        (450, "Loam", "Full Sun", 225.0),
        (500, "Clay", "Partial Shade", 250.0),
    ]

    sample_plantings = [
        (8, 1, 1, "2026-01-01", "2026-06-01"),
        (1, 5, 1, "2026-01-15", "2026-06-15"),
        (10, 4, 2, "2026-02-01", "2026-07-01"),
    ]

    cursor.executemany("INSERT INTO users (name) VALUES (?);", sample_users)
    cursor.executemany("INSERT INTO crops (name, ideal_soil, sun_requirements) VALUES (?, ?, ?);", sample_crops)
    cursor.executemany("INSERT INTO plots (size_sqm, soil_type, sun_exposure, monthly_fee) VALUES (?, ?, ?, ?);", sample_plots)
    cursor.executemany("INSERT INTO plantings (plot_id, crop_id, renter_id, planted_date, harvest_date) VALUES (?, ?, ?, ?, ?);", sample_plantings)

    conn.commit()
    conn.close()

def create_user(name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO users (name) VALUES (?);", (name,))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id

def update_user_name(user_id, new_name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE users SET name = ? WHERE user_id = ?;", (new_name, user_id))
    conn.commit()
    conn.close()

def get_user_name(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id = ?;", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "User"

def reset_database():
    db_path = "database/transaction.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    init_database()
    seed_database()