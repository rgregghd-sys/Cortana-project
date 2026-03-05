import sqlite3

conn = sqlite3.connect('aura_universe.db')
c = conn.cursor()
# User Table
c.execute('''CREATE TABLE IF NOT EXISTS users 
             (username TEXT PRIMARY KEY, password TEXT)''')
# Memory Table (Linked to username)
c.execute('''CREATE TABLE IF NOT EXISTS memory 
             (username TEXT, prompt TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()
conn.close()
print("Aura's Database Initialized.")
