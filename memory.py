import sqlite3

def get_all_memory():
    conn = sqlite3.connect('aura_memory.db')
    cursor = conn.cursor()
    
    # NEW: Create the table if it's missing so the app doesn't crash
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            info TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    
    # Now try to select the data
    cursor.execute('SELECT info, timestamp FROM knowledge ORDER BY timestamp DESC')
    data = cursor.fetchall()
    conn.close()
    return data
