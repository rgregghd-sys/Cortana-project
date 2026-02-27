import sqlite3

def save_to_memory(data_list):
    conn = sqlite3.connect('cortana_memory.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS knowledge (info TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    for info in data_list:
        cursor.execute('INSERT INTO knowledge (info) VALUES (?)', (info,))
    conn.commit()
    conn.close()

def get_all_memory():
    conn = sqlite3.connect('cortana_memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT info, timestamp FROM knowledge ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows
