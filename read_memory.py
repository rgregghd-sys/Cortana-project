import sqlite3

def check_knowledge():
    conn = sqlite3.connect('cortana_memory.db')
    cursor = conn.cursor()
    
    print("--- CORTANA'S CURRENT KNOWLEDGE BASE ---")
    cursor.execute('SELECT * FROM knowledge')
    rows = cursor.fetchall()
    
    if not rows:
        print("Memory is empty.")
    for row in rows:
        print(f"Absorbed: {row[0]}")
    
    conn.close()

if __name__ == "__main__":
    check_knowledge()
