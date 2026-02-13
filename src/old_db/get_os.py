import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "../db/telemetria.db")

def get_os():
    # Set up connection
    try: 
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
        
            # Get the code
            lc_code = input("What's the contact id you want to check? ")
            while True:
                cursor.execute("""
                SELECT id 
                FROM contador
                WHERE id = ?""", (lc_code,)
                )
                rows = cursor.fetchall()
                
                if rows:
                    print("That is a valid code. Searching for OS occurrences...")
                    break
                else:
                    lc_code = input("The code is invalid. Type the code again: ")

            
            cursor.execute("""
            SELECT contador_id, id, sintoma
            FROM os
            WHERE contador_id = ?""", (lc_code,)         
            )
            rows = cursor.fetchall()

            # If no OS
            if len(rows) == 0:
                print(f"No occurrences found for id {lc_code}")
            else:
                keys = ['contador_id', 'os_id', 'type']
                res = []
                for result in rows:
                    res.append({x:y for x, y in zip(keys, result)})

                print(f"The following occurrences were found for id {lc_code}:")
                print(res)

    except sqlite3.Error as e:
        print(f"Error connection to the database: {e}")
        raise
    
if __name__ == '__main__':
    get_os()