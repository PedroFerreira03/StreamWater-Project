import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.db_path = self.base_dir / "../../db/telemetria.db"
        self.db_path.parent.mkdir(exist_ok=True)  # Create db directory if needed
    
    def get_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            return conn
        
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def create_tables(self, drop_existing=False):
        """Create tables with option to preserve existing data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if drop_existing:
                    logger.warning("Dropping existing tables - all data will be lost!")
                    cursor.execute("DROP TABLE IF EXISTS os")
                    cursor.execute("DROP TABLE IF EXISTS contador")
                
                # Create contador table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS contador(
                    id VARCHAR(10),
                    calibre INTEGER NOT NULL,
                    ano_contador INTEGER NOT NULL,
                    tipo_consumo VARCHAR(100) NOT NULL,
                    PRIMARY KEY(id)
                )""")
                
                # Create os table with better constraints
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS os(
                    id INTEGER,
                    contador_id VARCHAR(10) NOT NULL,
                    sintoma VARCHAR(100) NOT NULL,
                    obs TEXT NOT NULL,
                    data_in TEXT NOT NULL,
                    data_fim TEXT NOT NULL,
                    dur VARCHAR(8) NOT NULL,
                    PRIMARY KEY(id),
                    FOREIGN KEY(contador_id) REFERENCES contador(id)
                )""")
                                
                conn.commit()
                logger.info("Tables created successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to create tables: {e}")
            raise


if __name__ == '__main__':
    db_manager = DatabaseManager()
    drop_existing = input("Drop existing tables? (y/N): ").lower() == 'y' # Ask user before dropping existing data
    db_manager.create_tables(drop_existing=drop_existing)
