import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPopulator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.db_path = self.base_dir / "../db/telemetria.db"
        
        # Configuration with defaults
        self.config = {
            'os_data_path': 'Data/OS.csv',
            'contador_data_path': 'Data/Contadores.xlsx',
            'csv_encoding': 'latin1',
            'csv_delimiter': ';',
            'lc_pattern': r'(LC\d+)'
        }
        

    def validate_file_exists(self, file_path):
        """Validate that required data files exist"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {file_path}")
        return path
    
    def get_table_columns(self, table_name):
        """Get column names from database table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                return [col[1] for col in columns]
        except sqlite3.Error as e:
            logger.error(f"Failed to get columns for table {table_name}: {e}")
            raise
    
    def clean_os_data(self, df):
        """Clean and validate OS data"""
        try:
            logger.info(f"Processing {len(df)} OS records")
             
            # Rename problematic column
            if len(df.columns) > 2:
                df.rename(columns={df.columns[2]: "obs"}, inplace=True)
            
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                logger.info(f"Dropped unnamed columns: {unnamed_cols}")

            original_columns = list(df.columns.values)

            # Extract LC code with validation
            df["lc_code"] = df["obs"].str.extract(self.config['lc_pattern'], expand=False)
            
            # Log how many records have no LC code
            missing_lc = df['lc_code'].isna().sum()
            if missing_lc > 0:
                logger.warning(f"Dropping {missing_lc} records with missing LC codes")
            
            df = df[~df['lc_code'].isna()]
            
            new_column_order = [original_columns[0]] + ['lc_code'] + original_columns[1:]
            df = df[new_column_order]
            
            logger.info(f"Successfully cleaned {len(df)} OS records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to clean OS data: {e}")
            raise
    
    def map_columns(self, df, target_columns):
        """Map DataFrame columns to database schema"""
        try:
            if len(df.columns) != len(target_columns):
                logger.warning(f"Column count mismatch: {len(df.columns)} vs {len(target_columns)}")
            
            column_map = {old: new for old, new in zip(df.columns, target_columns)}
            df.rename(columns=column_map, inplace=True)
            
            # Validate all required columns exist
            missing_cols = set(target_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df[target_columns]  
            
        except Exception as e:
            logger.error(f"Failed to map columns: {e}")
            raise
    
    def populate_os_data(self):
        """Populate OS table"""
        try:
            # Validate file exists
            data_path = self.validate_file_exists(self.config['os_data_path'])
            
            # Read data with error handling
            try:
                df = pd.read_csv(data_path, 
                               encoding=self.config['csv_encoding'], 
                               delimiter=self.config['csv_delimiter'])
            except UnicodeDecodeError:
                logger.warning(f"Failed with {self.config['csv_encoding']} encoding")
            
            # Clean and validate data
            df = self.clean_os_data(df)
            
            # Get target schema
            os_columns = self.get_table_columns('os')
            
            # For OS data, use the reordered columns from cleaning for positional mapping
            df = self.map_columns(df, os_columns)
            
            # Insert data
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql("os", conn, if_exists="append", index=False)
                logger.info(f"Successfully inserted {len(df)} OS records")
                
        except Exception as e:
            logger.error(f"Failed to populate OS data: {e}")
            raise
    
    def populate_contador_data(self):
        """Populate contador table"""
        try:
            # Validate file exists
            data_path = self.validate_file_exists(self.config['contador_data_path'])
            
            # Read Excel file
            df = pd.read_excel(data_path)
            logger.info(f"Processing {len(df)} contador records")
            
            # Get target schema
            contador_columns = self.get_table_columns('contador')
            
            # Map columns
            df = self.map_columns(df, contador_columns)
             
            # Insert data with conflict resolution
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql("contador", conn, if_exists="append", index=False)
                logger.info(f"Successfully inserted {len(df)} contador records")
                
        except Exception as e:
            logger.error(f"Failed to populate contador data: {e}")
            raise
    
    def populate_all(self):
        """Populate all tables in correct order"""
        try:
            logger.info("Starting data population")
            
            # Order matters due to foreign key constraints
            self.populate_contador_data()
            self.populate_os_data()
            
            logger.info("Data population completed successfully")
            
        except Exception as e:
            logger.error(f"Data population failed: {e}")
            raise

if __name__ == '__main__':
    populator = DataPopulator()
    populator.populate_all()
