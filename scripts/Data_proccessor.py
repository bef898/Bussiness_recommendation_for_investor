import pandas as pd

class DataProcessor:
    def __init__(self, file_path, sheet_name):
        """Initialize with file path and sheet name."""
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.data = None
    
    def load_data(self, nrows=None):
        """Load data from the Excel file."""
        self.data = pd.read_excel(self.file_path, sheet_name=self.sheet_name, nrows=nrows)
        print("Data loaded successfully.")
        return self.data
    
def clean_data(self):
    """Clean the dataset by filling or replacing missing values and handling duplicates."""
    if self.data is not None:
        # Replace missing values in specific columns (IMSI, MSISDN/Number) with 0
        self.data['IMSI'].fillna(0, inplace=True)
        self.data['MSISDN/Number'].fillna(0, inplace=True)

        # Replace missing values in all other columns with 0 (or any other appropriate value)
        self.data.fillna(0, inplace=True)
        
        # Remove any duplicate rows
        self.data.drop_duplicates(inplace=True)
        
        print("Data cleaned successfully.")
    else:
        raise ValueError("No data loaded. Please call load_data() first.")

    
    def transform_data(self):
        """Transform the dataset by creating new features and modifying existing ones."""
        if self.data is not None:
            self.data['Start'] = pd.to_datetime(self.data['Start'])
            self.data['End'] = pd.to_datetime(self.data['End'])
            self.data['Duration (seconds)'] = self.data['Dur. (ms)'] / 1000
            print("Data transformed successfully.")
        else:
            raise ValueError("No data loaded. Please call load_data() first.")
    
     
    def get_data(self):
        """Return the processed data."""
        return self.data
