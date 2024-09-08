from sqlalchemy import create_engine

class send_to_db():
    def __init__(self,cleaned_df):
        self.cleaned_df = cleaned_df
        
    def connection(self,cleaned_df):
        # Step 1: Replace these credentials with your actual database credentials
        username = 'postgres'
        password = '1864'
        host = 'localhost'  # or the IP address where your DB is hosted
        port = '5432'  # default port for PostgreSQL
        database = 'telecom'

        # Step 2: Create a database connection string
        connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'

        # Step 3: Create an SQLAlchemy engine
        
        engine = create_engine(connection_string)

        # Step 4: Send the DataFrame to the database
        # 'cleaned_df' is the DataFrame you want to send
        table_name = 'cleaned_data_table'  # Name of the table in the database
        cleaned_df.to_sql(table_name, engine, if_exists='replace', index=False)

        print(f"Data successfully sent to the '{table_name}' table in the '{database}' database!")



