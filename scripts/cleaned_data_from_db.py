
# Now import the module
from connection import PostgresConnection
import pandas as pd

db = PostgresConnection(dbname='telecom', user='postgres', password='1864')
db.connect()

# Example query
query = "SELECT * FROM cleaned_data_table"
result = db.execute_query(query)

# Convert the result to a Pandas DataFrame
df = pd.DataFrame(result, columns=[desc[0] for desc in db.cursor.description])
print(df.head())  # Display the first few rows of the DataFrame

# Close the connection when done
db.close_connection()