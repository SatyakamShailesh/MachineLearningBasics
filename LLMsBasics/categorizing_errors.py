import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('Filtered_sandbox_errors.xlsx')

# Function to get first two lines from the 'Error' column
def get_first_two_lines(text):
    # Split the text by line breaks and return the first two lines
    lines = text.split('\\\\n')
    return '\n'.join(lines[:2])

print(f"First two lines of the first row: {get_first_two_lines(df['Error'][0])}")

# Apply the function to the 'Error' column
df['Error'] = df['Error'].apply(get_first_two_lines)

# Save the modified DataFrame back to Excel
df.to_excel('modified_file.xlsx', index=False)

