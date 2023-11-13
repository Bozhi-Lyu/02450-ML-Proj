import pandas as pd
import re

def parse_single_table(table_text):
    """
    Parses a single table text into a DataFrame.
    """
    rows = [row for row in table_text.split('\n') if not row.startswith('+')]
    header = rows[0].split('|')[1:-1]
    data = [row.split('|')[1:-1] for row in rows[1:]]
    header = [cell.strip() for cell in header]
    data = [[cell.strip() for cell in row] for row in data]
    df = pd.DataFrame(data, columns=header)
    return df

def convert_ann_E_to_float(df):
    """
    Converts the 'ann_E' column values from tensor format to floats.
    """
    df['ann_E'] = df['ann_E'].str.extract(r'tensor\(([\d.]+)\)').astype(float)
    return df

def select_lowest_ann_E_rows(dataframes):
    """
    For each outer fold number, compares the 'ann_E' values across all tables
    and selects the row with the lowest 'ann_E' value.
    """
    selected_rows = []
    for fold_number in range(1, 11):
        min_ann_E_row = None
        min_ann_E_value = float('inf')
        for df in dataframes:
            row = df[df['Outer fold'] == str(fold_number)]
            if not row.empty and row['ann_E'].iloc[0] < min_ann_E_value:
                min_ann_E_row = row
                min_ann_E_value = row['ann_E'].iloc[0]
        if min_ann_E_row is not None:
            selected_rows.append(min_ann_E_row)
    new_df = pd.concat(selected_rows)
    return new_df

# Reading the file
file_path = 'performance_table.txt'
with open(file_path, 'r') as file:
    file_content = file.read()

# Parsing the tables
sections = file_content.split('\n' + '=' * 40 + '\n')
all_tables_dfs = [parse_single_table(section) for section in sections if section.strip()]

# Converting 'ann_E' column in each DataFrame
converted_dfs = [convert_ann_E_to_float(df.copy()) for df in all_tables_dfs]

# Creating the new DataFrame with rows having the lowest 'ann_E' values for each outer fold
lowest_ann_E_df = select_lowest_ann_E_rows(converted_dfs)
lowest_ann_E_df.reset_index(drop=True, inplace=True)  # Resetting index for clarity

# The final DataFrame 'lowest_ann_E_df' is the result

