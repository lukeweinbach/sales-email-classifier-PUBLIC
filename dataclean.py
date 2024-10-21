# INPUT CLEANER
# Author: Luke Weinbach

import re
import pandas as pd

# Define all file paths
inputFile = 'raw-data-final.xlsx'
outputFile_excel = 'clean-data-final.xlsx'
outputFile_csv = 'clean-data-final.csv'

# Define our function to clean email body text
targetColumn = 'Body'
concatColumn = 'Subject'
def cleanCell(row):
    cleaned = re.sub("%+", " ", str(row[targetColumn]).replace("_x000D_", "").replace("\n", " ").replace(" ", "%"))
    concatStr = str(row[concatColumn])
    return concatStr + ". " + cleaned

# Initialize the dataframe from the input file
df = pd.read_excel(inputFile)

# Apply cleanCell() to all cells in the Body column
df[targetColumn] = df.apply(cleanCell, axis=1)

# Save the data frame to a file

# NOTE To save to a .csv file
df.to_csv(outputFile_csv, index=False)

# NOTE To save to a .xlsx file
# df.to_excel(outputFile_excel, index=False)