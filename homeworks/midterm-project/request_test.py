import requests


url = "http://localhost:9696/predict"

client = {'Borrowing dependency': 0.376368593751048,
  'Continuous interest rate (after tax)': 0.781663448412922,
  'Current Liability to Assets': 0.0675362575859224,
  'Current Liability to Current Assets': 0.0251228798919942,
  'Equity to Liability': 0.0289662078265387,
  'Interest Coverage Ratio (Interest expense to EBIT)': 0.56583268839997,
  'Interest Expense Ratio': 0.631092382296133,
  'Net Income to Total Assets': 0.80895702605271}

response = requests.post(url, json=client).json()

print(response)
