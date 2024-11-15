"""magma_curves_to_parquet.py

This script reads a JSON file containing elliptic curve data and converts it
to a parquet file. The JSON file contains a list of elliptic curve data, where
each curve is represented as a list of coefficients
[a1, a2, a3, a4, a6, conductor, rank].

The script reads the JSON file, computes the normalized a_p values for the
first B primes (B being a parameter), and saves the data to a parquet file.

To use this script, run the following from the top level of the repository:

    $ sage -python python/magma_curves_to_parquet.py

Do be sure to change the INPUT_FILE and OUTPUT_FILE variables as needed.

"""


import pandas as pd
import json
import numpy as np
from sage.all import EllipticCurve, primes_first_n, round

# Step 1: Define the constants
INPUT_FILE = 'data_files/e6cond30.txt'
OUTPUT_FILE = 'data_files/e6cond30.parquet'
NUM_DECIMAL_PLACES = 4  # Number of decimal places to round normalized a_p values to
NUM_AP_VALS = 1000  # Number of primes to use for the a_p values
OUTPUT_COLS = [str(p) for p in primes_first_n(NUM_AP_VALS)] + ['conductor', 'rank']

# Step 2: Open the file
with open(INPUT_FILE, 'r') as f:
    # Step 3: Use json.load() to read the file
    data = json.load(f)


def ap_normalized(E, p):
    ap = E.ap(p)
    normalization_quotient = 2 * p.sqrt()
    return np.float32(round(ap / normalization_quotient, NUM_DECIMAL_PLACES))

# Function to get the data and labels
def get_data(tab):
    data = []
    for curve in tab:
        ainvs = curve[:-2]
        E = EllipticCurve(ainvs)
        data_this_curve = {str(p): ap_normalized(E, p) for p in primes_first_n(NUM_AP_VALS)}
        data_this_curve['conductor'] = curve[-1]
        data_this_curve['rank'] = curve[-2]
        data.append(data_this_curve)
    return data

data_with_ap = get_data(data)

# Step 4: Convert the result into a pandas DataFrame
df = pd.DataFrame(data_with_ap, columns=OUTPUT_COLS)

# Step 5: Save the DataFrame to a parquet file
df.to_parquet(OUTPUT_FILE, index=False)
