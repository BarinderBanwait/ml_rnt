import os
import pandas as pd

dir = r"..\data_files\1000_aps"
# get all files in the directory
files = os.listdir(dir)
# delete any previous combined file
try:
    files.remove("ecq_B_1000_all_one_per_iso_all.parquet")
except:
    pass

# initiate an empty dataframe
df = pd.DataFrame()

# concatenate all files
for file in files:
    # combine path with file name
    path = os.path.join(dir, file)
    df1 = pd.read_parquet(path)
    df = pd.concat([df,df1], axis=0)

df.sort_values(by='conductor', inplace=True)
df.to_parquet(r"..\data_files\1000_aps\ecq_B_1000_all_one_per_iso_all.parquet", index=False)