
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utils
from datetime import datetime
import pickle
import os 

testset_fraction=0.2

dataset_one_path = os.path.join(os.path.dirname(__file__),"..", "data", "dataset_one.csv")
dataset_two_path = os.path.join(os.path.dirname(__file__),"..", "data", "dataset_two.xlsx")
preprocessed_data_path = os.path.join(os.path.dirname(__file__),"..", "data", "data_preprocessed.pkl")

# load both datasets
data1 = pd.read_csv(dataset_one_path, sep=";", decimal=",", encoding="iso-8859-1", delimiter=";", parse_dates={"date and time": ["Date", "Time"]})
data2 = pd.read_excel(dataset_two_path, parse_dates={"date and time": ["date", "time"]})

# rename columns
data1.rename(columns={"Mean": "LAI"}, inplace=True)
data1.rename(columns=lambda name: name if name not in map(str, utils.spectrogram_wavelengths) else int(name), inplace=True)

# group data
data1["group"] = data1["date and time"].map(lambda date: "A" if date <= datetime(2012, 1, 1) else "B")
data2["group"] = data2["date and time"].map(lambda date: "C" if date <= datetime(2016, 1, 1) else "D")

# combine data
data = pd.concat([data1, data2], axis=0, join="inner")[["group", "date and time", "LAI"]+list(utils.spectrogram_wavelengths)]

def split(df):
    num_test_samples = round(testset_fraction*df.shape[0])
    idxs = np.random.choice(df.shape[0], num_test_samples, replace=False)
    mask = np.zeros(df.shape[0],dtype=bool)
    mask[idxs] = True
    df["test"] = mask
    return df    

# add training / testing set split column
data = data.groupby("group").apply(split)
data.set_index(["test", "group", "date and time"], inplace=True)
data.sort_index(inplace=True)

# save data
with open(preprocessed_data_path, "wb") as file:
    pickle.dump(data, file)
