import zarr
import numpy as np
from zarr.storage import LocalStore
import pandas as pd

store = LocalStore("data.zarr")
root = zarr.open(store=store, mode="r")

print(root.tree())                 # show arrays
print("data shape:", root["data"].shape, root["data"].dtype)

# read a few values
x0 = root["data"][0]               # loads only row 0
print("x0:", x0.shape, x0.dtype, "min/max:", np.nanmin(x0), np.nanmax(x0))

print("sub[0]:", root["sub"][0])
print("ch[0]:", root["ch"][0])
print("file[0]:", root["file"][0])
t = pd.to_datetime(root["t_ns"][:])