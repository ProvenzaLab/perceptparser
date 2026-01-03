import pandas as pd
import numpy as np
import glob
import os
PATH_ = "/scratch/timonmerk/get_data_NBU/perceptparser/out_filter"

# get all csv files in PATH_
csv_files = glob.glob(os.path.join(PATH_, "**", "*.csv"), recursive=True)

l_f = []
l_subs = []
l_ch = []
l_data = []
l_date = []
idx_ = 0

for f in csv_files:
    sub = f.split("/")[-3]
    # if len(l_f) > 100000:
    #     arr = np.array(l_data)
    #     np.save(f"data_{idx_}.npy", arr)
    #     d_ = {"files": l_f, "subs": l_subs, "chs": l_ch, "dates": l_date}
    #     df_meta = pd.DataFrame(d_)
    #     df_meta.to_csv(f"metadata_{idx_}.csv", index=False)
    #     l_f = []
    #     l_subs = []
    #     l_ch = []
    #     l_data = []
    #     l_date = []
    #     idx += 1

    df = pd.read_csv(f, index_col=0, parse_dates=True)
    chs = df.columns.tolist()[:-1]
    idxs = df["idx_counter"].unique()
    for idx_cnt in idxs:
        df_idx = df[df["idx_counter"] == idx_cnt]
        if len(df_idx) == 2500:
            for ch in chs:
                data = df_idx[ch].values.astype(np.float16)
                # count number of NaN's
                n_nan = np.sum(np.isnan(data))
                if n_nan == 0:
                    l_f.append(f)
                    l_subs.append(sub)
                    l_ch.append(ch)
                    l_data.append(data)
                    date_ = df_idx[ch].index[0]
                    l_date.append(date_)


arr = np.array(l_data)
np.save(f"data.npy", arr)
d_ = {"files": l_f, "subs": l_subs, "chs": l_ch, "dates": l_date}
df_meta = pd.DataFrame(d_)
df_meta.to_csv(f"metadata.csv", index=False)

#import zarr
# from zarr.codecs import BloscCodec, VLenUTF8Codec
# create zarr file
# --- after your loop finishes and you have l_data, l_f, l_subs, l_ch, l_date ---

# zarr_out = "data.zarr"

# n = len(l_data)
# seg_len = 2500
# if n == 0:
#     raise RuntimeError("No valid segments collected (n=0). Nothing to write.")

# root = zarr.group(zarr_out, overwrite=True)

# compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")

# # data: (n_segments, 2500)
# data_z = root.create_array(
#     "data",
#     shape=(n, seg_len),
#     chunks=(min(4096, n), seg_len),   # tune if you like
#     dtype=np.float16,
#     compressors=compressor
# )
# # string metadata (variable-length UTF-8)
# str_codec = VLenUTF8Codec()

# f_z = root.create_array("file", shape=(n,), chunks=(min(8192, n),), dtype=str, overwrite=True, serializer=str_codec)
# sub_z = root.create_array("sub",  shape=(n,), chunks=(min(8192, n),), dtype=str, overwrite=True, serializer=str_codec)
# ch_z = root.create_array("ch",   shape=(n,), chunks=(min(8192, n),), dtype=str, overwrite=True, serializer=str_codec)

# t_ns = pd.to_datetime(l_date).view("int64")
# t_z  = root.create_array("t_ns", shape=(n,), chunks=(min(8192, n),), dtype="i8", compressors=compressor)

# for i, x in enumerate(l_data):
#     data_z[i] = x

# sub_z[:] = np.array(l_subs, dtype=str)
# f_z[:] = np.array(l_f, dtype=str)
# ch_z[:] = np.array(l_ch, dtype=str)
# t_z[:] = t_ns

# # save and close
# print(f"Written {n} segments to {zarr_out}")