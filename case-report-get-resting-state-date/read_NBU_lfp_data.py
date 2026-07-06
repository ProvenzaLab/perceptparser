import pandas as pd

# NBU 06-03 17.46 - 17.49; SLEEP, phone, sitting, (Neural available) Not sure
# NBU 06-04 05:31-05:33; SLEEP,sitting, resting, YES
# NBU 06-10 12.25 - 12:30, phone, sitting, YES
# NBU 06-11  08:23 - 08:28 phone, setting, Not sure
# NBU 06-17 NONE
# NBU 06-18 NONE
# NBU 11-04 13:25 - 13:30 - phone, sitting in bed, YES
# NBU 2026-02-19 12:56 - 13:00, phone, lying in bed, YES
# NBU 2026-05-12 17.27 - 17.30, phone, sitting at desk, YES

# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-03/rawlfp/raw/TRBD001_2025-06-03-2025-06-05_timedomain_LFP.parq
# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-03/rawlfp/raw/TRBD001_2025-06-03-2025-06-05_timedomain_LFP.parq
# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-10/rawlfp/raw/TRBD001_2025-06-10-2025-06-11_timedomain_LFP.parq
# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-10/rawlfp/raw/TRBD001_2025-06-10-2025-06-11_timedomain_LFP.parq
#
# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-11-04/rawlfp/raw/TRBD001_2025-11-04-2025-11-05_timedomain_LFP.parq
# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2026-02-19/rawlfp/raw/TRBD001_2026-02-19-2026-02-20_timedomain_LFP.parq
# /mnt/labworlds/Provenza/NBUPipeline/TRBD001/2026-05-12/rawlfp/raw/TRBD001_2026-05-12-2026-05-13_timedomain_LFP.parq

# CT = America/Chicago; handles CST/CDT automatically
LOCAL_TZ = "America/Chicago"

events = [
    {
        "label": "NBU_2025-06-03_sleep_phone_sitting",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-03/rawlfp/raw/TRBD001_2025-06-03-2025-06-05_timedomain_LFP.parq",
        "start_ct": "2025-06-03 17:47",
        "end_ct":   "2025-06-03 17:49",
    },
    {
        "label": "NBU_2025-06-04_sleep_phone_sitting",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-03/rawlfp/raw/TRBD001_2025-06-03-2025-06-05_timedomain_LFP.parq",
        "start_ct": "2025-06-04 05:31",
        "end_ct":   "2025-06-04 05:33",
    },
    {
        "label": "NBU_2025-06-10_phone_sitting",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-10/rawlfp/raw/TRBD001_2025-06-10-2025-06-11_timedomain_LFP.parq",
        "start_ct": "2025-06-10 12:25",
        "end_ct":   "2025-06-10 12:30",
    },
    {
        "label": "NBU_2025-06-11_phone_sitting",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-10/rawlfp/raw/TRBD001_2025-06-10-2025-06-11_timedomain_LFP.parq",
        "start_ct": "2025-06-11 08:23",
        "end_ct":   "2025-06-11 08:28",
    },
    {
        "label": "NBU_2025-11-04_phone_sitting_bed",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-11-04/rawlfp/raw/TRBD001_2025-11-04-2025-11-05_timedomain_LFP.parq",
        "start_ct": "2025-11-04 13:25",
        "end_ct":   "2025-11-04 13:30",
    },
    {
        "label": "NBU_2026-02-19_phone_lying_bed",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2026-02-19/rawlfp/raw/TRBD001_2026-02-19-2026-02-20_timedomain_LFP.parq",
        "start_ct": "2026-02-19 12:56",
        "end_ct":   "2026-02-19 13:00",
    },
    {
        "label": "NBU_2026-05-12_phone_sitting_desk",
        "parq": "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2026-05-12/rawlfp/raw/TRBD001_2026-05-12-2026-05-13_timedomain_LFP.parq",
        "start_ct": "2026-05-12 17:30",
        "end_ct":   "2026-05-12 18:00",
    },
]


def ct_to_utc(ts):
    return (
        pd.Timestamp(ts)
        .tz_localize(LOCAL_TZ)
        .tz_convert("UTC")
    )


def ensure_utc_index(df):
    idx = pd.to_datetime(df.index)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    df = df.copy()
    df.index = idx
    return df.sort_index()


all_segments = []

for ev in events:
    df = pd.read_parquet(ev["parq"])
    df = ensure_utc_index(df)

    start_utc = ct_to_utc(ev["start_ct"])
    end_utc = ct_to_utc(ev["end_ct"])

    seg = df.query("index >= @start_utc and index <= @end_utc").copy()
    if seg.empty:
        print(f"WARNING: No rows found for event {ev['label']} between {start_utc} and {end_utc}")
        continue

    seg["event_label"] = ev["label"]
    seg["start_ct"] = ev["start_ct"]
    seg["end_ct"] = ev["end_ct"]

    print(
        ev["label"],
        "| CT:", ev["start_ct"], "to", ev["end_ct"],
        "| UTC:", start_utc, "to", end_utc,
        "| rows:", len(seg)
    )

    all_segments.append(seg)

df_segments = pd.concat(all_segments)

# Optional: save extracted rows
df_segments.to_parquet("TRBD001_selected_NBU.parq")