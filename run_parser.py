from percept_parser import percept
from percept_parser import plotter
import os
from joblib import Parallel, delayed
from loguru import logger

def parse_file(filename, PATH_, PATH_OUT_BASE_SUB):
    print(f"Processing file: {filename}")
    parser = percept.PerceptParser(os.path.join(PATH_, filename))
    def create_logger():
        logger.add(
            f"logs/{PATH_}_{filename}.log",
            format="{time} | {process} | {level} | {message}",
            level="INFO"
        )
    if not hasattr(parser, 'js'):
        create_logger()
        logger.info(f"{PATH_} {filename} failed to read json")
        return
    try:
        df_indefinite_streaming = parser.read_timedomain_data(indefinite_streaming=True)
        df_brainsense_streaming = parser.read_timedomain_data(indefinite_streaming=False)
    except Exception as e:
        create_logger()
        logger.info(f"{PATH_} {filename} Error processing file: {e}")
        return
    if len(df_indefinite_streaming) == 0 and len(df_brainsense_streaming) == 0:
        return
    out_dir_name = os.path.join(PATH_OUT_BASE_SUB, filename[:-len(".json")])
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    
    l_dfs = []
    for BSTD in [True, False]:
        if BSTD:
            dfs = df_brainsense_streaming
        else:
            dfs = df_indefinite_streaming
        if len(dfs) > 0:
            for df in dfs:
                plotter.plot_df_timeseries(df, out_dir_name, BSTD, FILTER=True)
                l_dfs.append(df)
    
    plotter.plot_time_domain_ranges(l_dfs, out_dir_name)

    for i, df in enumerate(l_dfs):
        time_start = df.index[0]
        time_end = df.index[-1]
        # both are in TImeStamp format
        # make a string of year-month-day_hour-minute-second
        time_start_ = time_start.strftime("%Y-%m-%d_%H-%M-%S")
        time_end_ = time_end.strftime("%Y-%m-%d_%H-%M-%S")
        str_time = f"{time_start_}_TO_{time_end_}"
        df.to_csv(os.path.join(out_dir_name, f"{str_time}.csv"))

PATH_l = [
          "/mnt/datalake/data/PerceptOCD-48392/P015/LFP/R",
          "/mnt/datalake/data/PerceptOCD-48392/P015/LFP/L", 
          "/mnt/datalake/data/TRBD-53761/TRBD001/LFP/R",
          "/mnt/datalake/data/TRBD-53761/TRBD002/LFP/R",
]

if __name__ == "__main__":
    
    compute_l = []

    PATH_OUT_BASE = "out_filter"
    for PATH_ in PATH_l:
        sub = PATH_.split("/")[-3]
        if sub == "P015":
            sub += "_"+PATH_[-1]
        PATH_OUT_BASE_SUB = os.path.join(PATH_OUT_BASE, sub)
        if not os.path.exists(PATH_OUT_BASE_SUB):
            os.makedirs(PATH_OUT_BASE_SUB)

        files_parse = [f for f in os.listdir(PATH_) if f.endswith(".json")]
        for file in files_parse:
            out_dir_name = os.path.join(PATH_OUT_BASE_SUB, file[:-len(".json")])
            #file = '._Report_Json_Session_Report_20241216T125500.json'
            compute_l.append((file, PATH_, PATH_OUT_BASE_SUB))
        #file_parse = "Report_Json_Session_Report_20250605T083624.json"
            #parse_file(file, PATH_, PATH_OUT_BASE_SUB)  # Test with a single file first

    Parallel(n_jobs=40)(delayed(parse_file)(filename, PATH_, PATH_OUT_BASE_SUB)
                        for filename, PATH_, PATH_OUT_BASE_SUB in compute_l)
