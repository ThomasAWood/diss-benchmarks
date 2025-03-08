from benchmarks import Benchmarks
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
import pandas as pd
import re
import glob
import os
import time
import parallel

p = parallel.Parallel()

bm = Benchmarks()
emissions_tracker = EmissionsTracker(output_dir="cc_logs", output_file="codecarbon.csv", allow_multiple_runs=True)

def run_benchmark(benchmark, iteration, run_time):

    # Delete existing logs
    delete_logs()

    # Load benchmark
    bm.load_benchmark(benchmark)
    
    carbontracker = CarbonTracker(epochs=1, log_dir="./ct_logs/", log_file_prefix="ct")
    carbontracker.epoch_start()

    benchmark_start_time = time.time()
    p.setData(0xFF)
    # Run benchmark
    with emissions_tracker:
        bm.run(benchmark, run_time)
    p.setData(0x00)

    carbontracker.epoch_end()   
    results_filename = f"./results/{benchmark_start_time}_{benchmark}_{iteration}_output.csv" 
    log_results(benchmark, results_filename)

    bm.free_memory(benchmark)

'''

LOGS PROCESSING
FUNCTIONS 

''' 
def read_carbontracker_logs(filepath): 
    actual_time = None
    actual_energy = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Loop through the log file lines.
    for i, line in enumerate(lines):
        if "Actual consumption for" in line:
            # Compile regex patterns to capture "Time" and "Energy" values.
            time_pattern = re.compile(r"Time:\s*([\d:]+)")
            energy_pattern = re.compile(r"Energy:\s*([\d\.Ee+-]+)")
            
            # Iterate through the lines following the "Actual consumption for" block.
            for subsequent_line in lines[i+1:]:
                if actual_time is None:
                    match_time = time_pattern.search(subsequent_line)
                    if match_time:
                        actual_time = match_time.group(1)
                        # Continue searching for energy if necessary.
                        continue
                if actual_energy is None:
                    match_energy = energy_pattern.search(subsequent_line)
                    if match_energy:
                        actual_energy = match_energy.group(1)
                        # Break out early if both values have been found.
                        break
            if actual_time is not None and actual_energy is not None:
                break
    # If not none, return them as a pandas data frame.
    if actual_time != None and actual_energy != None:
        # return pd.DataFrame({'duration': [actual_time], 'cpu_energy': [None], 'gpu_energy': [None], 'ram_energy': [None], 'energy_consumed': [actual_energy]})
        return pd.DataFrame({'duration': [actual_time], 'energy_consumed': [actual_energy]})

def read_codecarbon_logs(filepath):
    df = pd.read_csv(filepath)
    # columns_of_interest = ['duration','cpu_power','gpu_power','ram_power','cpu_energy','gpu_energy','ram_energy','energy_consumed']
    columns = ['duration','cpu_energy','gpu_energy','ram_energy','energy_consumed']
    return df[columns]

def log_results(benchmark, filepath):
    print("Logging Results for:", benchmark)
    carbontracker_data = read_carbontracker_logs(glob.glob("./ct_logs/*_output.log")[0])
    codecarbon_data = read_codecarbon_logs("./cc_logs/codecarbon.csv")
    # Modify carbontracker_data to have ct_ as a prefix for all column names
    carbontracker_data.columns = ['ct_' + column for column in carbontracker_data.columns]
    # Repeat for codecarbon_data
    codecarbon_data.columns = ['cc_' + column for column in codecarbon_data.columns]
    # Combine them into one dataframe
    combined_data = pd.concat([codecarbon_data, carbontracker_data], axis=1)
    combined_data['benchmark'] = benchmark
    # Write this to the csv file at filepath
    combined_data.to_csv(filepath, index=False)
    print("Results Logged!")

def delete_logs():
    print("Deleting Logs...")
    for file in glob.glob("./ct_logs/*.log"):
        os.remove(file)
    for file in glob.glob("./cc_logs/codecarbon.csv"):
        os.remove(file)
    print('Logs Deleted!')

'''


MAIN PROCESSING


'''
# Set the parallel port to 0, for any ongoing tests.
p.setData(0x00)

# Delete any existing logs
delete_logs()

# Run the actual benchmarks
run_time = 0.2
for benchmark in ["text_gen", "image_gen", "image_classification"]:
    for run in range(1):
        run_benchmark(benchmark, (run+1), run_time)
