from benchmarks import Benchmarks
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
from carbontracker import parser
import pandas as pd
import re
import glob
import os
import time
import parallel

p = parallel.Parallel()

bm = Benchmarks()

def run_benchmark(benchmark, benchmark_iteration, run, run_time):

    # Delete existing logs
    delete_logs()

    # Load benchmark
    bm.load_benchmark(benchmark)
    
    carbontracker = CarbonTracker(epochs=1, log_dir="./ct_logs/", log_file_prefix="ct", update_interval=1)
    carbontracker.epoch_start()

    # benchmark_start_time = time.time()
    p.setData(0xFF)
    # Run benchmark
    tracker.start_task()
    bm.run(benchmark, run_time)
    p.setData(0x00)
    cc_data = tracker.stop_task()
    carbontracker.epoch_end()   

    results_filename = f"./results/output.csv" 
    log_results(benchmark, benchmark_iteration, run, results_filename, cc_data)

    bm.free_memory(benchmark)

'''

LOGS PROCESSING
FUNCTIONS 

''' 
def read_carbontracker_logs(log_dir): 
    actual_time = None
    actual_energy = None

    logs = parser.parse_all_logs(log_dir)
    assert len(logs) > 0, "CarbonTracker: No logs found"
    first_log = logs[0]
    actual_time = first_log['actual']['duration (s)']
    actual_energy = first_log['actual']['energy (kWh)']
    assert actual_time is not None and actual_energy is not None, "CarbonTracker: Could not find time and energy in logs"
    return pd.DataFrame({'duration': [actual_time], 'energy_consumed': [actual_energy]})

def read_codecarbon_logs(filepath):
    df = pd.read_csv(filepath)
    # columns_of_interest = ['duration','cpu_power','gpu_power','ram_power','cpu_energy','gpu_energy','ram_energy','energy_consumed']
    columns = ['duration','cpu_energy','gpu_energy','ram_energy','energy_consumed']
    return df[columns]

def log_results(benchmark, benchmark_iteration, run, filepath, cc_data):
    print("Logging Results for:", benchmark)
    # carbontracker_data = read_carbontracker_logs(glob.glob("./ct_logs/*_output.log")[0])
    carbontracker_data = read_carbontracker_logs('./ct_logs/')

    # codecarbon_data = read_codecarbon_logs("./cc_logs/codecarbon.csv")
    data = {
        'cc_duration': [cc_data.duration],
        'cc_cpu_energy': [cc_data.cpu_energy],
        'cc_gpu_energy': [cc_data.gpu_energy],
        'cc_ram_energy': [cc_data.ram_energy],
        'cc_energy_consumed': [cc_data.energy_consumed]
    }
    codecarbon_data = pd.DataFrame(data)

    # Modify carbontracker_data to have ct_ as a prefix for all column names
    carbontracker_data.columns = ['ct_' + column for column in carbontracker_data.columns]

    # Combine them into one dataframe
    combined_data = pd.concat([codecarbon_data, carbontracker_data], axis=1)
    combined_data['benchmark'] = benchmark
    combined_data['benchmark_iteration'] = benchmark_iteration
    combined_data['run'] = run
    # Write this to the csv file at filepath
    file_exists = not(os.path.exists(filepath))
    combined_data.to_csv(filepath, mode='a', header=file_exists, index=False)
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
# If file exists, delete it
if os.path.exists('./results/output.csv'):
    os.remove('./results/output.csv')
# Set the parallel port to 0, for any ongoing tests.
p.setData(0x00)
tracker = EmissionsTracker(output_dir="cc_logs", output_file="codecarbon.csv", allow_multiple_runs=True, measure_power_secs=1)

# Run the actual benchmarks
run_time = 5
run = 0
for benchmark in ["text_gen", "image_gen", "image_classification"]:
    for benchmark_iteration in range(10):
        run_benchmark(benchmark, benchmark_iteration, run, run_time)
        run += 1

tracker.stop()
