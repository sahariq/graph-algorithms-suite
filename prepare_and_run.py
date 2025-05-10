import urllib.request
import gzip
import shutil
import os
import subprocess

# Download the dataset
dataset_url = 'https://snap.stanford.edu/data/ca-GrQc.txt.gz'
gz_file = 'ca-GrQc.txt.gz'
txt_file = 'ca-GrQc.txt'
sample_file = 'sample_graph.txt'

print('Downloading dataset...')
urllib.request.urlretrieve(dataset_url, gz_file)

print('Extracting dataset...')
with gzip.open(gz_file, 'rb') as f_in:
    with open(txt_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Clean the file (remove comment lines)
print('Cleaning dataset...')
with open(txt_file, 'r') as fin, open(sample_file, 'w') as fout:
    for line in fin:
        if not line.startswith('#') and line.strip():
            fout.write(line)

# Pick a valid source node (first node in the file)
with open(sample_file, 'r') as f:
    first_line = f.readline()
    source_node = int(first_line.split()[0])

print(f'Using source node: {source_node}')

# Run the main script with correct arguments
print('Running graph_algorithms.py...')
subprocess.run(['python', 'graph_algorithms.py', '--file', sample_file, '--source', str(source_node)])

# Clean up downloaded files (optional)
os.remove(gz_file)
os.remove(txt_file)
print('Done! All results and traces are generated.') 