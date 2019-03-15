import sys

all_scale_pickle_path = sys.argv[1]

with open(all_scale_pickle_path, "rb") as f:
    data = pickle.load(f)
    print(data)
