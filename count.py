import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

print(len(data))
