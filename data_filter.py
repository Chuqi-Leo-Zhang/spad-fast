import os

cnt = 0
total = 0
dir_path = "training_data/training"
for filename in os.listdir(dir_path):
    path = os.path.join(dir_path, filename)
    if os.path.isdir(path) and len(os.listdir(path)) != 13:
        cnt += 1
        os.system(f"rm -rf {path}")
    total += 1

print(cnt)
print(total)