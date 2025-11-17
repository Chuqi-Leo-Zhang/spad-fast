import os
import subprocess
import pickle

import objaverse
import multiprocessing

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb+') as f:
        pickle.dump(data, f)

with open('objaverse_filtered.txt', 'r') as f:
   all_uids = [line.strip() for line in f.readlines()]

uids = all_uids[:5]

save_pickle(uids, f'training/uid_set.pkl')

processes = 1
objects = objaverse.load_objects(
    uids=uids,
    download_processes=processes,
)

print(objects)

def blender_cmd(objects, uid):
    os.makedirs('training', exist_ok=True)
    cmds = ['blender', '--background', '--python','blender_spad.py','--',
            '--object_path', f'{objects[uid]}',
            '--output_dir','./training/','--camera_type','random']
    subprocess.run(cmds)

if __name__ == '__main__':      

    processes = multiprocessing.cpu_count()
    print(f"Using {processes} processes")
    with multiprocessing.Pool(processes=processes // 3) as pool:
        pool.starmap(blender_cmd, [(objects, uid) for uid in uids])

    print("Done rendering")

