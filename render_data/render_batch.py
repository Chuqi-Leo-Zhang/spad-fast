import os
import subprocess
import pickle

import objaverse
import multiprocessing

import json


def load_captions(jsonl_path):
    uid2cap = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            uid = item["uid"]          # your “uid”
            caption = item["caption"]
            uid2cap[uid] = caption
    return uid2cap


def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb+') as f:
        pickle.dump(data, f)


def blender_cmd(objects, uid, caption):
    os.makedirs('training', exist_ok=True)
    cmds = ['blender', '--background', '--python','blender_spad.py','--',
            '--object_path', f'{objects[uid]}',
            '--output_dir','./training/','--camera_type','random',
            '--caption', caption]
    subprocess.run(cmds)

if __name__ == '__main__':
    
    uid2cap = load_captions("cap3d_intersect_37k_caption.jsonl")

    with open('cap3d_intersect.txt', 'r') as f:
        all_uids = [line.strip() for line in f.readlines()]
    uids = all_uids[9:10]
    save_pickle(uids, f'training/uid_set.pkl')

    processes = 1
    objects = objaverse.load_objects(
        uids=uids,
        download_processes=processes,
    )
    print(objects)

    processes = multiprocessing.cpu_count()
    print(f"Using {processes} processes")
    with multiprocessing.Pool(processes=processes // 3) as pool:
        pool.starmap(blender_cmd, [(objects, uid, uid2cap[uid]) for uid in uids])

    print("Done rendering")

