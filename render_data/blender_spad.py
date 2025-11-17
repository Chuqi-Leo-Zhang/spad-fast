"""
Blender script to render images of 3D models in a way that matches
the SPAD / Zero123 Objaverse setup.

Usage:
    blender -b -P blender_spad.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --camera_type random \
        --num_images 12 \
        --device CUDA
"""

import argparse
import math
import os
import sys
from pathlib import Path
import pickle

import numpy as np
import bpy
from mathutils import Vector

# ------------------------- I/O utils -------------------------


def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


# ------------------------- Args -------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument(
    "--camera_type", type=str, default="random", choices=["random", "fixed"]
)
parser.add_argument("--num_images", type=int, default=12)  # SPAD uses 12 views
parser.add_argument("--device", type=str, default="CUDA")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print("===================", args.engine, "===================")

# ------------------------- SPAD constants -------------------------

SPAD_FOV = 0.702769935131073  # ~40.26 degrees
CAMERA_RADIUS = 3.5           # distance to origin
RESOLUTION = 256              # 256 x 256
ELEV_MIN_DEG = -90.0
ELEV_MAX_DEG = 90.0
ENV_LIGHT = 1.0               # white background


def get_spad_K():
    """SPAD/Zero123 intrinsics: K = diag(focal, focal, 1)."""
    focal = 1.0 / math.tan(SPAD_FOV / 2.0)
    K = np.array(
        [
            [focal, 0.0, 0.0],
            [0.0, focal, 0.0],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    return K


# ------------------------- Scene setup -------------------------

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]

cam.data.angle = SPAD_FOV
cam.location = (0.0, CAMERA_RADIUS, 0.0)

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = RESOLUTION
render.resolution_y = RESOLUTION
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True

scene.render.film_transparent = False

bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = args.device  # CUDA / OPTIX / METAL / OPENCL
bpy.context.scene.cycles.tile_size = 8192


# ------------------------- Geometry helpers -------------------------


def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths) * np.cos(elevations)
    y = np.sin(azimuths) * np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x, y, z], -1)


def set_camera_location(cam_pt):
    x, y, z = cam_pt
    camera = bpy.data.objects["Camera"]
    camera.location = (x, y, z)
    return camera


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    for obj in list(bpy.data.objects):
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a glb/fbx model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene():
    """Center & rescale objects to fit into a unit cube (-1..1)^3, as in SPAD."""
    bbox_min, bbox_max = scene_bbox()
    scale = 1.0 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2.0
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


# function from shapenet renderer
def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], 1)
    return RT


# ------------------------- Main render -------------------------


def save_images(object_file: str) -> None:
    object_uid = os.path.basename(object_file).split(".")[0]
    (Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)

    reset_scene()
    load_object(object_file)
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # simple white world background (not transparent)
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes["Background"]
    back_node.inputs["Color"].default_value = Vector(
        [ENV_LIGHT, ENV_LIGHT, ENV_LIGHT, 1.0]
    )
    back_node.inputs["Strength"].default_value = ENV_LIGHT

    # *** CHANGED: radius matches SPAD's 3.5, not 1.5 ***
    distances = np.asarray([CAMERA_RADIUS for _ in range(args.num_images)], np.float32)

    # *** CHANGED: elevation range [-90, 90] deg; azimuth uniform [0, 2π] ***
    if args.camera_type == "fixed":
        azimuths = (
            np.arange(args.num_images) / args.num_images * 2 * np.pi
        ).astype(np.float32)
        elev_deg = np.random.uniform(ELEV_MIN_DEG, ELEV_MAX_DEG)
        elevations = np.deg2rad(
            np.full(args.num_images, elev_deg, dtype=np.float32)
        )
    elif args.camera_type == "random":
        azimuths = np.random.uniform(0.0, 2 * np.pi, args.num_images).astype(
            np.float32
        )
        elev_deg = np.random.uniform(
            ELEV_MIN_DEG, ELEV_MAX_DEG, args.num_images
        ).astype(np.float32)
        elevations = np.deg2rad(elev_deg)
    else:
        raise NotImplementedError

    cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    cam_poses = []

    for i in range(args.num_images):
        camera = set_camera_location(cam_pts[i])
        RT = get_3x4_RT_matrix_from_blender(camera)
        cam_poses.append(RT)

        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        if os.path.exists(render_path):
            continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)

    # save meta: intrinsics + spherical coords + radii + extrinsics
    K = get_spad_K()
    cam_poses = np.stack(cam_poses, 0)
    meta_path = os.path.join(args.output_dir, object_uid, "meta.pkl")
    meta = {
        "K": K,                         # (3x3) intrinsic matrix
        "azimuths": azimuths,           # (num_views,) in radians
        "elevations": elevations,       # (num_views,) in radians
        "distances": distances,         # (num_views,) camera radius
        "cam_poses": cam_poses,         # (num_views, 3, 4)
        "object_id": object_uid,        # string id
    }

    save_pickle(meta, meta_path)


if __name__ == "__main__":
    save_images(args.object_path)
