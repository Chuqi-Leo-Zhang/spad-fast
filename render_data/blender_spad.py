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
parser.add_argument("--caption", type=str, default="")  # caption for the object
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

# --- Make white actually white (disable Filmic tone mapping) ---
scene.display_settings.display_device = "sRGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
# ---------------------------------------------------------------


bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = args.device  # CUDA / OPTIX / METAL / OPENCL
bpy.context.scene.cycles.tile_size = 8192


# ------------------------- Geometry helpers -------------------------


def spherical_to_cartesian(spherical_coords):
    """
    Convert from spherical to cartesian coordinates.

    spherical_coords: array of shape [N, 3] with columns
        [elevation (theta), azimuth (phi), radius]
    This matches the convention used in SPAD's geometry.
    """
    theta, azimuth, radius = spherical_coords.T
    x = radius * np.sin(theta) * np.cos(azimuth)
    y = radius * np.sin(theta) * np.sin(azimuth)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def look_at(eye, center, up):
    """
    Same look_at as SPAD: build a camera-from-world rotation and translation.
    """
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = center - eye
    f = f / np.linalg.norm(f)

    up_norm = up / np.linalg.norm(up)
    s = np.cross(f, up_norm)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    R = np.array(
        [
            [s[0], s[1], s[2]],
            [u[0], u[1], u[2]],
            [-f[0], -f[1], -f[2]],
        ],
        dtype=np.float32,
    )
    T = -R @ eye
    return R, T


def get_blender_from_spherical_np(elevation, azimuth, radius=CAMERA_RADIUS):
    """
    NumPy clone of SPAD's get_blender_from_spherical.

    Returns a 3x4 camera matrix [R|t] in the same convention SPAD uses
    for Plücker / epipolar geometry.
    """
    cart = spherical_to_cartesian(
        np.array([[elevation, azimuth, radius]], dtype=np.float32)
    )
    eye = cart[0]
    center = np.zeros(3, dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)

    R, T = look_at(eye, center, up)
    # same post-processing as original SPAD code
    R = R.T
    T = -R @ T
    RT = np.concatenate([R, T[:, None]], axis=1)  # [3,4]
    return RT, eye  # return both extrinsic and camera center


def set_camera_location(cam_pt):
    x, y, z = cam_pt
    camera = bpy.data.objects["Camera"]
    camera.location = (float(x), float(y), float(z))
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
    """Center & rescale objects so longest side fits into (-1, 1)^3."""
    bbox_min, bbox_max = scene_bbox()
    longest = max(bbox_max - bbox_min)       # length of longest side
    scale = 2.0 / float(longest)            # longest side spans [-1,1]
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2.0
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


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

    # radii are fixed at CAMERA_RADIUS
    distances = np.asarray(
        [CAMERA_RADIUS for _ in range(args.num_images)], np.float32
    )

    # sample spherical angles
    if args.camera_type == "fixed":
        azimuths = (
            np.arange(args.num_images) / args.num_images * 2 * np.pi
        ).astype(np.float32)
        elev_deg = 45.0  # fixed elevation (like SPAD's eval)
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

    cam_poses = []

    for i, (elev, az) in enumerate(zip(elevations, azimuths)):
        # get SPAD-style extrinsic + camera center from spherical angles
        RT, eye = get_blender_from_spherical_np(float(elev), float(az))
        cam_poses.append(RT)

        # move Blender camera to the same center; TRACK_TO enforces orientation
        camera = set_camera_location(eye)

        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        if os.path.exists(render_path):
            continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)

    # save meta: intrinsics + spherical coords + radii + extrinsics
    K = get_spad_K()
    cam_poses = np.stack(cam_poses, 0)  # [num_views, 3, 4]
    meta_path = os.path.join(args.output_dir, object_uid, "meta.pkl")
    meta = {
        "K": K,                         # (3x3) intrinsic matrix
        "azimuths": azimuths,           # (num_views,) in radians
        "elevations": elevations,       # (num_views,) in radians
        "distances": distances,         # (num_views,) camera radius
        "cam_poses": cam_poses,         # (num_views, 3, 4) SPAD-style RT
        "object_id": object_uid,        # string id
        "caption": args.caption,        # caption for the object
    }

    save_pickle(meta, meta_path)


if __name__ == "__main__":
    save_images(args.object_path)