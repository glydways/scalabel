"""Convert kitti to Scalabel format."""
import argparse
import copy
import math
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from ..common.parallel import NPROC
from ..common.typing import NDArrayF64
from ..label.transforms import xyxy_to_box2d
from ..label.utils import cart2hom, rotation_y_to_alpha
from .io import save
from .kitti_utlis import KittiPoseParser, list_from_file, read_calib, read_oxts
from .scalabel_typing import (
    Box3D,
    Category,
    Config,
    Dataset,
    Extrinsics,
    Frame,
    FrameGroup,
    ImageSize,
    Intrinsics,
    Label,
)
from .utils import (
    get_box_transformation_matrix,
    get_extrinsics_from_matrix,
    get_matrix_from_extrinsics,
)
import numpy as np
import struct
from open3d import *

kitti_cats = {
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Tram": "tram",
    "Person": "pedestrian",
    "Person_sitting": "pedestrian",
    "Misc": "misc",
    "Glydcar": "glydcar",
    "DontCare": "dontcare",
}

kitti_used_cats = ["pedestrian", "cyclist", "car", "truck", "tram", "glydcar", "misc"]


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to scalabel")
    parser.add_argument(
        "--input-dir",
        "-i",
        default=None,
        help="path to the input coco label file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="path to save scalabel format label file",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default=None,
        help="same path to save and load the labels",
    )
    parser.add_argument(
        "--split",
        default="training",
        choices=["training", "testing"],
        help="split for kitti dataset",
    )
    parser.add_argument(
        "--data-type",
        default="detection",
        choices=["tracking", "detection"],
        help="type of kitti dataset",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def convert_kitti_bin_to_ply(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    return pcd


def heading_transform(box3d: Box3D, calib: NDArrayF64) -> float:
    """Transform heading from camera coordinate into lidar system."""
    rot_y = box3d.orientation[1]
    tm: NDArrayF64 = calib @ get_box_transformation_matrix(
        box3d.location,
        (box3d.dimension[0], box3d.dimension[1], box3d.dimension[2]),
        rot_y,
    )
    pt1: NDArrayF64 = np.array([-0.5, 0.5, 0, 1.0], dtype=np.float64)
    pt2: NDArrayF64 = np.array([0.5, 0.5, 0, 1.0], dtype=np.float64)
    pt1 = np.matmul(tm, pt1).tolist()
    pt2 = np.matmul(tm, pt2).tolist()
    return -math.atan2(pt2[2] - pt1[2], pt2[0] - pt1[0])


def parse_lidar_labels(
    labels: List[Label],
    calib: Extrinsics,
) -> List[Label]:
    """Function parsing tracking / detection labels into lidar frames."""
    lidar2cam_mat = get_matrix_from_extrinsics(calib)
    cam2lidar_mat = np.linalg.inv(lidar2cam_mat)

    new_labels = []
    for label in labels:
        if label.box3d is not None:
            box3d = label.box3d

            center: NDArrayF64 = np.array([box3d.location], dtype=np.float64)
            center_lidar: Tuple[float, float, float] = tuple(  # type: ignore
                np.dot(cam2lidar_mat, cart2hom(center).T)[:3, 0].tolist()
            )
            if abs(cam2lidar_mat - np.eye(4)).sum() > 1e-6:
                heading = heading_transform(box3d, cam2lidar_mat)
                orientation = (0.0, heading, 0.0)
            else:
                heading = box3d.orientation[1]
                orientation = (0.0, 0.0, box3d.orientation[1])

            new_box3d = Box3D(
                orientation=orientation,
                location=center_lidar,
                dimension=box3d.dimension,
                alpha=rotation_y_to_alpha(heading, center_lidar),
            )
        new_labels.append(
            Label(
                category=label.category,
                box3d=new_box3d if label.box3d is not None else None,
                id=label.id,
            )
        )

    return new_labels


def get_extrinsics(
    rect: NDArrayF64,
    velo2cam: NDArrayF64,
    cam2global: Extrinsics,
) -> Tuple[Extrinsics, Extrinsics]:
    """Convert KITTI velodyne to global extrinsics."""
    lidar2cam_mat = np.dot(rect, velo2cam)
    lidar2global_mat = np.dot(
        get_matrix_from_extrinsics(cam2global), lidar2cam_mat
    )
    lidar2cam = get_extrinsics_from_matrix(lidar2cam_mat)
    lidar2global = get_extrinsics_from_matrix(lidar2global_mat)
    return lidar2cam, lidar2global


def parse_label(
    data_type: str,
    label_file: str,
    trackid_maps: Dict[str, int],
    global_track_id: int,
    glydway_datset: bool
) -> Tuple[Dict[int, List[Label]], Dict[str, int], int]:
    """Function parsing tracking / detection labels."""
    if data_type == "tracking":
        offset = 2
    else:
        offset = 0

    labels_dict: Dict[int, List[Label]] = {}

    labels = list_from_file(label_file)
    track_id = -1

    for label_line in labels:
        if label_line == "":
            continue
        label = label_line.split()
        if data_type == "tracking":
            seq_id = int(label[0])
        else:
            seq_id = 0

        if seq_id not in labels_dict:
            labels_dict[seq_id] = []

        cat = label[0 + offset]
        if cat in ["DontCare"]:
            continue
        class_name = kitti_cats[cat]

        if data_type == "tracking":
            if label[1] in trackid_maps.keys():
                track_id = trackid_maps[label[1]]
            else:
                track_id = global_track_id
                trackid_maps[label[1]] = track_id
                global_track_id += 1
        else:
            track_id += 1

        box2d = xyxy_to_box2d(
            float(label[4 + offset]),
            float(label[5 + offset]),
            float(label[6 + offset]),
            float(label[7 + offset]),
        )

        if not glydway_datset:
            y_cen_adjust = float(label[8 + offset]) / 2.0
        else:
            y_cen_adjust = 0.0

        box3d = Box3D(
            orientation=(0.0, float(label[14 + offset]), 0.0),
            location=(
                float(label[11 + offset]),
                float(label[12 + offset]) - y_cen_adjust,
                float(label[13 + offset]),
            ),
            dimension=(
                float(label[8 + offset]),
                float(label[9 + offset]),
                float(label[10 + offset]),
            ),
            alpha=float(label[3 + offset]),
        )

        labels_dict[seq_id].append(
            Label(
                category=class_name,
                box2d=box2d,
                box3d=box3d,
                id=str(track_id),
            )
        )

    return labels_dict, trackid_maps, global_track_id


def generate_labels_cam(labels: List[Label], offset: float) -> List[Label]:
    """Adjust label with different cameras."""
    labels_cam = []
    for label in labels:
        label_cam = copy.deepcopy(label)
        assert label_cam.box3d is not None
        label_cam.box3d.location = (
            label_cam.box3d.location[0] + offset,
            label_cam.box3d.location[1],
            label_cam.box3d.location[2],
        )
        labels_cam.append(label_cam)

    return labels_cam


def find_nearest_lidar_frame(
    velodyne_name: str,
    velodyne_dir: str,
    video_name: str,
    frame_idx: int,
    velodyne_names: List[str],
) -> str:
    """Find the nearest lidar frame to handle the missing one."""
    count_last = 0
    last_frame = velodyne_name
    while not last_frame in velodyne_names:
        count_last += 1
        last_frame = osp.join(
            velodyne_dir,
            video_name,
            f"{str(frame_idx-count_last).zfill(6)}.bin",
        )

    count_next = 0
    next_frame = velodyne_name
    while not next_frame in velodyne_names:
        count_next += 1
        next_frame = osp.join(
            velodyne_dir,
            video_name,
            f"{str(frame_idx+count_next).zfill(6)}.bin",
        )

    if count_last <= count_next:
        velodyne_name = last_frame
    else:
        velodyne_name = next_frame

    return velodyne_name


def from_kitti_det(
    data_dir: str,
    data_type: str,
) -> Dataset:
    """Function converting kitti detection data to Scalabel format."""
    frames, groups = [], []

    velodyne_dir = osp.join(data_dir, "velodyne")
    label_dir = osp.join(data_dir, "label_2")
    calib_dir = osp.join(data_dir, "calib")
    ply_dir = osp.join(data_dir, "ply")

    if not osp.exists(ply_dir):
        os.makedirs(ply_dir)

    img_names = sorted(os.listdir(velodyne_dir))

    global_track_id = 0
    i = 0

    rect = None
    glydway_dataset = False

    for frame_idx, velodyne_name in enumerate(img_names):
        i += 1
        if i % 50 == 0:
            print("Progress: ", i, "/", len(img_names))

        img_name = velodyne_name.split(".")[0] + ".png"
        trackid_maps: Dict[str, int] = {}
        frame_names = []
        
        if not osp.exists(calib_dir) :
            if not glydway_dataset:
                print("Warning: Calibration folder not found, assuming Idenity.  If you are using Glydcar dataset, " + \
                    "this is ok.  If you are using KITTI dataset, please download the calibration files from the " + \
                    "KITTI website.")
            glydway_dataset = True
        else:
            glydway_dataset = False
            

        if osp.exists(label_dir):
            label_file = osp.join(
                label_dir, f"{str(frame_idx).zfill(6)}.txt"
            )
            labels_dict, _, _ = parse_label(
                data_type, 
                label_file, 
                trackid_maps, 
                global_track_id, 
                glydway_dataset
            )
            if len(labels_dict) > 0:
                labels = labels_dict[0]
            else:
                labels = []
        else:
            labels = []
        
        
        if not glydway_dataset:
            projections, rect, velo2cam, left_to_right_offset = read_calib(
                calib_dir, int(img_name.split(".")[0]), mode="detection"
            )

            for cam in ["image_2", "image_3"]:
                img_dir = osp.join(data_dir, cam)

                if not osp.exists(img_dir):
                    continue
                with Image.open(osp.join(img_dir, img_name)) as img:
                    width, height = img.size
                    image_size = ImageSize(height=height, width=width)

                offset = 0.0
                if cam == "image_3":
                    offset = left_to_right_offset

                intrinsics = Intrinsics(
                    focal=(projections[cam][0][0], projections[cam][1][1]),
                    center=(projections[cam][0][2], projections[cam][1][2]),
                )

                labels_cam = generate_labels_cam(labels, offset)

                full_path = osp.join(img_dir, img_name)
                url = data_type + full_path.split(data_type)[-1]

                f = FrameGroup(
                    name=f"{cam}_" + img_name,
                    frameIndex=frame_idx,
                    url=osp.join("items", url),
                    size=image_size,
                    intrinsics=intrinsics,
                    labels=labels_cam,
                    frames=[]
                )
                frame_names.append(f"{cam}_" + img_name)
                groups.append(f)


        full_path = osp.join(velodyne_dir, velodyne_name)
        url = data_type + full_path.split(data_type)[-1]

        ply_name = osp.join(
            ply_dir, f"{str(frame_idx).zfill(6)}.ply", 
        )
        # print(full_path)
        if not osp.exists(ply_name):
            open3d.io.write_point_cloud(
                ply_name, convert_kitti_bin_to_ply(full_path), 
                write_ascii=False, compressed=False, print_progress=False
            )

        if rect is not None:
            lidar2cam_mat = np.dot(rect, velo2cam)
            lidar2cam = get_extrinsics_from_matrix(lidar2cam_mat)
        else:
            lidar2cam = get_extrinsics_from_matrix(np.eye(4))
        
        frames.append(
            FrameGroup(
                name=velodyne_name,
                url=osp.join("items", ply_name),
                extrinsics=lidar2cam,
                frames=frame_names,
                labels=parse_lidar_labels(labels, lidar2cam),
            )
        )

    cfg = Config(categories=[Category(name=n) for n in kitti_used_cats])
    dataset = Dataset(frames=frames, groups=groups, config=cfg)
    return dataset


def from_kitti(
    data_dir: str,
    data_type: str,
) -> Dataset:
    """Function converting kitti data to Scalabel format."""
    if data_type == "detection":
        return from_kitti_det(data_dir, data_type)

    frames, groups = [], []

    velodyne_dir = osp.join(data_dir, "velodyne")
    label_dir = osp.join(data_dir, "label_02")
    calib_dir = osp.join(data_dir, "calib")
    oxt_dir = osp.join(data_dir, "oxts")
    ply_dir = osp.join(data_dir, "ply")

    video_names = sorted(os.listdir(velodyne_dir))

    global_track_id = 0
    total_img_names = {}

    for video_name in video_names:
        for cam in ["image_02", "image_03"]:
            img_dir = osp.join(data_dir, cam)
            total_img_names[cam] = sorted(
                [
                    f.path
                    for f in os.scandir(osp.join(img_dir, video_name))
                    if f.is_file() and f.name.endswith("png")
                ]
            )
        velodyne_names = sorted(
            [
                f.path
                for f in os.scandir(osp.join(velodyne_dir, video_name))
                if f.is_file() and f.name.endswith("bin")
            ]
        )
        trackid_maps: Dict[str, int] = {}

        projections, rect, velo2cam, left_to_right_offset = read_calib(
            calib_dir, int(video_name)
        )

        if osp.exists(label_dir):
            label_file = osp.join(label_dir, f"{video_name}.txt")
            labels_dict, trackid_maps, global_track_id = parse_label(
                data_type, label_file, trackid_maps, global_track_id
            )

        i = 0
        for frame_idx in range(len(total_img_names["image_02"])):
            i += 1
            if i % 50 == 0:
                print(i, "/", len(total_img_names["image_02"]))

            fields = read_oxts(oxt_dir, int(video_name))
            poses = [KittiPoseParser(fields[i]) for i in range(len(fields))]

            rotation = tuple(
                R.from_matrix(poses[frame_idx].rotation)
                .as_euler("xyz")
                .tolist()
            )
            position = tuple(
                np.array(
                    poses[frame_idx].position - poses[0].position
                ).tolist()
            )

            frame_names = []
            for cam, img_names in total_img_names.items():
                img_name = img_names[frame_idx]
                with Image.open(img_name) as img:
                    width, height = img.size
                    image_size = ImageSize(height=height, width=width)

                intrinsics = Intrinsics(
                    focal=(projections[cam][0][0], projections[cam][1][1]),
                    center=(projections[cam][0][2], projections[cam][1][2]),
                )

                offset = 0.0
                if cam == "image_03":
                    offset = left_to_right_offset

                cam2global = Extrinsics(
                    location=(position[0] + offset, position[1], position[2]),
                    rotation=rotation,
                )

                if osp.exists(label_dir):
                    if not frame_idx in labels_dict:
                        labels = []
                    else:
                        labels = labels_dict[frame_idx]
                else:
                    labels = []

                url = data_type + img_name.split(data_type)[-1]
                img_name_list = img_name.split(f"{cam}/")[-1].split("/")
                img_name = f"{cam}_" + "_".join(img_name_list)
                labels_cam = generate_labels_cam(labels, offset)

                img_frame = Frame(
                    name=img_name,
                    videoName=video_name,
                    frameIndex=frame_idx,
                    url=url,
                    size=image_size,
                    extrinsics=cam2global,
                    intrinsics=intrinsics,
                    labels=labels_cam,
                )
                frame_names.append(img_name)
                groups.append(img_frame)

            velodyne_name = osp.join(
                velodyne_dir, video_name, f"{str(frame_idx).zfill(6)}.bin"
            )

            ply_name = osp.join(
                ply_dir, video_name, f"{str(frame_idx).zfill(6)}.ply", 
            )

            open3d.io.write_point_cloud(
                ply_name, convert_kitti_bin_to_ply(velodyne_name), 
                write_ascii=False, compressed=False, print_progress=False
            )

            if not velodyne_name in velodyne_names:
                velodyne_name = find_nearest_lidar_frame(
                    velodyne_name,
                    velodyne_dir,
                    video_name,
                    frame_idx,
                    velodyne_names,
                )

            group_name = "_".join(
                velodyne_name.split(velodyne_dir)[-1].split("/")[1:]
            )

            url = data_type + velodyne_name.split(data_type)[-1]

            lidar2cam, lidar2global = get_extrinsics(
                rect, velo2cam, cam2global
            )

            frames.append(
                FrameGroup(
                    name=group_name,
                    videoName=video_name,
                    frameIndex=frame_idx,
                    url=url,
                    extrinsics=lidar2global,
                    frames=frame_names,
                    labels=parse_lidar_labels(labels, lidar2cam),
                )
            )

    cfg = Config(categories=[Category(name=n) for n in kitti_used_cats])
    dataset = Dataset(frames=frames, groups=groups, config=cfg)
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    if args.dir is not None:
        args.input_dir = args.dir
        args.output_dir = args.dir

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = f"{args.data_type}_{args.split}.json"

    data_dir = osp.join(args.input_dir, args.data_type, args.split)

    scalabel = from_kitti(
        data_dir,
        data_type=args.data_type,
    )

    save(osp.join(args.output_dir, output_name), scalabel)


if __name__ == "__main__":
    run(parse_arguments())
