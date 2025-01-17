Note: This repo is now archived and will be read-only
Please use the following repo moving forward
https://github.com/glydways/ext-scalabel 

See PER-164 for details

### Visualizing KITTI formatted labels and point cloud to Scalabel with ext-Complex-YOLOv4-Pytorch repo

## Step 1. Setup Docker 
Follow the instructions in `glyd/ext-Complex-YOLOv4-Pytorch` repo to setup a docker, and launch the docker

## Step 2. Prepare the Data 
For your Yolo repo to function properly with default settings, the folder `kitti` is located at the root of `~/ext-Complex-YOLOv4-Pytorch/dataset`, please make sure the following content exists:
```
kitti
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
└── training
    ├── calib  # calib files in this directory must exist
    ├── label_2  # label files in this directory must exist
    └── velodyne  # velodyne files in this directory must exist
```

However, scalabel somehow can only read data from `~/ext-Complex-YOLOv4-Pytorch/items`, therefore, there are symbolic links in this repository to make it work.  

## Step 3. Convert the Data to Scalabel format:
Launch the `cyolo4:latest` docker, and run the following command inside it:
`python -m scalabel.label.from_kitti --split training --data-type detection --nproc 18 --dir dataset/kitti`

## Step 4. Pull the web server docker from the web
Outside the docker, run:
`docker pull scalabel/www`

## Step 5. (Optional) Setup alias for running the web server docker
`echo "alias ds='docker run -it -v ~/ext-Complex-YOLOv4-Pytorch/dataset:/opt/scalabel/local-data -p 8686:8686 -p 6379:6379 scalabel/www node app/dist/main.js --config /opt/scalabel/local-data/scalabel/config.yml --max-old-space-size=8192'" >> ~/.bashrc`
Note that if you cloned the parent repo to a different directory, change `-v ~/ext-Complex-YOLOv4-Pytorch/dataset:/opt/scalabel/local-data` to `-v <your_directory>:/opt/scalabel/local-data`.

## Step 6. Launch the docker
If you have completed Step 5:
`ds`

Otherwise:
`docker run -it -v ~/ext-Complex-YOLOv4-Pytorch/dataset:/opt/scalabel/local-data -p 8686:8686 -p 6379:6379 scalabel/www node app/dist/main.js --config /opt/scalabel/local-data/scalabel/config.yml --max-old-space-size=8192`

Note that if you cloned the parent repo to a different directory, change `-v ~/ext-Complex-YOLOv4-Pytorch/dataset:/opt/scalabel/local-data` to `-v <your_directory>:/opt/scalabel/local-data`.


## Step 7. Open Scalabel Web UI using a browser
Go to: `http://127.0.0.1:8686/create`

## Step 8. Create a dataset
Item Type: Point Cloud

Label Type: 3D Bounding Box

Check "Submit single file"

For Dataset, select the file we generated in Step 3, which should be at `~/ext-Complex-YOLOv4-Pytorch/dataset/kitti/detection_training.json`.
Set tasksize to something reasonable, (say 100).  

Warning: If task is too small, say 1, it would take a very long time for it to pre-process and you may have to kill the web sever, then delete dataset/items/<project_name>, and restart.

## Step 9. Launch visualizer
Click "Launch Dashboard" then select a row and click on the icon on column "Task Link". 

To view the keyboard controls for the visualizer go here:
<a href="https://github.com/glydways/scalabel/blob/master/doc/src/keyboard.rst">https://github.com/glydways/scalabel/blob/master/doc/src/keyboard.rst</a>
Refer to "Point Cloud Bounding Box" Section





### NOTE

We released a new platform for visual learning from human feedback, called [nutsh](https://github.com/SysCV/nutsh). It is in active development and it covers most of the functionalies of Scalabel with many more features.

<p align="center"><img width=250 src="https://s3-us-west-2.amazonaws.com/scalabel-public/www/logo/scalable_dark.svg" /></p>

---

![Build & Test](https://github.com/scalabel/scalabel/workflows/Build%20&%20Test/badge.svg?branch=master)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/scalabel/scalabel.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scalabel/scalabel/context:javascript)
[![Language grade:
Python](https://img.shields.io/lgtm/grade/python/g/scalabel/scalabel.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scalabel/scalabel/context:python)
![Docker Pulls](https://img.shields.io/docker/pulls/scalabel/www)
![System Support](https://img.shields.io/badge/os-linux%20%7C%20macos-green)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/scalabel/scalabel)
[![PyPI version](https://badge.fury.io/py/scalabel.svg)](https://badge.fury.io/py/scalabel)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scalabel)
![npm](https://img.shields.io/npm/v/scalabel)
![node-lts](https://img.shields.io/node/v-lts/scalabel)
![Redis Version](https://img.shields.io/badge/redis-%3E%3D5-blue)
![npm type definitions](https://img.shields.io/npm/types/scalabel)

[Scalabel](https://www.scalabel.ai) (pronounced "sca&#8901;label") is a versatile and scalable annotation platform, supporting both 2D and 3D data labeling. [BDD100K](https://www.bdd100k.com/) is labeled with this tool.

[**Documentation**](https://doc.scalabel.ai/) |
[**Overview Video**](https://go.yf.io/scalabel-video-demo) |
[**Discussion**](https://groups.google.com/g/scalabel) |
[**Contributors**](https://github.com/scalabel/scalabel/graphs/contributors)

![scalabel interface](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/banner-app.png)

---

### Main features

- [Creating a new annotation project](#creating-a-new-annotation-project)
- [Image tagging](#image-tagging)
- [2D bounding box detection and tracking annotation](#2d-bounding-box-detection-and-tracking-annotation)
- [2D polygon/polyline and tracking annotation](#2d-polygonpolyline-and-tracking-annotation)
- [3D bounding box detection and tracking annotation on point clouds](#3d-bounding-box-detection-and-tracking-annotation-on-point-clouds)
- [Real-time session synchronization for seamless collaboration](#real-time-session-synchronization-for-seamless-collaboration)
- [Semi-automatic annotation with label pre-loading](#semi-automatic-annotation-with-label-pre-loading)
- [Python API for label handling and visualization](#python-api-for-label-handling-and-visualization)

<br>

### Quick Start

Try Scalabel on your local machine

```bash
git clone https://github.com/scalabel/scalabel
cd scalabel

chmod +x scripts/setup_ubuntu.sh scripts/setup_osx.sh scripts/setup_local_dir.sh
# Or run scripts/setup_osx.sh for MacOS
. scripts/setup_ubuntu.sh
. scripts/setup_local_dir.sh
npm run serve
```

Open your browser and go to [http://localhost:8686](http://localhost:8686) to use Scalabel. You can check our project [configuration examples](./examples/) to create some sample projects.

We also provide docker image to avoid installing the libraries. To pull the image:

```bash
docker pull scalabel/www
```

Launch the server through docker

```bash
docker run -it -v "`pwd`/local-data:/opt/scalabel/local-data" -p \
    8686:8686 -p 6379:6379 scalabel/www \
    node app/dist/main.js \
    --config /opt/scalabel/local-data/scalabel/config.yml \
    --max-old-space-size=8192
```

The Python API can be installed through `pip`:

```bash
python3 -m pip install -U scalabel
```

For other operating systems or if you wish to use the Docker image, please refer to the [installation guide](https://doc.scalabel.ai/setup.html).

<br>

### [Creating a new annotation project](https://doc.scalabel.ai/quick-start.html)

- Supporting importing popular data formats such as 2D images and 3D point clouds
- Convenient data uploading using integrated or multiple configuration files for items, categories, and attributes
- Divide a project into multiple tasks using variable task sizes

![Create project](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/project-creation.png)

Creating a project consists of filling in the fields that specify the task, data type, and other parameters such as the task size. Item lists, categories, and attributes must be in the [Scalabel format](https://doc.scalabel.ai/) when uploaded.

<br>

### Image tagging

Images can be tagged with multiple attributes. Categories include weather, scene, and time of day as defaults, but can be freely customised.

![Image tagging](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/tagging.png)

<br>

### [2D bounding box detection and tracking annotation](https://doc.scalabel.ai/2d-bb.html)

- Simple click-and-drag area selection
- Group boxes into a wide range of categories
- Provides extra configurable options such as occlusion, truncation, and traffic light colours
- Tracking between keyframes

![2D bounding box](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/2d_bbox.gif)

Create bounding boxes by selecting an area on the canvas. Bounding boxes can be freely adjusted and moved around. Categories and attributes of the bounding boxes can be customised. Bounding boxes can be linked between frames if the object disappears and reappears in subsequent frames. Linked bounding boxes are colour-coded to indicate the link.

![2D bounding box tracking](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/2d_bbox_tracking.gif)

Bounding boxes are interpolated between keyframes if the position, orientation, or scale of the bounding boxes differ. This is useful for tracking objects that move between keyframes.

<br>

### [2D polygon/polyline and tracking annotation](https://doc.scalabel.ai/instance-segmentation.html)

- Choosing between closed paths for image segmentation or open paths for lane marking
- Supporting bezier curves for precise annotation for round objects
- Tracking interpolation between keyframes

![2D polygon](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/2d_seg.gif)

Click on multiple points on the canvas to generate vertices of a polygon. Click on the first vertex to close the polygon. Vertices can be moved around, and new vertices can be added by clicking on the midpoint of the line segment between two vertices. Creating bezier curves is also supported for smoother labeling of curved objects.

![2D polyline](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/lane_marking.gif)

Polylines can be created for lane marking. They support the same functions as polygons, but do not have to be closed.

![2D polygon tracking](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/2d_seg_tracking.gif)

Polygons are interpolated between keyframes if the position, orientation, or scale of the polygons differ. This is useful for tracking objects that move between keyframes.

<br>

### [3D bounding box detection and tracking annotation on point clouds](https://doc.scalabel.ai/3d-bb.html)

- Multi-sensor view for side-by-side comparison with corresponding 2D images
- Simple four-point click to generate 3D bounding boxes
- Supporting undo and panning functions during annotation for added precision
- Tracking interpolation between keyframes

![3D bounding box](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/3d_bbox.gif)

Click on the canvas to define the length, breadth, and height of the 3D bounding box. There is an in-built ground plane prediction that aligns the 3D bounding box with the ground. Bounding boxes can be freely adjusted and moved around. Categories and attributes of the bounding boxes can be customised. Bounding boxes can be linked between frames if the object disappears and reappears in subsequent frames. Linked bounding boxes are colour-coded to indicate the link.

![3D bounding box tracking](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/3d_bbox_tracking.gif)

Bounding boxes are interpolated between keyframes if the position, orientation, or scale of the bounding boxes differ. This is useful for tracking objects that move between keyframes.

<br>

### Real-time session synchronization for seamless collaboration

Multiple sessions can be initialised by opening new windows or tabs. Each session synchronises its changes in labels and tags with the other sessions in real-time. Tracking changes are also updated in real-time as well.

![Session synchronisation](https://raw.githubusercontent.com/scalabel/scalabel-doc-media/main/readme/sync.gif)

<br>

### [Semi-automatic annotation with label pre-loading](https://doc.scalabel.ai/auto-label.html)

Deep learning models can be used to assist annotation for large batches of data. New models can be trained on a subset of the data, and the remaining data can be uploaded for Scalabel to label automatically. The labels can be preloaded in the backend and can also be manually adjusted in the interface.

<br>

### [Python API for label handling and visualization](https://doc.scalabel.ai/tools.html)

- Providing convenience [scripts](https://doc.scalabel.ai/label.html) for converting from and to other popular formats, such as COCO, KITTI, Waymo
- Supporting [evaluation](https://doc.scalabel.ai/eval.html) of various tasks with the Scalabel format
  - Image tagging
  - Detection
  - Pose estimation
  - Instance segmentation
  - Semantic segmentation
  - Panoptic segmentation
  - Boundary detection
  - Bounding box tracking
  - Segmentation tracking
- Contains a [visualizer](https://doc.scalabel.ai/visual.html) to easily visualize annotations as well as model predictions in Scalabel format

Python backend provides a convenient API for label handling and visualization. For compatibility with Scalabel's interface, datasets can be converted to the appropriate formats via scripts. Algorithms can be evaluated for each task by uploading the predictions and corresponding ground truth annotations in the Scalabel format. The labels can be easily visualised using the Python interface.

<br>

### Contributors

Scalabel is currently supported by [ETH VIS Group](https://www.vis.xyz/). [Many contributors](https://github.com/scalabel/scalabel/graphs/contributors) have contributed to the project.
