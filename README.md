# synthimg4mrcnn
synth image tools for mask-rcnn

We know that mask-rcnn is a SOTA model for both detection and segmentation, and MS-COCO pretrained model works fine for daily projects.
However, for specific projects, the categories in MS-COCO might not be the targets class, and 3 practical problem always exist:
1. usually, we dont have sufficient annotated samples like MS-COCO for certain project;
2. we dont have sufficient computation power to launch training from scratch;
3. MS-COCO might contains only part of our target classes.
To solve these problem, we develop a solution.

This repo provide a synth image and corresponding COCO format json file generator, which we found helps a lot to prevent over-fitting.
source images comes from google image, background images comes from CVPR2019 indoor database, tuning should be needed for paras within this repo in different application scene.

keras based mask-rcnn repo will be uploaded later.
