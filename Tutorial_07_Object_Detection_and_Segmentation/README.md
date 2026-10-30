# Tutorial 07 Object Detection and Instance Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_07_Object_Detection_and_Segmentation/tutorial_07_pytorch.ipynb)

A self-contained PyTorch-backed YOLO11 lesson. Setup runs inside Colab; no local installation, API key, repository clone, or Drive mount is needed.

- Compare the original image, detection boxes, instance masks, and a binary mask.
- Explore confidence and actual NMS IoU controls, including a hand-checkable toy example.
- Export/reload CSV boxes and NPZ masks; handle empty results safely.
- Inspect YOLO annotations/YAML, run five training epochs on COCO128, and reload the resulting detector.

Defaults: CPU supported, image size 320, `RUN_TRAINING=True`, five epochs, ten training/four validation images. Set `RUN_TRAINING=False` to skip training. Two nano weights total approximately 12 MB; COCO128 is approximately 7 MB. The pinned Ultralytics package installs automatically. Outputs go to `outputs/tutorial_07/`.

COCO128 is a **pipeline smoke test**, not an independent benchmark: its images come from COCO training data used for pretraining. No detection accuracy is inferred from unlabelled example images. Larger/GPU experiments require separate verification.

[Validation evidence](../docs/validation.md) · [Series index](../README.md). Attribution and separate software/model/data terms are linked in the notebook.
