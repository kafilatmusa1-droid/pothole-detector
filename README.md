---
title: Pothole Detector
emoji: 🛣️
colorFrom: gray
colorTo: orange
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---

# Edge AI Pothole and Road-Defect Detection (Prototype)

Upload a road photograph and the model marks potholes and cracks it finds.
The model is a compressed YOLO26s detector (INT8, 10 MB) trained on RDD2022
and exported to LiteRT. It runs on the free CPU of this Space.

**Please do not upload photographs of people, number plates or private
property.** Uploads are processed by this Space; do not send anything you
would not want stored on a public service.

This is a research prototype. Results are a screening aid and must be
confirmed by a person before any repair decision. It misses some damage,
including damage under standing water, and its accuracy on Nigerian roads
is still under evaluation.

Author: Kafilat Musa, Ahmadu Bello University, Zaria (COEN872 capstone).
