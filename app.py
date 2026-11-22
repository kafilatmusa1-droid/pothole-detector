"""
pothole_app.py
Prototype road-damage detector with a web interface.

What it does:
    Loads a compressed YOLO model exported to LiteRT (TensorFlow Lite),
    accepts a road photograph through a browser page, runs one forward pass,
    draws boxes around detected damage, and reports the counts, the highest
    class score and the inference time.

How to run:
    python pothole_app.py
    then open http://127.0.0.1:7860 in a browser.

Requirements:
    pip install ai-edge-litert numpy pillow gradio
"""

import time
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from ai_edge_litert.interpreter import Interpreter

# ---- Settings -------------------------------------------------------------
MODEL = "best_baseline_w8a32.tflite"   # compressed INT8 model in this folder
CLASSES = ["D00 longitudinal crack", "D10 transverse crack",
           "D20 alligator crack", "D40 pothole"]
SIZE = 640            # the network expects 640 x 640 input
CONF_THRESH = 0.05    # low value used for diagnosis; raise for deployment
IOU_THRESH = 0.45     # overlap above which duplicate boxes are removed

# ---- Load the model once at start-up ---------------------------------------
interp = Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]     # where to write the input image
out = interp.get_output_details()[0]    # where to read the predictions


def letterbox(img):
    """Scale the photo to fit SIZE x SIZE without distorting it, then pad.

    Returns the padded canvas, the scale factor r, and the padding offsets,
    which are needed to map boxes back onto the original photo.
    """
    w, h = img.size
    r = min(SIZE / w, SIZE / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    padx, pady = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas.paste(resized, (padx, pady))
    return canvas, r, padx, pady


def nms(boxes, scores, iou_thresh):
    """Non-maximum suppression: keep the best box where boxes overlap heavily."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_thresh]   # drop boxes that overlap the kept one
    return keep


def detect(pil_img):
    """Run the full pipeline on one uploaded photograph."""
    if pil_img is None:
        return None, "Please upload a road image first."

    img = pil_img.convert("RGB")
    canvas, r, padx, pady = letterbox(img)

    # Normalise pixels to 0-1 and reorder to channels-first, as the model expects.
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]

    # One forward pass, timed.
    t0 = time.perf_counter()
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    ms = (time.perf_counter() - t0) * 1000

    # The output is 8 values for each of 8,400 candidates:
    # rows 0-3 are the box, rows 4-7 are the four class scores.
    pred = interp.get_tensor(out["index"])[0]
    boxes_xywh = pred[:4].T
    class_scores = pred[4:].T
    labels_all = class_scores.argmax(1)
    conf_all = class_scores.max(1)
    best_score = float(conf_all.max())

    # Keep candidates above the threshold.
    m = conf_all > CONF_THRESH
    cx, cy, bw, bh = boxes_xywh[m].T

    # Heuristic: the exported model returns box values as fractions of the image
    # for these candidates. Scale them to pixels. Replace with a check against the
    # documented output format in a future version.
    if cx.size and cx.max() <= 1.5:
        cx, cy, bw, bh = cx * SIZE, cy * SIZE, bw * SIZE, bh * SIZE

    # Convert centre-width-height into corner coordinates.
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    scores = conf_all[m]
    labels = labels_all[m]

    draw_img = img.copy()
    d = ImageDraw.Draw(draw_img)
    W, H = img.size
    found = {c: 0 for c in range(4)}

    # Apply non-maximum suppression separately for each class.
    for c in range(4):
        sel = labels == c
        if not sel.any():
            continue
        cb, cs = boxes[sel], scores[sel]
        for k in nms(cb, cs, IOU_THRESH):
            b, s = cb[k], cs[k]
            # Map the box from the padded canvas back to the original photo.
            x1 = np.clip((b[0] - padx) / r, 0, W)
            y1 = np.clip((b[1] - pady) / r, 0, H)
            x2 = np.clip((b[2] - padx) / r, 0, W)
            y2 = np.clip((b[3] - pady) / r, 0, H)
            d.rectangle([x1, y1, x2, y2], outline="red", width=3)
            d.text((x1, max(0, y1 - 12)), f"{CLASSES[c]} {s:.2f}", fill="red")
            found[c] += 1

    total = sum(found.values())
    summary = (f"Detections: {total}\n"
               f"Highest class score: {best_score:.3f}\n"
               f"Potholes (D40): {found[3]}\n"
               f"Cracks (D00/D10/D20): {found[0] + found[1] + found[2]}\n"
               f"Model inference time: {ms:.0f} ms")
    if total == 0:
        summary += "\nNo road damage detected above the confidence threshold."
    return draw_img, summary


# ---- Web interface ----------------------------------------------------------
demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="Upload a road photo"),
    outputs=[gr.Image(label="Detected road damage"), gr.Textbox(label="Result", lines=6)],
    title="Edge AI Pothole and Road-Defect Detection (Prototype)",
    description="Compressed YOLO model (INT8, 10 MB) running on this laptop's CPU.",
)

if __name__ == "__main__":
    demo.launch()
