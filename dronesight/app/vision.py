"""CPU person detection through ONNX Runtime plus a small IoU tracker.

ONNX keeps PyTorch out of the web server, which shrinks the image and the attack surface.
"""
import numpy as np


def letterbox(frame, size):
    """Resize keeping aspect ratio and pad to size x size. Returns (image, scale, pad_x, pad_y)."""
    import cv2

    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def iou(a, b):
    """IoU between box arrays of shape (N, 4) and (M, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms(boxes, scores, threshold=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        best = order[0]
        keep.append(best)
        if order.size == 1:
            break
        overlaps = iou(boxes[[best]], boxes[order[1:]])[0]
        order = order[1:][overlaps <= threshold]
    return np.array(keep, dtype=int)


def decode(output, conf, scale, pad_x, pad_y, frame_w, frame_h):
    """Convert raw YOLO output of shape (1, 4 + classes, N) to normalised frame boxes."""
    pred = output[0].T
    # The first two classes are people on the VisDrone model; a person-only model has just one.
    scores = pred[:, 4:6].max(axis=1)
    pred, scores = pred[scores >= conf], scores[scores >= conf]
    if not len(pred):
        return np.zeros((0, 4)), np.zeros(0)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(boxes, scores)
    boxes, scores = boxes[keep], scores[keep]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale / frame_w, 0, 1)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale / frame_h, 0, 1)
    return boxes, scores


class OnnxDetector:
    def __init__(self, model_path, conf):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.size = model_input.shape[2] if isinstance(model_input.shape[2], int) else 960
        self.conf = conf

    def __call__(self, frame):
        h, w = frame.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(frame, self.size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        return decode(output, self.conf, scale, pad_x, pad_y, w, h)


def closeness(tracked, boxes):
    """Similarity from 0 to 1: centre distance measured in box diagonals.

    People seen from the air are so small that their boxes often stop overlapping between
    sampled frames, which makes IoU useless here.
    """
    centre_a = (tracked[:, None, :2] + tracked[:, None, 2:]) / 2
    centre_b = (boxes[None, :, :2] + boxes[None, :, 2:]) / 2
    distance = np.linalg.norm(centre_a - centre_b, axis=2)
    diagonal = np.linalg.norm(tracked[:, 2:] - tracked[:, :2], axis=1)[:, None]
    return np.clip(1 - distance / np.maximum(diagonal, 1e-6) / 2, 0, 1)


class Tracker:
    """Greedy nearest matching. Tracks survive `max_missed` sampled frames without a match."""

    def __init__(self, threshold=0.25, max_missed=2):
        self.threshold = threshold
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes):
        ids = [0] * len(boxes)
        known = list(self.tracks)
        if known and len(boxes):
            scores = closeness(np.array([self.tracks[k][0] for k in known]), boxes)
            for t, d in sorted(np.ndindex(*scores.shape), key=lambda p: -scores[p]):
                if scores[t, d] < self.threshold:
                    break
                if known[t] is not None and ids[d] == 0:
                    ids[d] = known[t]
                    self.tracks[known[t]] = (boxes[d], 0)
                    known[t] = None
        for k in known:
            if k is not None:
                box, missed = self.tracks[k]
                if missed >= self.max_missed:
                    del self.tracks[k]
                else:
                    self.tracks[k] = (box, missed + 1)
        for d, box in enumerate(boxes):
            if ids[d] == 0:
                ids[d] = self.next_id
                self.tracks[self.next_id] = (box, 0)
                self.next_id += 1
        return ids
