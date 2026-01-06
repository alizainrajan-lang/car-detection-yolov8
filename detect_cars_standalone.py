"""
Car Detection Script (No OpenCV Dependency - Standalone NMS)

This script manually processes the image and includes a standalone implementation
of Non-Maximum Suppression (NMS) to avoid importing broken dependencies.
"""
import sys
import argparse
import requests
import io
import os
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision

# --- MOCK CV2 ---
class MockCV2:
    def __getattr__(self, name):
        def dummy(*args, **kwargs): return None 
        return dummy
    INTER_LINEAR = 1
sys.modules['cv2'] = MockCV2()

from ultralytics import YOLO

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    return torchvision.ops.box_iou(box1, box2)

def non_max_suppression_standalone(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """
    Standalone NMS implementation suitable for YOLOv8 output
    """
    # YOLOv8 model() returns a list/tuple. The first element is the predictions [batch, 84, 8400]
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 4  # number of classes
    
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Transpose to [8400, 84]
        x = x.transpose(0, -1)[xc[xi]]
        
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Confidence
        conf, j = x[:, 4:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
            
        # Batched NMS
        c = x[:, 5:6] * 7680  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        output[xi] = x[i]
        
    return output

def scale_boxes_standalone(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    
    # clip_boxes(boxes, img0_shape)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return boxes

def detect_cars_pure_func(img_path_or_obj=None):
    """
    Refactored function that returns the PIL Image object with detections drawn.
    Does NOT save to disk by default, returns the object for the specific caller to handle.
    """
    print("Loading YOLOv8 model...", file=sys.stderr)
    model = YOLO('yolov8n.pt')
    
    # Load Image
    if img_path_or_obj is None:
        print("Using sample image...", file=sys.stderr)
        url = "https://ultralytics.com/images/zidane.jpg"
        response = requests.get(url)
        img_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
    elif isinstance(img_path_or_obj, str):
        if not os.path.exists(img_path_or_obj):
            print(f"Error: Image not found at {img_path_or_obj}", file=sys.stderr)
            return None
        img_pil = Image.open(img_path_or_obj).convert("RGB")
    elif isinstance(img_path_or_obj, Image.Image):
        img_pil = img_path_or_obj.convert("RGB")
    else:
        return None

    # Logic to resize maintaining aspect ratio (Letterbox) for Inference
    target_size = 640
    im_w, im_h = img_pil.size
    scale = min(target_size / im_w, target_size / im_h)
    new_w, new_h = int(im_w * scale), int(im_h * scale)
    
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Create canvas 
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas.paste(img_resized, (pad_w, pad_h))
    
    img_np = np.array(canvas)
    
    # Ensure correct memory layout for torch
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() 
    img_tensor /= 255.0 
    img_tensor = img_tensor.unsqueeze(0) 

    # Inference
    results = model.model(img_tensor)
    
    # NMS
    preds = non_max_suppression_standalone(results, conf_thres=0.30, iou_thres=0.45, classes=[2, 5, 7])
    
    det = preds[0]
    
    draw = ImageDraw.Draw(img_pil)
    
    vehicle_count = len(det)
    
    if len(det) > 0:
        # Scale boxes back to original image size
        det[:, :4] = scale_boxes_standalone(img_tensor.shape[2:], det[:, :4], (img_pil.height, img_pil.width))

        for *xyxy, conf, cls in det:
            cords = [float(x) for x in xyxy]
            class_id = int(cls)
            conf = float(conf)
            label_text = f"{model.names[class_id]} {conf:.2f}"
            
            draw.rectangle(cords, outline="#00ff00", width=4) # Neon Green
            try:
                text_bbox = draw.textbbox((cords[0], cords[1]), label_text)
                draw.rectangle(text_bbox, fill="#00ff00")
            except AttributeError:
                 pass
            draw.text((cords[0], cords[1]), label_text, fill="black")
    
    return img_pil, vehicle_count

def detect_cars_pure(image_path=None):
    # Wrapper for CLI
    res, count = detect_cars_pure_func(image_path)
    if res:
        output_filename = "output_detection_pure.jpg"
        res.save(output_filename)
        print(f"Found {count} vehicles.")
        print(f"Success! Output saved to: {os.path.abspath(output_filename)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    args = parser.parse_args()
    detect_cars_pure(args.image)
