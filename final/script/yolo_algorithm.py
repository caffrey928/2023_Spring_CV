import numpy as np

def yolo_algorithm(model , image ):
    results = model.predict(image,stream=True , iou = 0.5 )  # list of Results objects
    conf = 0.0
    output = np.zeros((480, 640))
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        if len(boxes.conf) == 0:
            return output, conf
        elif len(boxes.conf) == 1 :
            output, conf =  masks.data[0].cpu().numpy(), boxes.conf[0].cpu().item()
        else: 
            output, conf =  output + masks.data[0].cpu().numpy(), np.mean(result.boxes.conf.numpy())

    return output, conf

# yolo_model_path = 'best.pt'
# image_path =  'test.jpg'

# model = YOLO(yolo_model_path)  # load a custom model
# output, conf = yolo_algorithm(model , image_path)
# print(output, conf)