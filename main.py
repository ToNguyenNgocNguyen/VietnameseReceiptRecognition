from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import defaultdict
from request_ocr import inference_ocr
from io import BytesIO

# Load a model
model = YOLO("models/detect.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model("image.png", imgsz=640, conf=0.25)  # return a list of Results objects

# Open image using PIL
img = Image.open("image.png")
# Dictionary to store images grouped by their labels
grouped_images = defaultdict(list)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    labels = result.names  # Get the label names
    for box in boxes:
        print("2222", box)
        # Get coordinates of the box (xmin, ymin, xmax, ymax) and move to CPU if necessary
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()  # Move tensor to CPU and convert to numpy array
        label = labels[int(box.cls[0].cpu().numpy())]  # Get label based on index
        print("1111 ", box.cls)

        # Crop based on box coordinates
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        
        # Group cropped images by label
        grouped_images[label].append(cropped_img)

# Resize and concatenate images for each label group
print(grouped_images)
for label, images in grouped_images.items():
    if images:
        # Get the size of the first image in the group
        target_size = images[0].size

        # Resize all images in the group to match the target size
        resized_images = [img.resize(target_size) for img in images]

        # Concatenate images vertically (or horizontally) for the current label
        concatenated_image = np.vstack([np.array(img) for img in resized_images])

        # Convert numpy array back to a PIL Image
        concatenated_image = Image.fromarray(concatenated_image)

        # Save the concatenated image for the label
        concatenated_image.save(f"results/{label}.jpg")

        # Convert the image to bytes
        buffer = BytesIO()
        concatenated_image.save(buffer, format="JPEG")  # Specify the desired format (e.g., JPEG, PNG)
        image_bytes = buffer.getvalue()


        url = 'https://3b45-34-147-94-61.ngrok-free.app/api/inference'
        headers = {
            'accept': 'application/json',
        }
        prompt = f'Hãy trích xuất {label} và trả về theo theo dạng JSON.\n\n'
        prompt += f'\{label}: kết quả trích xuất.'

        response = inference_ocr(image_bytes, url, headers, prompt)

        with open(f"results/{label}.txt", "w") as f:
            f.write(str(response))
