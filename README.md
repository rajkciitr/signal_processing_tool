# signal_processing_tool
Codes to learn basic application of signal (speech, image etc.) processing techniques for real applications

Useful links:
https://github.com/SuperKogito/spafe

# source for Android app running onnxruntime-genai model
https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md

# Steps to convert tf model to tflite
```{python}

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

# HAR Papers
[1] µBi-ConvLSTM: An Ultra-Lightweight Efficient Model for Human Activity Recognition on Resource Constrained Devices, 2026
https://arxiv.org/pdf/2602.06523
[2] Deep convolutional state space model as human activity recognizer, 2025, https://www.sciencedirect.com/science/article/abs/pii/S1566253525010449
[3] Activity Recognition Using a Multi-head Convolution Neural Network Integrated with Attention Mechanism, 2026 (find file attached above).
[4] DeepConvContext: A Multi-Scale Approach to Timeseries Classification in Human Activity Recognition, 2025, https://arxiv.org/abs/2505.20894

# Pitch
https://brookemosby.github.io/Signal_Analysis/_modules/Signal_Analysis/features/signal.html

Wearable energy harvesters generating electricity from low-frequency human limb movement
https://www.nature.com/articles/s41378-018-0024-3

An ultraflexible energy harvesting-storage system for wearable applications
https://www.nature.com/articles/s41467-024-50894-w

Flexible self-charging power sources
https://www.nature.com/articles/s41578-022-00441-0?fromPaywallRec=false

# Code to draw box
```python
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os

image_path = r"C:\Users\Raj Kumar\Desktop\amex\screenshot"
anno_path = r"C:\Users\Raj Kumar\Desktop\amex\element_ano"



from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_boxes(image_path, boxes, labels, output_path):
    img = Image.open(image_path)
    width, height = img.size

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Make axes fill the entire figure
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", pad=2),
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # keep image coordinate system
    ax.axis("off")

    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor="none",
        edgecolor="none",
        pad_inches=0,
    )
    plt.close(fig)
    
import json

json_file = os.listdir(anno_path)[0]
json_path = os.path.join(anno_path, json_file)
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

boxes = []
labels = []

for element in data["clickable_elements"]:
    boxes.append(element["bbox"])  # [x1, y1, x2, y2]

    # xml_desc is a list, e.g. ["click"]
    label = " ".join(element.get("xml_desc", []))
    labels.append(label)

print(boxes)
print(labels)
# Example
image_file = json_file[:-5] + ".png"  # Assuming the image has the same name as the JSON but with .png extension
image_path2 = os.path.join(image_path, image_file)
output_path = os.path.join(image_path, "annotated_" + image_file)    
print(image_file)   
draw_boxes(
    image_path2,
    boxes,
    labels,
    output_path
)
```
REVISITING MULTIMODAL POSITIONAL ENCODING IN
VISION–LANGUAGE MODELS
https://arxiv.org/pdf/2510.23095
IMPROVING GUI GROUNDING WITH EXPLICIT
POSITION-TO-COORDINATE MAPPING
https://arxiv.org/pdf/2510.03230
https://www.emergentmind.com/topics/interleaved-mrope
