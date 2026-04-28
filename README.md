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

