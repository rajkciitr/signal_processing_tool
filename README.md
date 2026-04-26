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
