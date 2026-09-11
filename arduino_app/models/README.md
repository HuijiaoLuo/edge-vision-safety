# Runtime Models

Place runtime ONNX files here before deployment.

Required files:

```text
helmet_classifier.onnx
face_recognition.onnx
face_detection_yunet.onnx
```

`helmet_classifier.onnx` is exported from the fine-tuned helmet classifier.

The face models can be downloaded from OpenCV Zoo:

- SFace: `models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
- YuNet: `models/face_detection_yunet/face_detection_yunet_2023mar.onnx`

Model binaries are not committed to this repository.
