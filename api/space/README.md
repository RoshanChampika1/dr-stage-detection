---
title: DR Stage Detection API
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Diabetic Retinopathy Stage Detection API

FastAPI inference service for a CNN (EfficientNet-B0, transfer learning) that
grades colour fundus photographs into five diabetic retinopathy stages
(0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR) and shows a
Grad-CAM heatmap of the regions behind the decision.

- `GET /health`: status and model information
- `POST /predict`: upload an image as form field `file`
- `GET /app/`: web page
- `GET /docs`: interactive API documentation

Code: https://github.com/RoshanChampika1/dr-stage-detection

**Research prototype only. Not a medical device; not for diagnosis.**
