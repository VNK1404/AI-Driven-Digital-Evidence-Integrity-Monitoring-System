"""
forgery_detection
=================
Image Forgery Detection integration package.

Quick start
-----------
from forgery_detection.core.predictor import ForgeryPredictor
from PIL import Image

predictor = ForgeryPredictor()
result = predictor.predict(Image.open("test.jpg"))
# {'label': 'tampered', 'confidence': 0.92, 'scores': {...}}
"""
