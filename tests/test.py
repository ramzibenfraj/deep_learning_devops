import os
import json
import unittest
from io import BytesIO
from app import app


class FlaskTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_homepage(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Home Page', response.data)

    def test_upload_image(self):
        # Test uploading an image
        with open('test_image.jpg', 'rb') as f:
            image_data = f.read()
        response = self.app.post('/upload', data={'file': (BytesIO(image_data), 'test_image.jpg')}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction', response.data)

    def test_predict(self):
        # Test prediction
        with open('test_image.jpg', 'rb') as f:
            image_data = f.read()
        response_upload = self.app.post('/upload', data={'file': (BytesIO(image_data), 'test_image.jpg')}, content_type='multipart/form-data')
        self.assertEqual(response_upload.status_code, 200)
        response_predict = self.app.post('/predict', data={'filename': 'test_image.jpg'})
        self.assertEqual(response_predict.status_code, 200)
        self.assertIn(b'Prediction Result', response_predict.data)

    def test_last_prediction(self):
        # Test retrieving the last prediction
        response = self.app.get('/last_prediction')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Last Prediction', response.data)

    def test_all_predictions(self):
        # Test retrieving all predictions
        response = self.app.get('/all_predictions')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'All Predictions', response.data)


if __name__ == '__main__':
    unittest.main()
