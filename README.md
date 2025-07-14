# Indian Gesture (or Indian Sign Language a.k.a ISL) Recognition

This project (implemented for my NLP course) implements a hand gesture recognition system using a Convolutional Neural Network (CNN) and OpenCV. It includes tools to create custom gestures, train a model, and recognize gestures in real-time using a webcam. The project report can be found in 'docs/'.

<p align="center">
  <img src="https://github.com/user-attachments/assets/065d695c-888b-4b04-b64c-a0d864e0d59b" alt="ProjectBlockDiagram">
</p>

## Environment Setup

1. Install dependencies: pip install -r requirements.txt

## To set hand histogram
1.  Run: python code/set_hand_hist.py
2.  Place your hand (ideally with a green glove) inside the green boxes in the camera frame.
3.  Press C to view the masked frame. Make sure your hand is clearly visible.
4.  Press S to save the histogram

## Create custom gestures
1. Run: python create_gestures.py
2. Enter a gesture ID (start from 29 if 0–28 are already used) and a name for the gesture.
3. Place your hand in the green box and press C to begin.
4. Around 1200 images will be captured over time (about 1 per second).

## Preprocess the images (and data augmentation)
1. To get the flipped variants: python flip_images.py
2. To split the data into train, test, and validation sets: python load_images.py
3. To view all stored gestures: python display_all_gestures.py

<p align="center">
  <img src="https://github.com/user-attachments/assets/6a0f9de8-ceff-4f10-8699-8fa9bdaaa67b" alt="full_img">
</p>


## Train the model
1. Train the CNN model: python cnn_keras.py
2. To view the training stats / graphs: tensorboard --logdir="logs"
3. Then open the provided URL (e.g., http://127.0.0.1:6006) in your browser.

## Evaluate the model
1. Generate performance reports on the test set: python get_model_reports.py
2. Outputs include: Precision, Recall, F1 Score, Confusion Matrix

## Real-Time Gesture Recognition
1. Run: python recognize_gesture.py

Note: If lighting or glove color changes, you may need to reset the histogram: python set_hand_hist.py
