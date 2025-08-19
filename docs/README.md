# Exercise Tracker

**Exercise Tracker** is a Python-based system that uses computer vision to track and evaluate exercise form in real time.  
It can count repetitions (e.g., squats), analyze movement quality, and provide feedback using heuristics and trained ML models.  

The project focuses on the **squat exercise**. Using the [MediaPipe](https://developers.google.com/mediapipe) library, pose landmarks are extracted frame by frame and stored in CSV format. These raw data are then processed and used to train a custom classification model.  

The workflow was:  
1. **Feature Engineering:** Researched which metrics define a "proper" squat (e.g., angles, alignment) and designed functions to compute them.  
2. **Data Collection:** Captured pose data of people performing squats.  
3. **Model Development:** Trained a model on the collected features to distinguish between correct and incorrect squat form.  
4. **Prediction:** Integrated the model into a real-time application that can evaluate each repetition either via a live webcam feed or from prerecorded videos.  

In practice, the application continuously takes pose data during each squat repetition, evaluates the movement, and labels it as **correct** or **incorrect**.  

<img width="572" height="635" alt="Ekran görüntüsü 2025-08-19 160639" src="https://github.com/user-attachments/assets/b1550b71-fa87-465c-b681-06208be1f314" />
<img width="314" height="985" alt="Ekran görüntüsü 2025-08-19 160837" src="https://github.com/user-attachments/assets/d6919f72-2ddc-4173-ba0d-502178d16cb1" />


## Features

- 📹 **Real-time Tracking** – Uses webcam or video files to capture and analyze squat movements.  
- 🧍 **Pose Estimation** – Extracts body landmarks with MediaPipe and processes them into meaningful features (angles, alignments, etc.).  
- 📊 **Data Logging** – Saves raw pose data into CSV files for further analysis or model training.  
- 🤖 **Model-based Evaluation** – Trained ML classifier to decide if each squat rep is correct or incorrect.  
- 🔄 **Rep Counting** – Automatically counts squat repetitions using a state machine (up/down detection).  
- 🎥 **Video Support** – If no camera is available, users can run predictions on sample videos located in `data/raw/`.


## Installation

### Requirements
- Python 3.10 or newer  
- A working webcam (optional, not required if using sample videos)  

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/emirhaan11/exercise-tracker.git
cd exercise-tracker
pip install -r requirements.txt
```

## Usage

You can run the application either with a **webcam** or with **video files**.  
If you don’t have a camera, use the sample clips under `data/raw/`.

### 1) Webcam Inference

```bash
# Start real-time prediction with the default camera (index 0)
python -m main/pipeline_realtime_predict.py

# Predict on a sample video shipped in the repo
python -m main.video_predict --input data/raw/sample_squat.mp4
```

## Project Structure

```bash
exercise-tracker/
├─ core/ # Pose estimator, feature extractors, counters, rules
├─ data/
│ ├─ raw/ # Sample input videos (use these if you have no camera)
│ └─ processed/ # Preprocessed datasets
├─ docs/ # Notes
│ └─ README.md
├─ main/
│ └─ pipeline.py # Pose analyze without ML model
│ ├─ pipeline_realtime_predict.py # Entry point for webcam inference
│ └─ video_predict.py # Entry point for file-based inference
├─ models/
│ └─ squat_clf.pkl # Trained classifier 
├─ notebooks/ # Training notebooks
├─ test/ # Unit tests
├─ requirements.txt # Python dependencies

```
