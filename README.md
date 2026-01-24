# Smart-Traffic-Signal
Smart Pedestrian-Aware Traffic Signal System
Project Description

This project presents an intelligent traffic signal system designed to optimize pedestrian crossing intervals using real-time computer vision analysis. The system utilizes a YOLOv8 deep learning model to detect and count pedestrians from uploaded images or video footage. Based on the detected pedestrian density, the application dynamically adjusts red-light duration to ensure safer crossings and improved traffic efficiency.

The solution is implemented as an interactive Streamlit web application that allows users to upload visual data, specify road width, and define average walking speed. The application processes the visual input, calculates the optimal crossing time, and simulates the resulting signal cycle using a clear visual interface. The adaptive timing algorithm ensures that the total signal duration remains within municipal safety standards, with a default cap to prevent excessive delays in vehicular flow.

Key Differentiating Aspect

Unlike conventional traffic signal systems that rely solely on fixed timers or generic motion sensors, this system uses advanced object detection to identify and count only human pedestrians. It can differentiate humans from animals, vehicles, and other irrelevant entities, ensuring that signal adjustments are based on actual pedestrian presence rather than generic motion or false triggers. This significantly enhances reliability and context-aware traffic control.

Project Contributors / Team Members

Participant 1: Kartik Jha

Participant 2: Subodh Koli

Participant 3: Mayuresh Marathe

Participant 4: Harish Mudaliar


Key Objectives :

1)Enhance pedestrian safety at crossings.

2)Reduce unnecessary vehicular idle time using adaptive signaling.

3)Demonstrate the role of AI in intelligent traffic management.

Core System Features :

1)Real-time pedestrian detection using YOLOv8.

2)Ability to differentiate humans from animals, vehicles, and other objects.

3)Support for both image and video inputs.

4)Adjustable parameters including road width and walking speed.

5)Automatic timing calculation based on pedestrian count.

6)Dynamic simulation of signal phases with clear visual feedback.

Potential Real-World Applications :

Urban traffic signal planning

Smart city infrastructure

Pedestrian-dense zones (schools, markets, hospitals)

Future IoT-based adaptive traffic systems

Skills and Technologies Demonstrated :

Computer Vision and Object Detection

Python Programming

Streamlit Framework Development

Image and Video Processing

Applied Machine Learning Model Integration

UI/UX Design for Simulation
