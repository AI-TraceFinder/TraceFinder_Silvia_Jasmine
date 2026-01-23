README – Milestone 4
TraceFinder: Model Deployment and Application
Objective

The objective of Milestone 4 is to deploy the trained CNN model as a functional application that allows users to upload a document image and receive the predicted scanner source.

System Architecture
User Interface
      ↓
FastAPI Backend
      ↓
Trained CNN Model

Project Structure
milestone4/
├── backend/
│   ├── app.py
│   ├── best_forensic_cnn.pth
│   └── requirements.txt
├── frontend/
│   └── index.html

Backend Implementation

Developed using FastAPI

Loads trained CNN model at startup

Performs preprocessing and inference

Exposes REST endpoints for prediction

To start the backend server:

cd milestone4/backend
python app.py


The server runs locally at:

http://localhost:8000

Frontend Implementation

Simple HTML-based user interface

Supports image upload

Displays predicted scanner result

Communicates with backend via HTTP requests

Inference Workflow

User uploads a document image

Image is resized and normalized using training configuration

CNN predicts the scanner class

Result is returned to the frontend

Deployment Readiness

Backend tested using Uvicorn

Model loads successfully on CPU

Modular separation of frontend and backend

Ready for demonstration and further deployment

Milestone 4 Outcome

End-to-end functional application completed

Model successfully deployed for inference

Real-time scanner identification achieved

Project ready for final review or demonstration
