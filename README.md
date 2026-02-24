# AI Security Project

This project demonstrates a model extraction (model stealing) attack against a deployed machine learning API, along with different defense strategies.

## 📌 Project Structure

- `train_model.py`  
  Trains the original machine learning model and saves it as `original_model.pkl`.

- `app.py`  
  Starts the Flask API server and exposes the `/predict` endpoint.

- `attack.py`  
  Performs a **hard-label model extraction attack** (uses only predicted class labels).

- `attack_pro.py`  
  Performs a **soft-label model extraction attack** (uses predicted class probabilities).

`run_all_attacks.py`
Runs both attack.py and attack_pro.py automatically multiple times and calculates the average agreement score for each attack.
This file is recommended for experimental evaluation and comparison.
---


### Configuration

Inside both:

attack.py

attack_pro.py

You can modify:
NUM_QUERIES = 1000

NUM_QUERIES controls:
The number of requests sent to the API

### Installation

pip install -r requirements.txt


## 🚀 How to Run the Project
You need to open two separate terminal windows:

Terminal 1 → Run the API server

Terminal 2 → Run the attack scripts

The API server must be running before you start any attack.

Follow these steps in order


### 1️⃣ Train the Original Model
```bash
python train_model.py
```
### Step 2️⃣: Start the API Server (Terminal 1)

In the first terminal window, run:
python app.py
This starts the Flask server and exposes:
http://localhost:5000/predict
Keep this terminal running.

### Step 3️⃣: Run the Attack (Terminal 2)

Open a second terminal window and choose one of the following:

🔹 Hard-Label Attack
python attack.py
🔹 Soft-Label Attack
python attack_pro.py

or 

Run Both Attacks Automatically
python run_all_attacks.py