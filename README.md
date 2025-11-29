# AI Scheduler Enterprise (ML-Powered)

## Overview
This is a robust employee scheduling system powered by a **Hybrid AI Pipeline**. It parses unstructured text requests to automatically generate conflict-free work rosters.

Unlike basic scripts, this project utilizes a **Scikit-Learn Classifier** trained on a dataset of **85+ sentences** to intelligently detect user intent (Availability, Unavailability, Requests, Preferences).

## Key Features
* **Smart Classification:** Uses Naive Bayes ML to understand context (e.g., "I am sick" vs "I am free").
* **Conflict Resolution:** Automatically flags employees who cannot work and explains why.
* **Supply & Demand Scheduling:** Fills requested shifts first, then ensures all other available employees are assigned inferred shifts.
* **Preference Optimization:** Prioritizes employees based on their stated preferences (e.g., "I prefer mornings").

## Dependencies
* Python 3.8+
* `streamlit` (Frontend)
* `spacy` (NLP)
* `scikit-learn` (Machine Learning)
* `pandas` (Data Management)

## Quick Start
1.  **Install Requirements:**
    ```bash
    pip install streamlit spacy scikit-learn pandas
    ```
2.  **Download NLP Data:**
    ```bash
    python -m spacy download en_core_web_sm
    ```
3.  **Run App:**
    ```bash
    streamlit run app.py
    ```

## Project Structure
* `app.py`: Main application code (includes ML training data).
* `Technical_Report.md`: Detailed system architecture and logic explanation.
* `TEST_RESULTS.txt`: Sample run logs proving system functionality.