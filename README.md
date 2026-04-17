# MH-EWSS: Mental Health Early Warning Speech System

A powerful, multimodal AI pipeline designed to detect early indicators of depression and mental fatigue directly from human speech. Built with LangGraph, it synthesizes acoustic physics, linguistic patterns, and chronological baseline shifts through a custom-trained Machine Learning model and the Gemini 1.5 Flash API.

## 🚀 Key Features
1. **Clinical Machine Learning**: Trained strictly on the **DAIC-WOZ** clinical dataset (real psychiatric interviews). 
2. **Advanced Feature Engineering**: Employs `Librosa` to extract over 20 vocal markers per 5-second chunk—including 13 MFCCs (Mel-Frequency Cepstral Coefficients), Pitch, Energy Variability, and Harmonics-to-Noise Ratio (HNR).
3. **SMOTE Data Balancing**: The Gradient Boosting Classifier is trained on a highly refined matrix that algorithms mathematically synthesize `High Risk` samples so the model treats severe distress equally to neutral speech.
4. **LangGraph Pipeline Orchestration**: 
   - **Prosody Agent:** Transcribes structural vocal physics.
   - **Language Agent:** Transcribes voice-to-text to perform vocabulary and sentiment mapping using `TextBlob`.
   - **Temporal Agent:** Logs daily/weekly inputs into a fast `SQLite` database to generate a unique "Personal Baseline" for each user.
   - **ML/LLM Decision Agent:** Combines raw probabilities from the local Gradient Booster with context-aware logic from Gemini.
5. **Interactive UI**: A sleek `Streamlit` dashboard providing live analysis, historical risk charts, and custom emergency actions.

## 🛠️ Architecture Workflow
```text
           [Audio Upload] ───────────────> (DAIC-WOZ Trained Gradient Booster)
                  │                                         │
                  │---> (Prosody Agent)                     │
                  │           │                             V
                  │           V                    [Hybrid Confidence]
                  │---> (Language Agent)                    │
                              │                             V
                              ▼                      (Gemini LLM) -> JSON 
                      (Temporal Agent)                      │
                      [SQLite Baseline] <-------------------│
                                                            ▼
                                                   [Streamlit Dashboard]
```
<img width="708" height="695" alt="Screenshot 2026-03-25 192720" src="https://github.com/user-attachments/assets/7d2f237e-0816-4814-b613-007ad804eb93" />


<img width="1919" height="872" alt="Screenshot 2026-03-26 084838" src="https://github.com/user-attachments/assets/2d29a5a5-1f2f-4802-a0c4-a3bc7e60d158" />
<img width="1919" height="872" alt="Screenshot 2026-03-26 084838" src="https://github.com/user-attachments/assets/54af6dd5-b57b-40fd-be36-896088ece70c" />
<img width="1919" height="871" alt="Screenshot 2026-03-25 205403" src="https://github.com/user-attachments/assets/046f6f74-ba99-44ee-824f-ab025414dc2c" />
<img width="1919" height="862" alt="Screenshot 2026-03-25 205416" src="https://github.com/user-attachments/assets/bff1a2b5-ecf4-4728-ba53-c738fcfc0827" />



## ⚙️ Installation & Setup

1. **Clone & Install Dependencies:**
   ```bash
   pip install librosa pandas numpy speechrecognition textblob langgraph google-generativeai python-dotenv streamlit sqlite3 imbalanced-learn joblib scikit-learn
   ```
2. **Environment Setup:**
   Create a `.env` file in the main folder and add your Gemini API Key:
   ```env
   GEMINI_API_KEY="YOUR_API_KEY_HERE"
   ```
3. **Re-Train Model (Optional):**
   ```bash
   # Extract Features using Multiprocessing (joblib)
   python source_codes/extract_daic_woz.py
   
   # Train using SMOTE and Gradient Boosting
   python source_codes/train_daic_woz_model.py
   ```
4. **Launch the Dashboard:**
   ```bash
   streamlit run source_codes/app_dashboard.py
   ```

## 📁 Project Structure 
*   `source_codes/extract_daic_woz.py`: Multiprocessing script that slices large clinical interviews to extract MFCCs and basic prosodic features.
*   `source_codes/train_daic_woz_model.py`: Generates the `.pkl` ML model utilizing SMOTE to balance the clinical data.
*   `source_codes/langgraph_workflow.py`: The brain of the operation, piping data securely between the acoustic math, linguistic parsing, and the Gemini API.
*   `source_codes/app_dashboard.py`: The frontend dashboard with plotly visualizations.

## ⚠️ Medical Disclaimer
This software is an AI prototype intended to highlight changes in vocal biomarkers as a **warning system**. It is NOT a medical diagnostic tool. If you or someone you know is in crisis, consult a medical professional immediately.


