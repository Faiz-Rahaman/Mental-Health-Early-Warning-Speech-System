# Teacher Q&A Defense Sheet 

During your presentation, professors and examiners will likely try to test how deeply you understand the code, the data, and the architecture you built. Here are the trickiest questions you can expect and exactly how to answer them.

---

### Question 1: "Why did you use SMOTE? Isn't that just faking data?"
**Your Defense:** 
"In real clinical datasets like DAIC-WOZ, naturally, there are far fewer people classified as 'High Risk' (severely depressed) than there are 'Medium Risk' or healthy individuals. If we train an AI purely on that unequal data, the model develops *Majority Class Bias*—meaning it gets a high accuracy just by blindly guessing 'Medium Risk' every time, but completely fails to detect actual severely depressed cases.

By using **SMOTE** *(Synthetic Minority Over-sampling Technique)*, we aren't just duplicating data; the algorithm actually calculates the mathematical distances (K-Nearest Neighbors) between existing High-Risk patients and interpolates new data points. This forces the model to learn the exact *boundaries* of what depression sounds like, vastly improving our Recall for the minority class without needing to collect thousands of extra clinical interviews."

---

### Question 2: "If you are already using Gemini (an LLM), what is the point of training your own local Gradient Boosting Machine Learning model?"
**Your Defense:** 
"Large Language Models like Gemini are brilliant at reasoning, providing context, and generating empathetic responses, but they are not built to ingest rigid arrays of mathematical physics. 

We trained our own local **Gradient Boosting Classifier** to act as a *Specialist Sub-Agent*. Our local model exists purely to analyze the raw vocal numbers—like the 13 MFCC fingerprint arrays, the Harmonics-to-Noise Ratio, and the Zero Crossing Rates. It computes a precise statistical probability (e.g., 88% chance of High Risk). We take that *hard probability* and inject it dynamically into Gemini's prompt. So Gemini isn't doing the math; it's using the result of our local ML model's math to generate the comprehensive human-readable report on the dashboard."

---

### Question 3: "Why did you switch from your initial CREMA-D dataset to DAIC-WOZ?"
**Your Defense:** 
"CREMA-D is a highly respected dataset, but it is an *actor-based emotional dataset*. People in that dataset are explicitly pretending to be sad, angry, or happy. 

A Mental Health Early Warning System requires genuine physiological distress markers. The **DAIC-WOZ dataset** consists of real clinical psychiatric interviews. A person masking true depression sounds vastly different from an actor loudly faking sadness. By switching to DAIC-WOZ, our AI is trained to listen for the subtle, involuntary micro-fluctuations in speech rate and high-frequency spectral rolloffs that physically manifest during actual fatigue and depression, making our project clinically relevant rather than just an emotion detector."

---

### Question 4: "Extracting features from 15-minute voice recordings takes a ton of processing power. How does your script manage that?"
**Your Defense:** 
"Audio analysis with the `librosa` library is notoriously single-threaded and CPU intensive. For 16 long participants, sequentially processing them took over an hour, which is unscalable. 

To solve this, we implemented **Multiprocessing via the `joblib` library**. We completely rewrote the `extract_daic_woz.py` script to bypass the Python GIL limitation, slicing the audio into 5-second segments and deploying those calculations across 100% of our laptop’s CPU cores simultaneously (`n_jobs=-1`). This parallelization dropped the feature extraction time from multiple hours down to just a few minutes."

---

### Question 5: "What are MFCCs and why did you use 13 of them?"
**Your Defense:** 
"MFCCs *(Mel-Frequency Cepstral Coefficients)* are essentially the mathematical shape of the voice. They represent the vocal tract's physical shape—your tongue, teeth, throat, and nasal cavity. 

When someone is depressed, their muscle tension drops, which changes the micro-shape of their vocal tract. The first 13 MFCCs capture exactly that—the lowest level energy shapes of the human voice, filtering out background noise and non-human sounds. It gives our Gradient Boosting model a literal fingerprint of the speaker's physiology to base its diagnosis on."
