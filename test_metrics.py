import tracer
import numpy as np

print(" Loading embeddings and fitting the pipeline (this takes about 2 mins)...")

# Load our synthetic training data assets
X = np.load('tracer-demo-output/embeddings.npy')

# Fit the 90% routing threshold configuration
res = tracer.fit(
    'tracer-demo-output/traces.jsonl', 
    embeddings=X, 
    config=tracer.FitConfig(target_teacher_agreement=0.90)
)

# Look safely inside the successfully calibrated results object
print("\n--- Calibration Engine Metrics ---")
for key, value in res.__dict__.items():
    if key != 'router':  # Skip printing the heavy binary router object wrapper
        print(f" {key}: {value}")