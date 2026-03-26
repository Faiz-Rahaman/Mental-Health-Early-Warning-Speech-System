import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load features and metadata
features = pd.read_csv("output/all_features_crema.csv")
metadata = pd.read_csv("dataset/cremad/crema_metadata.csv")

# Merge
df = pd.merge(features, metadata, on='filename')

# Set up the plot style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# 1. Pitch by emotion
plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='emotion', y='pitch_mean', order=['anger','disgust','fear','happy','neutral','sad'])
plt.title('Pitch Mean by Emotion')
plt.xticks(rotation=45)

# 2. Pitch variability by emotion
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='emotion', y='pitch_std', order=['anger','disgust','fear','happy','neutral','sad'])
plt.title('Pitch Variability by Emotion')
plt.xticks(rotation=45)

# 3. Energy by emotion
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='emotion', y='energy_mean', order=['anger','disgust','fear','happy','neutral','sad'])
plt.title('Energy Mean by Emotion')
plt.xticks(rotation=45)

# 4. Speech rate by emotion
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='emotion', y='speech_rate', order=['anger','disgust','fear','happy','neutral','sad'])
plt.title('Speech Rate by Emotion')
plt.xticks(rotation=45)

# 5. Pause ratio by emotion
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='emotion', y='pause_ratio', order=['anger','disgust','fear','happy','neutral','sad'])
plt.title('Pause Ratio by Emotion')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('output/features_by_emotion.png', dpi=150)
print("✅ Saved features_by_emotion.png")

# Print average values
print("\n📊 Average pitch by emotion:")
print(df.groupby('emotion')['pitch_mean'].mean().sort_values())

print("\n📊 Average energy by emotion:")
print(df.groupby('emotion')['energy_mean'].mean().sort_values())

print("\n📊 Average speech rate by emotion:")
print(df.groupby('emotion')['speech_rate'].mean().sort_values())