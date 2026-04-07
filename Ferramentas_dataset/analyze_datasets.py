import pandas as pd
import numpy as np

# Load datasets
train_df = pd.read_csv('dataset_final.csv', sep=';')
test_df = pd.read_csv('Subm2/subm2_labels_revealed.csv', sep=';')

# Analyze text characteristics
print('=== TEXT LENGTH ANALYSIS ===')
train_df['text_length'] = train_df['Text'].str.len()
test_df['text_length'] = test_df['Text'].str.len()

print(f'Train text length - Mean: {train_df["text_length"].mean():.1f}, Std: {train_df["text_length"].std():.1f}')
print(f'Test text length - Mean: {test_df["text_length"].mean():.1f}, Std: {test_df["text_length"].std():.1f}')

print(f'Train length range: {train_df["text_length"].min()} - {train_df["text_length"].max()}')
print(f'Test length range: {test_df["text_length"].min()} - {test_df["text_length"].max()}')

# Check for any patterns in misclassifications
print('\n=== MISCLASSIFICATION ANALYSIS ===')
# Load predictions
pred_df = pd.read_csv('Subm2/subm2-g6-MIA-A.csv', sep=';')
merged = pd.merge(test_df, pred_df, on='ID', suffixes=('_true', '_pred'))

# Find misclassifications
misclassified = merged[merged['Label_true'] != merged['Label_pred']]
print(f'Total misclassifications: {len(misclassified)}/{len(merged)} ({len(misclassified)/len(merged)*100:.1f}%)')

print('Misclassification breakdown by true label:')
for label in ['Human', 'OpenAI', 'Google', 'Meta', 'Anthropic']:
    subset = misclassified[misclassified['Label_true'] == label]
    print(f'  {label}: {len(subset)} misclassified')

print('Misclassification breakdown by predicted label:')
for label in ['Human', 'OpenAI', 'Google', 'Meta', 'Anthropic']:
    subset = misclassified[misclassified['Label_pred'] == label]
    print(f'  {label}: {len(subset)} predicted (but wrong)')