#!/usr/bin/env python3
"""
Update model checkpoint with optimal calibrated threshold
"""
import torch
import json

# Load current checkpoint
checkpoint = torch.load('model/deepfake_detector.pth', map_location='cpu', weights_only=True)

# Load calibration results
with open('calibration_results/calibration_results.json') as f:
    cal = json.load(f)

print('Current threshold:', checkpoint.get('best_threshold'))
print('Current F1:', checkpoint.get('best_accuracy'))

# Update with optimal threshold - use Temperature Scaled (better calibration)
optimal_threshold = cal['recommendations']['Temperature Scaled']['threshold']
temperature = cal['calibration']['temperature']

print('Optimal threshold (temp scaled):', optimal_threshold)
print('Temperature:', temperature)

# Update checkpoint
checkpoint['best_threshold'] = optimal_threshold
checkpoint['temperature'] = temperature
checkpoint['calibration_method'] = 'temperature_scaling'
# Store the F1 in a properly-named field (this was previously written into
# 'best_accuracy', which is a misnomer and confused the metric provenance).
f1 = cal['recommendations']['Temperature Scaled']['f1'] * 100
acc = cal['recommendations']['Temperature Scaled']['accuracy'] * 100
checkpoint['best_f1'] = f1
checkpoint['best_accuracy'] = acc

print('New best_f1:', checkpoint['best_f1'])
print('New best_accuracy:', checkpoint['best_accuracy'])

# Save updated checkpoint
torch.save(checkpoint, 'model/deepfake_detector.pth')
print('✅ Checkpoint updated!')