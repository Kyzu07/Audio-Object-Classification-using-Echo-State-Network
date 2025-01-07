# Audio Classification with Echo State Networks (ESN)

This project implements an audio classification pipeline to classify the sounds of everyday objects in our surroundings. It utilizes Echo State Networks (ESN) and various machine learning techniques to process, analyze, and classify audio data effectively.

## Features
- **Audio Preprocessing**: Converts audio signals into Mel spectrograms for feature extraction.
- **Echo State Network**: Implements a reservoir-based approach for capturing temporal dynamics in audio.
- **Classification**: Trains and evaluates a classifier using Ridge Regression for embedding representations.
- **Metrics and Visualization**: Provides detailed analysis using accuracy, F1-scores, confusion matrix, and class-wise metrics.

## Project Structure
```
├── data/                  # Dataset folder (audio files)
├── notebooks/             # Jupyter notebooks for development and experiments
├── src/                   # Source code for the project
│   ├── preprocessing.py   # Audio preprocessing pipeline
│   ├── esn.py             # Echo State Network implementation
│   ├── utils.py           # Utility functions
├── results/               # Output files, metrics, and analysis results
├── README.md              # Project documentation
```

## Dependencies
The project uses the following Python libraries:
- `numpy`
- `scipy`
- `pandas`
- `librosa`
- `matplotlib`
- `seaborn`
- `torch`
- `scikit-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset
The project processes an audio dataset of daily surrounding objects, such as:
- **Classes**: Rope, Chair, Clothes, Batteries, Paper

### Dataset Preparation
1. Store audio files in the `data/` folder, organized by class labels.
2. Run the metadata generation script to create a CSV file with labels and paths.

## Usage
### Preprocessing
Run the preprocessing pipeline to convert raw audio into Mel spectrograms:
```python
from src.preprocessing import AudioPreprocessingPipeline

pipeline = AudioPreprocessingPipeline(metadata_file='audio_metadata.csv', audio_dir='data/')
mel_spectrogram, label = pipeline[0]  # Example usage
```

### Echo State Network
Train the ESN model on the processed data:
```python
from src.esn import Reservoir

res = Reservoir(n_internal_units=30, spectral_radius=0.6, leak=0.6)
input_repr = res.getReservoirEmbedding(X, pca, ridge_embedding)
```

### Evaluation
Run the result analysis script to compute metrics and visualize performance:
```python
python analyze_results.py
```

## Hyperparameter Tuning
Hyperparameter tuning plays a critical role in optimizing the performance of the Echo State Network. Below are the key hyperparameters and how to tune them:

1. **Number of Internal Units (`n_internal_units`)**:
   - Represents the number of processing units in the reservoir.
   - More units generally increase the model's capacity but can lead to overfitting.
   - Suggested Range: 10–1000.
   - Tuning Strategy: Start with smaller values (e.g., 30) and incrementally increase, observing validation performance.

2. **Spectral Radius (`spectral_radius`)**:
   - Controls the largest eigenvalue of the reservoir's internal weight matrix.
   - Affects the network's stability and ability to capture temporal dependencies.
   - Suggested Range: 0.1–1.5.
   - Tuning Strategy: Begin with 0.6 and adjust in small increments.

3. **Leakage Rate (`leak`)**:
   - Determines the amount of leakage in the reservoir state update.
   - Higher values favor slower dynamics, which might be beneficial for long-term dependencies.
   - Suggested Range: 0.1–1.0.
   - Tuning Strategy: Test values like 0.2, 0.6, and 1.0.

4. **Input Scaling (`input_scaling`)**:
   - Scales the input weights.
   - Controls the influence of input data on the reservoir states.
   - Suggested Range: 0.01–1.0.
   - Tuning Strategy: Start with 0.1 and adjust as needed.

5. **Connectivity (`connectivity`)**:
   - Percentage of non-zero connections in the internal weight matrix.
   - Suggested Range: 0.1–0.5.
   - Tuning Strategy: Test values like 0.2, 0.3, and 0.5.

6. **Noise Level (`noise_level`)**:
   - Adds Gaussian noise to the state update.
   - Suggested Range: 0.0–0.05.
   - Tuning Strategy: Start with a small value (e.g., 0.01) and adjust based on results.

### Tuning Workflow
- Use grid search or random search to explore the hyperparameter space.
- Evaluate performance using a validation set.
- Track metrics such as accuracy, F1-score, and loss trends for each configuration.

### Example Code for Hyperparameter Tuning
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'n_internal_units': [30, 50, 100],
    'spectral_radius': [0.5, 0.6, 0.8],
    'leak': [0.2, 0.6, 1.0],
    'input_scaling': [0.1, 0.5],
    'connectivity': [0.2, 0.3],
    'noise_level': [0.01, 0.02]
}

grid = ParameterGrid(param_grid)

best_score = 0
best_params = None

for params in grid:
    res = Reservoir(
        n_internal_units=params['n_internal_units'],
        spectral_radius=params['spectral_radius'],
        leak=params['leak'],
        input_scaling=params['input_scaling'],
        connectivity=params['connectivity'],
        noise_level=params['noise_level']
    )
    input_repr = res.getReservoirEmbedding(X, pca, ridge_embedding)
    # Perform evaluation and track performance
    score = evaluate(input_repr, Y)
    if score > best_score:
        best_score = score
        best_params = params

print("Best Score:", best_score)
print("Best Parameters:", best_params)
```

## Results
- Achieved **XX% accuracy** on the test dataset.
- Detailed classification report and confusion matrix are available in the `results/` folder.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Acknowledgements
Special thanks to the open-source community for providing libraries and tools that make this project possible.
