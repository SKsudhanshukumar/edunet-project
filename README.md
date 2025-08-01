# Tree Species Classification using CNN

A deep learning project for classifying tree species using Convolutional Neural Networks (CNN). This project implements multiple CNN architectures to identify 30 different tree species from images.

## 📋 Project Overview

This project focuses on automated tree species identification using computer vision and deep learning techniques. The system can classify images into 30 different tree species categories, making it useful for forestry research, environmental monitoring, and educational purposes.

## 🌳 Dataset

The project uses a comprehensive tree species dataset containing images of 30 different tree species:

- **Amla** - Indian gooseberry
- **Asopalav** - Polyalthia longifolia
- **Babul** - Acacia nilotica
- **Bamboo** - Bambusa species
- **Banyan** - Ficus benghalensis
- **Bili** - Aegle marmelos
- **Cactus** - Various cactus species
- **Champa** - Plumeria species
- **Coconut** - Cocos nucifera
- **Garmalo** - Careya arborea
- **Gulmohor** - Delonix regia
- **Gunda** - Cordia dichotoma
- **Jamun** - Syzygium cumini
- **Kanchan** - Bauhinia variegata
- **Kesudo** - Butea monosperma
- **Khajur** - Phoenix dactylifera
- **Mango** - Mangifera indica
- **Motichanoti** - Sesbania grandiflora
- **Neem** - Azadirachta indica
- **Nilgiri** - Eucalyptus species
- **Pilikaren** - Helicteres isora
- **Pipal** - Ficus religiosa
- **Saptaparni** - Alstonia scholaris
- **Shirish** - Albizia lebbeck
- **Simlo** - Bombax ceiba
- **Sitafal** - Annona squamosa
- **Sonmahor** - Cassia fistula
- **Sugarcane** - Saccharum officinarum
- **Vad** - Ficus benghalensis
- **Other** - Miscellaneous species

**Dataset Statistics:**
- Total Images: 1,600
- Number of Classes: 30
- Image Distribution: Varies by species (50-150 images per class)

## 🏗️ Model Architecture

The project implements multiple CNN architectures:

### 1. Basic CNN Model
- Custom CNN architecture for baseline performance
- Saved as: `basic_cnn_tree_species.h5`

### 2. Improved CNN Model
- Enhanced CNN with better performance
- Saved as: `improved_cnn_model.h5`

### 3. Transfer Learning Model
- Uses EfficientNetB0 as base model
- Pre-trained weights: `efficientnetb0_notop.h5`
- Final model: `tree_species_model.h5`

## 📁 Project Structure

```
edunet/
├── README.md
├── week 2/
│   ├── tree_CNN.ipynb              # Main Jupyter notebook
│   ├── basic_cnn_tree_species.h5   # Basic CNN model
│   ├── improved_cnn_model.h5       # Improved CNN model
│   ├── tree_species_model.h5       # Final trained model
│   ├── efficientnetb0_notop.h5     # Pre-trained EfficientNet weights
│   └── Tree_Species_Dataset/       # Dataset directory
│       ├── amla/
│       ├── asopalav/
│       ├── babul/
│       ├── bamboo/
│       ├── banyan/
│       ├── bili/
│       ├── cactus/
│       ├── champa/
│       ├── coconut/
│       ├── garmalo/
│       ├── gulmohor/
│       ├── gunda/
│       ├── jamun/
│       ├── kanchan/
│       ├── kesudo/
│       ├── khajur/
│       ├── mango/
│       ├── motichanoti/
│       ├── neem/
│       ├── nilgiri/
│       ├── other/
│       ├── pilikaren/
│       ├── pipal/
│       ├── saptaparni/
│       ├── shirish/
│       ├── simlo/
│       ├── sitafal/
│       ├── sonmahor/
│       ├── sugarcane/
│       └── vad/
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install opencv-python
pip install scikit-learn
pip install jupyter
```

### Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd edunet
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook "week 2/tree_CNN.ipynb"
   ```

3. **Run the notebook cells to:**
   - Load and explore the dataset
   - Preprocess the images
   - Train the CNN models
   - Evaluate model performance
   - Make predictions on new images

### Making Predictions

To use the trained model for predictions:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('week 2/tree_species_model.h5')

# Load and preprocess your image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust size based on model requirements
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
image_path = 'path/to/your/tree/image.jpg'
processed_image = preprocess_image(image_path)
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction)

# Tree species classes (adjust based on your model's class order)
tree_species = ['amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 
                'cactus', 'champa', 'coconut', 'garmalo', 'gulmohor', 'gunda',
                'jamun', 'kanchan', 'kesudo', 'khajur', 'mango', 'motichanoti',
                'neem', 'nilgiri', 'other', 'pilikaren', 'pipal', 'saptaparni',
                'shirish', 'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad']

print(f"Predicted tree species: {tree_species[predicted_class]}")
```

## 📊 Model Performance

The models are evaluated using various metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Detailed performance metrics and visualizations are available in the Jupyter notebook.

## 🔬 Technical Details

### Data Preprocessing
- Image resizing and normalization
- Data augmentation techniques
- Train-validation-test split

### Model Training
- Multiple CNN architectures tested
- Transfer learning with EfficientNetB0
- Hyperparameter optimization
- Early stopping and model checkpointing

### Evaluation
- Cross-validation
- Performance visualization
- Error analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset contributors and researchers in the field of botanical classification
- TensorFlow and Keras communities
- EfficientNet architecture developers

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note:** This project is part of an educational initiative (EduNet) focused on applying machine learning techniques to real-world problems in environmental science and forestry.