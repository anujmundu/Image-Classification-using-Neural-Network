🧠 Image Classification using Neural Network
A deep learning project that classifies images of dogs and cats using a custom neural network built with PyTorch. This repository includes data preprocessing, model training, evaluation, and visualization.

📁 Project Structure
Code
Image-Classification-using-Neural-Network/
├── train/
│   ├── dogs/
│   └── cats/
├── model.py
├── train.py
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
🚀 Features
Custom CNN architecture using PyTorch

Image preprocessing with transforms

Training and validation loops with live metrics

Git LFS support for large image datasets

Modular codebase for easy experimentation

🧪 Setup Instructions
1. Clone the repository
bash
git clone https://github.com/anujmundu/Image-Classification-using-Neural-Network.git
cd Image-Classification-using-Neural-Network
2. Create a virtual environment
bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Run training
bash
python train.py
📊 Sample Results
Metric	Value
Accuracy	92.3%
Loss	0.18
Epochs	25
📦 Dependencies
torch

torchvision

numpy

matplotlib

Pillow

See requirements.txt for full list.

📌 Notes
Large image files are tracked using Git LFS.

Dataset should be structured as:

Code
train/
  ├── dogs/
  └── cats/

👨‍💻 Author
Anuj Mundu
🎓 MCA, Maulana Azad National Institute of Technology (MANIT), Bhopal
🌐 GitHub
📧 Email: anujmark.edwin.ame@gmail.com
