Comparative Analysis of Traditional and Deep Learning Approaches for Document Classification using the 20 Newsgroups Dataset
Overview
This repository presents a comparative analysis of traditional machine learning and deep learning models for text classification using the 20 Newsgroups dataset. The study evaluates the effectiveness of Naive Bayes (NB), Support Vector Machine (SVM), Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and their hybrid versions, including BiLSTM with Attention and CNN-GRU with Attention Mechanism and Gradient Clipping.
The models are assessed based on performance metrics such as accuracy, F1 score, PR-AUC, ROC-AUC, and precision-recall curves. The 20 Newsgroups dataset serves as the benchmark, containing documents from 20 categories like science, politics, technology, and religion.
Features
•	Traditional Models: Naive Bayes (NB) and Support Vector Machine (SVM).
•	Deep Learning Models: Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN).
•	Hybrid Models:
o	BiLSTM with Attention Mechanism.
o	CNN with Attention Mechanism, GRU, and Gradient Clipping.
•	Performance Metrics: Accuracy, Precision, Recall, F1 Score, PR-AUC, ROC-AUC, Precision-Recall Curve, ROC Curve.
•	Dataset: 20 Newsgroups dataset, containing 20 categories of news documents.
Project Structure
bash
CopyEdit
├── data/                  # Contains 20 Newsgroups dataset
├── models/                # Model implementation files
│   ├── traditional/       # Traditional ML models (Naive Bayes, SVM)
│   └── deep_learning/     # Deep learning models (LSTM, CNN, BiLSTM with Attention, etc.)
├── preprocessing/         # Preprocessing scripts
├── results/               # Model evaluation results and plots
├── README.md              # This file
└── requirements.txt       # Python dependencies
Setup Instructions
Requirements
The following Python libraries are required to run the models:
•	numpy
•	pandas
•	scikit-learn
•	tensorflow
•	keras
•	matplotlib
•	seaborn
•	nltk
•	gensim
Installation
1.	Clone the repository:
bash
CopyEdit
git clone https://github.com/yourusername/Document-Classification.git
cd Document-Classification
2.	Create a virtual environment (optional but recommended):
bash
CopyEdit
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3.	Install the required dependencies:
bash
CopyEdit
pip install -r requirements.txt
Dataset
The dataset used in this project is the 20 Newsgroups dataset. The dataset can be downloaded and preprocessed using the code in the data/ folder.
Preprocessing
Before training models, ensure that the dataset is preprocessed. This includes tokenization, stopword removal, lemmatization, and converting the text data into numerical representations (TF-IDF, GloVe embeddings, etc.). Preprocessing scripts are located in the preprocessing/ folder.
Training the Models
Traditional Models
To train the Naive Bayes or SVM models, run the respective scripts in the models/traditional/ folder:
bash
CopyEdit
python models/traditional/naive_bayes.py
python models/traditional/svm.py
Deep Learning Models
To train the LSTM or CNN models, run the respective scripts in the models/deep_learning/ folder:
bash
CopyEdit
python models/deep_learning/lstm.py
python models/deep_learning/cnn.py
Hybrid Models
To train the hybrid models like BiLSTM with Attention or CNN-GRU with Attention Mechanism and Gradient Clipping:
bash
CopyEdit
python models/deep_learning/bilstm_attention.py
python models/deep_learning/cnn_gru_attention.py
Model Evaluation
The models' performance is evaluated using various metrics, such as accuracy, F1 score, and ROC-AUC. Evaluation results will be saved in the results/ folder, where you can also find precision-recall curves and ROC curves for each model.
Results
The comparative results of each model are displayed in the following metrics:
•	Accuracy: The proportion of correct predictions.
•	Precision, Recall, and F1 Score: These metrics are used for a more balanced evaluation, especially for imbalanced datasets.
•	PR-AUC: Precision-Recall AUC for evaluating the performance of classifiers on imbalanced datasets.
•	ROC-AUC: Receiver Operating Characteristic AUC, showing the model's ability to distinguish between positive and negative instances.
Conclusion
This study highlights the strengths and weaknesses of traditional machine learning models like Naive Bayes and SVM and deep learning models such as LSTM and CNN. Hybrid deep learning models, especially BiLSTM with Attention, show superior performance across all metrics.
License
This project is licensed under the MIT License - see the LICENSE file for details.

# Document-Classification
