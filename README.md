# Radical Eye: AI-Powered Detection of Extremist Language

## Project Overview
Radical Eye is an AI-powered system designed to detect extremist and offensive language using state-of-the-art natural language processing (NLP) and deep learning techniques. The project aims to contribute to online content moderation by identifying harmful language in textual data from various online platforms.

## Features
- **Pre-trained Transformer Model:** Fine-tuned XLM-RoBERTa for offensive language classification.
- **Custom Deep Learning Model:** LSTM-based architecture for contextual language understanding.
- **Data Preprocessing:** Includes text normalization, tokenization, lemmatization, stopword removal, and more.
- **Model Evaluation Metrics:** Uses accuracy, precision, recall, F1-score, and ROC AUC.
- **Deployment-Ready:** Designed for batch processing and real-time inference applications.

## Methodology
1. **Data Collection:**
   - Reddit comments dataset used for training and evaluation.
2. **Data Preprocessing:**
   - Text normalization, punctuation removal, tokenization, stopword removal, and lemmatization.
3. **Model Development:**
   - Fine-tuning XLM-RoBERTa for text classification.
   - Custom LSTM-based model with embedding layers.
4. **Training and Evaluation:**
   - Optimized using techniques like dropout regularization, learning rate scheduling, and early stopping.
   - Evaluated using precision, recall, F1-score, and confusion matrix.
5. **Deployment:**
   - Ready for integration with real-world applications in social media moderation and community management.

## Results
- Successfully trained models achieving high accuracy in detecting offensive language.
- Robust performance demonstrated through various NLP evaluation metrics.
- Deployment-ready framework adaptable to different online platforms.

## Future Improvements
- Expanding dataset diversity for improved generalization.
- Exploring additional deep learning architectures for enhanced contextual understanding.
- Implementing real-time monitoring for proactive content moderation.

## References
For detailed references, please check the `References` section in the final project report.

---
### How to Use
1. **Install Dependencies:** Ensure all necessary Python libraries (TensorFlow, PyTorch, Transformers, etc.) are installed.
2. **Run the Model:** Execute the training script to fine-tune or use the pre-trained model for inference.
3. **Evaluate Performance:** Utilize test datasets to measure the effectiveness of the detection model.

For further details, refer to the final project report.

**Project by:**  Chandu Gogineni

