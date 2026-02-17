NLP Image Captioning Bot: Bridging Pixels and Prose

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haresh8765/image-caption-bot)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An end-to-end **Neural Image Captioning System** that translates high-dimensional visual features into coherent, grammatically correct English sequences. This project implements a **CNN-LSTM architecture** with a **Soft Attention Mechanism** to intelligently describe image content.

---

## 🚀 Live Demo
Test the model's performance in real-time on Hugging Face Spaces:  
👉 **[Try the Image Captioning Bot](https://huggingface.co/spaces/haresh8765/image-caption-bot)**

---

## 🧠 Technical Architecture

The model utilizes a sophisticated Encoder-Decoder framework optimized for multimodal sequence-to-sequence tasks.

### 1. The Visual Encoder (ResNet-50)
* **Backbone:** 50-layer Residual Network (ResNet) pre-trained on ImageNet.
* **Feature Extraction:** Bypassed the final pooling and fully connected layers to extract a **7x7x2048** spatial feature map.
* **Impact:** This allows the model to maintain 49 distinct spatial regions of the image, rather than collapsing the entire scene into a single vector.

### 2. The Bridge (Soft Attention)
* **Hidden Layers:** Linear layer with **256 neurons**.
* **Logic:** Implemented a **Soft Attention** layer that computes attention weights (alpha) for each of the 49 image regions at every time step of the decoding process.

### 3. The Language Decoder (LSTM)
* **Word Embeddings:** 256-dimensional dense vector space.
* **RNN Core:** LSTM with **512 hidden units** to manage long-term dependencies and prevent vanishing gradients.
* **Vocabulary:** Trained on the **Flickr30k** dataset with custom tokenization and normalization.



---

## 🔍 Decoding Strategies

To ensure high-quality caption generation, the system supports two inference methods:
1. **Greedy Search:** Rapidly selects the word with the highest probability at each step.
2. **Beam Search (k=3):** Maintains the top 3 most probable sequences simultaneously, significantly improving grammatical coherence and descriptive accuracy.



---

## 📊 Performance & Training
| Hyperparameter | Value |
| :--- | :--- |
| **Encoder Channels** | 2048 |
| **Attention Dim** | 256 |
| **Embedding Dim** | 256 |
| **LSTM Hidden Dim** | 512 |
| **Dropout** | 0.5 |
| **Beam Size** | 3 |

---

## 🛠️ Installation & Setup

### 1. Clone & Install Dependencies
git clone [https://github.com/haresh8765/image-caption-bot.git](https://github.com/haresh8765/image-caption-bot.git)
cd image-caption-bot
pip install -r requirements.txt
2. Required Files
Ensure the following files are in your root directory:

best_model_bleu.pth (The trained weights)

vocab.pkl (The pickled Vocabulary class)

3. Run Locally
Bash
python app.py
📁 File Structure
app.py: Gradio interface and inference logic for Greedy/Beam search.

best_model_bleu.pth: Model checkpoints.

vocab.pkl: Pre-processed vocabulary mapping.

requirements.txt: List of required Python packages.
