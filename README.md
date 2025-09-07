# 📩 Smart Email Spam Classifier Web App

A machine learning powered **Spam Email Classifier** built with **Streamlit** 🚀.  
This web app helps in classifying emails/messages as **Spam** or **Not Spam** using NLP techniques and a trained ML model.

---

## ✨ Features
- 🔎 **Real-time Email Classification** (Spam / Not Spam)
- 🧹 **Text Preprocessing** (stopword removal, stemming, tokenization)
- 📊 **Interactive Visualizations** (Matplotlib & Plotly support)
- 🌐 **Deployed with Streamlit Cloud**
- ⚡ Lightweight and fast inference using pre-trained model

---

## 🛠️ Tech Stack
- **Frontend / UI** → [Streamlit](https://streamlit.io/)
- **Machine Learning** → Scikit-learn
- **NLP** → NLTK (tokenizer, stopwords, stemming)
- **Visualization** → Matplotlib, Plotly
- **Others** → NumPy, Pandas, Requests

---

## 📂 Project Structure
```

├── model.pkl              # Trained ML model
├── tfidf.pkl              # TF-IDF Vectorizer
├── spam\_data.csv          # Dataset used
├── newapp.py              # Main Streamlit app code
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

````

````
## ⚡ Installation & Setup (Local)

1. **Clone the repo**
   ```bash
   git clone https://github.com/Rou45/Smart-Email-Spam-Classifier-Web-App.git
   cd Smart-Email-Spam-Classifier-Web-App
````

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Mac/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run newapp.py
   ```

---

## 🌐 Deployment

This project is deployed on **Streamlit Cloud**.
You can try the live app here 👉 [Live Demo](https://rou-smart-email-spam.streamlit.app/)

---

## 📝 Notes

* If you face **NLTK resource errors**, run:

  ```python
  import nltk
  nltk.download("punkt")
  nltk.download("punkt_tab")
  nltk.download("stopwords")
  ```

* For stability, you can pin **NLTK version 3.8.1** in `requirements.txt`.

---

## 📸 Screenshots

### 🏠 Home Page

<img width="1868" height="816" alt="image" src="https://github.com/user-attachments/assets/75cf73e0-af6f-459a-bc39-cc9f4a6983af" />

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first
to discuss what you would like to change.

---

## 📜 License

This project is licensed under the **MIT License**.

---

### 👨‍💻 Author

Developed with ❤️ by **[Rou45](https://github.com/Rou45)**

```
