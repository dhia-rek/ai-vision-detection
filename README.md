# 🎯 AI Vision OBJECT Detection

Système de détection d'objets en temps réel avec YOLO et interface web moderne.

## ✨ Fonctionnalités

- 🖼️ Upload d'images avec drag & drop
- 🔍 Détection YOLO en temps réel
- 📊 Affichage côte à côte (original + résultat)
- 🎨 Interface web responsive

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- Git

### Installation

```bash
# Cloner le repo
git clone https://github.com/AkramKhattabi/ai-vision-detection.git
cd ai-vision-detection

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
cd backend
python -m uvicorn app.main:app --reload
```

### Utiliser

Ouvrez votre navigateur et allez à : **http://localhost:8000**

## 📁 Structure

```
ai-vision-detection/
├── backend/
│   ├── app/main.py          # Backend FastAPI
│   ├── static/index.html    # Interface web
│   ├── yolov8n.pt           # Modèle YOLO
│   ├── temp/                # Images temporaires
│   └── outputs/             # Résultats
└── requirements.txt
```

## 🛠️ Technologies

- **Backend** : FastAPI, Python, YOLOv8, OpenCV
- **Frontend** : HTML, CSS, JavaScript
- **Server** : Uvicorn

## 👨‍💻 Auteur

Akram KHATTABI - [@AkramKhattabi](https://github.com/AkramKhattabi)
