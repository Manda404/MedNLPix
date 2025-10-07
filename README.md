# 🧠 MedNLPix — Medical NLP Intelligence Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-Poetry-60A5FA.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🏥 **Classification Multi-Classes de Rapports Médicaux par NLP et Machine Learning**

Architecture professionnelle Data Science pour la classification automatique de textes médicaux, combinant NLP, Machine Learning et Deep Learning dans une pipeline modulaire et industrialisable.

**Auteur :** Rostand Surel  
**GitHub :** [@Manda404](https://github.com/Manda404)

---

## 📋 Table des Matières

- [🎯 Objectif du Projet](#-objectif-du-projet)
- [📂 Jeu de Données](#-jeu-de-données)
- [🧩 Architecture du Projet](#-architecture-du-projet)
- [⚙️ Installation et Lancement](#️-installation-et-lancement)
- [🧠 Approche Méthodologique](#-approche-méthodologique)
- [📈 Résultats et Analyses](#-résultats-et-analyses)
- [🧪 Tests et Qualité du Code](#-tests-et-qualité-du-code)
- [🌐 Technologies Principales](#-technologies-principales)
- [📬 Contact et Contributions](#-contact-et-contributions)
- [🧾 Licence](#-licence)

---

## 🎯 Objectif du Projet

Développer une **solution complète de classification automatique** de rapports médicaux à partir de texte libre, en combinant :
- 📝 **NLP** (Natural Language Processing)
- 🤖 **Machine Learning** classique et avancé
- 🧬 **Deep Learning** (LSTM, BERT)
- 🏗️ Architecture **modulaire** et **industrialisable**

**But :** Prédire la spécialité médicale d'un rapport texte à l'aide de modèles ML/NLP, dans une pipeline entièrement automatisée et extensible.

---

## 📂 Jeu de Données

Basé sur un sous-ensemble de **HuggingFace Medical Abstracts** (~2 286 échantillons).

Les textes médicaux sont étiquetés selon **5 grandes classes** :

| Classe | Description |
|--------|-------------|
| 🧬 | **Neoplasms** (Tumeurs) |
| 💊 | **Digestive System Diseases** (Maladies digestives) |
| 🧠 | **Nervous System Diseases** (Système nerveux) |
| ❤️ | **Cardiovascular Diseases** (Cardiovasculaire) |
| 🧾 | **General Pathological Conditions** (Pathologies générales) |

---

## 🧩 Architecture du Projet

Le projet adopte une structure inspirée de la **Clean Architecture** pour garantir :
- ✅ Séparation claire entre données, modèles et application
- ♻️ Réutilisabilité du code pour d'autres cas NLP
- 🔄 Intégration fluide avec CI/CD, tests, et applications (Streamlit/FastAPI)

```
MedNLPix/
│
├── pyproject.toml              # Gestion des dépendances via Poetry
├── configs/                    # Configurations YAML : data, train, eval, app
├── data/                       # Données raw → processed
├── models/                     # Artifacts + registry.json
├── reports/                    # Métriques et visualisations
├── notebooks/                  # EDA et exploration
├── apps/                       # Streamlit (front) & FastAPI (backend)
├── src/mednlpix/               # Core du projet
│   ├── core/                   # Schemas & config Pydantic
│   ├── data/                   # Chargement et split
│   ├── features/               # Nettoyage et vectorisation
│   ├── models/                 # ML & DL
│   ├── pipelines/              # Composition sklearn
│   ├── training/               # Entraînement et évaluation
│   ├── inference/              # Prédiction et postprocessing
│   └── cli.py                  # CLI (Typer)
└── tests/                      # Tests unitaires
```

---

## ⚙️ Installation et Lancement

### 1️⃣ Clonage & Installation

```bash
# Cloner le projet
git clone https://github.com/Manda404/MedNLPix.git
cd MedNLPix

# Installer via Poetry
poetry install
poetry shell
```

### 2️⃣ Préparation des Données

```bash
poetry run mednlpix data
```

### 3️⃣ Entraînement d'un Modèle

```bash
poetry run mednlpix train --config configs/train.yaml
```

### 4️⃣ Évaluation des Performances

```bash
poetry run mednlpix evaluate
```

### 5️⃣ Prédiction sur un Texte

```bash
poetry run mednlpix predict --text "Patient présente des signes d'insuffisance respiratoire aiguë"
```

### 6️⃣ Application Streamlit

```bash
poetry run streamlit run apps/streamlit_app/app.py
```

💡 **L'application permet de saisir un texte médical et d'obtenir la classe prédite avec la probabilité associée.**

---

## 🧠 Approche Méthodologique

### 🩺 1. Prétraitement du texte
- Nettoyage et suppression de ponctuation
- Lemmatisation et gestion des stopwords
- Vectorisation avec **TF-IDF** ou embeddings pré-entraînés (**spaCy** / **BERT**)

### ⚙️ 2. Entraînement multi-modèles
- **Classiques :** Logistic Regression, SVM, Random Forest
- **Avancés :** LSTM, BERT (optionnels)

### 📊 3. Évaluation & sélection
- Utilisation de **F1-score**, **Précision**, **Rappel**, **Matrice de confusion**
- Sauvegarde du meilleur modèle dans `models/artifacts/`
- Mise à jour du `registry.json`

### 🧩 4. Industrialisation
- **CLI unifiée** (Typer) pour tout le pipeline : `data`, `train`, `evaluate`, `predict`
- Architecture **modulaire** pour l'extension future (autres datasets, tasks NLP)
- Suivi des métriques dans `reports/`

---

## 📈 Résultats et Analyses

- ✅ Les modèles classiques offrent une **très bonne performance de base** (F1 ~0.78)
- 🚀 L'intégration de **BERT** améliore la robustesse sur les classes minoritaires
- 📊 **Visualisations disponibles** dans `reports/figures/` :
  - Matrice de confusion
  - Top features TF-IDF
  - Courbes ROC

---

## 🧪 Tests et Qualité du Code

| Outil | Usage |
|-------|-------|
| **ruff** | Linting rapide |
| **black** | Formatage automatique |
| **mypy** | Typage statique |
| **pytest** | Tests unitaires |
| **pytest-cov** | Couverture de code |
| **GitHub Actions** | CI/CD automatisé |

✅ **Conformité PEP8** et **typage statique** garantis

---

## 🌐 Technologies Principales

| Catégorie | Outils / Librairies |
|-----------|---------------------|
| **Langage** | Python 3.11 |
| **Gestion environnement** | Poetry |
| **NLP / ML** | scikit-learn, TF-IDF, BERT, spaCy |
| **Deep Learning** | TensorFlow / PyTorch (LSTM, BERT) |
| **App / API** | Streamlit, FastAPI |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Qualité / CI** | pytest, ruff, black, mypy, GitHub Actions |

---

## 📬 Contact et Contributions

**👤 Rostand Surel**  
📧 Email : [rostand.surel@yahoo.com](mailto:rostand.surel@yahoo.com)  
💼 LinkedIn : [linkedin.com/in/rostand-surel](https://linkedin.com/in/rostand-surel)  
🐙 GitHub : [@Manda404](https://github.com/Manda404)

### 💡 Contributions

Les **Pull Requests** et **Issues** sont les bienvenues !  
N'hésitez pas à contribuer pour améliorer ce projet.

---

## 🧾 Licence

Ce projet est sous licence **MIT** — utilisation libre à des fins d'apprentissage ou de démonstration.

---

## 🚀 Résumé

**MedNLPix** est une **architecture de référence** pour les projets NLP médicaux modernes :

✅ **Reproductible**  
✅ **Maintenable**  
✅ **Orientée MLOps**  
✅ **Prête pour la production**

---

<div align="center">
  <strong>⭐ Si ce projet vous plaît, n'hésitez pas à lui donner une étoile ! ⭐</strong>
</div>