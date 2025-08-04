# Projet d'Analyse Prédictive des Ventes

## Description
Ce projet utilise des techniques avancées de Machine Learning pour prédire les ventes futures d'une chaîne de magasins de détail. En se basant sur des données historiques de ventes, des informations sur les produits et des facteurs externes, notre modèle vise à fournir des prévisions précises pour optimiser la gestion des stocks et la planification des ressources.

## Table des matières
1. [Installation](#installation)
2. [Structure du projet](#structure-du-projet)
3. [Utilisation](#utilisation)
4. [Données](#données)
5. [Modèle](#modèle)
6. [Évaluation](#évaluation)
7. [Déploiement](#déploiement)
8. [Contact](#contact)

## Installation
Pour installer et exécuter ce projet, suivez ces étapes :

```bash
git clone https://github.com/votre-nom/projet-analyse-predictive-ventes.git
cd gpt2_fine-tuned
pip install -r requirements.txt
```

Pour installer le projet comme un package Python :
```bash
pip install .
```

Pour utiliser le projet comme package utiliser la commande :
```bash
generate-text "Le prompt que vous voulez?"
```

## Structure du projet
```
gpt2_fine-tuned/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 1.0-exploration-donnees.ipynb
│   └── 3.0-modelisation.ipynb
│
├── src/
│   │── __init__.py
│   │── make_dataset.py
│   │── preprocessing.py
│   │── train.py
│   │── send_prompt.py
│   │── generator.py
│   │
│   │
│   │
│   ├── models/
│   │   ├── model_saved
│   │   └── model_trained
│   │
│   └── visualization/
│       └── visualize.py
│
├── tests/
│
├── requirements.txt
├── setup.py
└── README.md
```

## Utilisation
Pour faire le preprocessing des données et creer le dataset:
```bash
python src/preprocessing.py
python src/make_dataset.py
```

Pour entraîner le modèle :
```bash
python src/train.py
```

Pour exécuter le pipeline complet de generation de texte :

```bash
python src/send_prompt.py
```

Pour visualiser les résultats :
```bash
python src/visualization/visualize.py
```

## Données
Les données utilisées dans ce projet proviennent de [kaggle]. Elles incluent :
- Des questions de devinettes
- Et des reponses a ces devinettes

Les scripts de prétraitement des données se trouvent dans `src/`.

## Modèle
Nous utilisons un modèle de [GPT-2(fine-tuned)] pour ses performances élevées sur des tâches de génération de texte,.

## Évaluation
Les performances du modèle sont évaluées en utilisant :
- Trainer (une classe de la bibliothèque transformers ) qui calcule des métriques de 
performance pendant l'évaluation sur les 
ensembles de données d'entraînement et de test.

Les résultats sont disponibles dans le fichier `visualization/visualize.py` et `visualization/.logs`.

## Déploiement
Le modèle est déployé sur [plateforme, ex: AWS SageMaker] pour des generations de text.

## Contact
Pour toute question ou suggestion, veuillez contacter [Caleb Laurent](laurentcaleb99@gmail.com).
# gpt2_finetuned
