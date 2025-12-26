# small-LLM - Implémentation d'un Modèle de Langage Transformer

Une implémentation simple d'un modèle de langage de type GPT (Generative Pre-trained Transformer) en PyTorch.

## Description

Ce projet implémente un modèle de langage transformer avec attention multi-têtes, capable d'apprendre à partir d'un corpus de texte et de générer du texte de manière autonome. Le modèle utilise une architecture similaire à GPT avec des blocs de transformer comprenant de l'attention multi-têtes et des couches feed-forward.

- **Architecture Transformer** : Implémentation complète avec attention multi-têtes
- **Entraînement** : Boucle d'entraînement avec validation et sauvegarde automatique
- **Génération de texte** : Génération de texte basée sur un contexte initial
- **Sauvegarde/Restauration** : Sauvegarde automatique du modèle et de l'optimiseur

## Prérequis

- Python 3.x
- PyTorch
- Un fichier `dataset.txt` contenant le corpus d'entraînement

## Installation

1. Clonez ce dépôt :
```bash
git clone <url-du-repo>
cd GPT
```

2. Installez les dépendances :
```bash
pip install torch
```

3. Assurez-vous d'avoir un fichier `dataset.txt` dans le répertoire du projet contenant votre corpus d'entraînement.

## Utilisation

### Configuration

Modifiez les hyperparamètres dans `GPT.py` selon vos besoins :

```python
batch_size = 64          # Taille des lots
block_size = 256         # Taille du contexte
max_iters = 5000         # Nombre d'itérations d'entraînement
learning_rate = 3e-4     # Taux d'apprentissage
n_embd = 384             # Dimension des embeddings
n_head = 6               # Nombre de têtes d'attention
n_layer = 6              # Nombre de couches transformer
dropout = 0.2            # Taux de dropout
```

### Mode Entraînement

Pour entraîner le modèle, modifiez la variable `mode` :

```python
mode = 'train'
```

Puis exécutez :

```bash
python GPT.py
```

Le modèle et l'optimiseur seront sauvegardés automatiquement dans `model.pt` et `optimizer.pt`.

### Mode Génération

Pour générer du texte, modifiez la variable `mode` :

```python
mode = 'gen'
```

Puis exécutez :

```bash
python GPT.py
```

Le modèle chargera les poids sauvegardés et générera 500 tokens de texte.

## Structure du Code

- `Head` : Tête d'attention simple
- `MultiHeadAttention` : Attention multi-têtes
- `FeedForward` : Couche feed-forward avec ReLU
- `Block` : Bloc transformer complet
- `BigramLanguageModel` : Modèle de langage principal

## Licence

Ce projet est sous licence GNU General Public License v3.0. Voir le fichier `LICENSE` pour plus de détails.

## Auteur

Créé dans le cadre d'un projet d'apprentissage des transformers et des modèles de langage.

