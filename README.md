# **Projet de Transcription Vocale**

## **Description**
Ce projet effectue la transcription d'un fichier audio en texte en utilisant la bibliothèque Python `speech_recognition` et Google Speech Recognition. Pour contourner les limitations de durée, l'audio est découpé en segments, transcrit morceau par morceau, puis reconstitué en texte complet.

## **Fonctionnalités**
1. Découper automatiquement un fichier audio en segments basés sur les silences.
2. Transcrire chaque segment avec Google Speech Recognition.
3. Combiner les transcriptions pour produire un texte complet.
4. Gérer les fichiers temporaires pour éviter d'encombrer le système.

---

## **Technologies Utilisées**
- **Python** :
  - `pydub` : Pour découper et manipuler l'audio.
  - `speech_recognition` : Pour convertir la parole en texte.
  - `tempfile` : Pour la gestion propre des fichiers temporaires.

---

## **Installation**

### **Prérequis**
- Python 3.10 ou supérieur.
- FFmpeg (nécessaire pour `pydub`).

### **Étapes d'installation**
1. **Cloner le projet** :
   ```bash
   git clone <url-du-repo>
   cd <nom-du-dossier>
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances Python** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Installer FFmpeg** :
   - macOS :
     ```bash
     brew install ffmpeg
     ```
   - Ubuntu :
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - Windows : Téléchargez FFmpeg [ici](https://ffmpeg.org/download.html) et ajoutez-le à votre PATH.

---

## **Utilisation**

1. Placez votre fichier audio dans le dossier du projet (par exemple : `Lesson-001-Anglais.wav`).
2. Lancez le script Python :
   ```bash
   python main.py
   ```
3. La transcription sera affichée dans la console.

---

## **Structure du Projet**

```plaintext
├── main.py                 # Script principal pour la transcription
├── requirements.txt        # Dépendances Python
├── README.md               # Documentation
└── Lesson-001-Anglais.wav  # Exemple de fichier audio (à ajouter manuellement)
```

---

## **Exemple de Résultat**

### **Fichier Audio :**
_"Bonjour, je m'appelle Pierre et je suis ici pour apprendre le français."_

### **Transcription Générée :**
```plaintext
Bonjour, je m'appelle Pierre et je suis ici pour apprendre le français.
```

---

## **Améliorations Futures**
- Support des fichiers audio multilingues avec détection automatique de la langue.
- Intégration avec un autre moteur de transcription comme Whisper ou AssemblyAI pour une plus grande précision.
