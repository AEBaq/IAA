**Entraînement**
```bash
# Lancer l'entraînement en arrière-plan
nohup python3 main.py > training.log 2>&1 &

# Lancer avec rendu visuel
python3 main.py --render

# Reprendre depuis un checkpoint
nohup python3 main.py --resume > training.log 2>&1 &
```

**Suivre la progression**
```bash
# Voir les derniers épisodes
grep "Episode" training.log | tail -20

# Voir le meilleur score
cat checkpoints/training_state_best.json

# Voir à quel épisode on en est
grep "Episode" training.log | tail -1

# Voir si le processus tourne
ps aux | grep main.py
```

**Arrêter l'entraînement**
```bash
kill $(pgrep -f main.py)
```

**Evaluation**
```bash
# Évaluer le meilleur checkpoint
python3 main.py --eval-only

# Évaluer un checkpoint spécifique
python3 main.py --eval-only --checkpoint checkpoints/agent_ep500.pth
```

**Générer la courbe d'entraînement**
```bash
grep "Episode" training.log | awk -F'Score: ' '{print $2}' | awk -F' ' '{print $1}' > scores.txt

MPLBACKEND=Agg python3 -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

scores = []
with open('scores.txt') as f:
    for line in f:
        try:
            scores.append(float(line.strip()))
        except:
            pass

avg_scores = [np.mean(scores[max(0,i-100):i+1]) for i in range(len(scores))]

plt.figure(figsize=(12,4))
plt.plot(scores, label='Score', alpha=0.7)
plt.plot(avg_scores, label='Average Score (100 ep)')
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.savefig('training_metrics.png')
print('Saved', len(scores), 'episodes')
"
```
## PC-REDS
**Trouver ton job ID** \
`squeue -u $(whoami)`

**Voir les derniers épisodes (remplace <JOB_ID> par le vrai ID)** \
`grep "Episode" logs/training_<JOB_ID>.out | tail -20`

**Voir à quel épisode on en est** \
`grep "Episode" logs/training_<JOB_ID>.out | tail -1`

**Meilleur score** \
`cat checkpoints/training_state_best.json`

**Générer la courbe (en local ou dans un job dédié)** \
`grep "Episode" logs/training_<JOB_ID>.out | awk -F'Score: ' '{print $2}' | awk -F' ' '{print $1}' > scores.txt`

### Workflow typique
* 1. Soumettre un entraînement : `sbatch run_training.sbatch`

* 2. Surveiller : `squeue -u $(whoami)`

* 3. Si le job expire après 3h, reprendre : `sbatch run_resume.sbatch`

* 4. Évaluer à la fin : `sbatch run_eval.sbatch`

* 5. Consulter l'historique : `sacct -format=JobID,JobName,State,Elapsed`
 
### pyproject.toml
```toml
[tool.poetry]
name = "lab3"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
dependencies = [
    "gym>=0.9.0",
    "numpy==1.19.0",
    "pyglet>=1.4.0,<=1.5.0",
    "pygobject>=3.44.2",
    "typing-extensions>=4.7.1",
]
[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```
