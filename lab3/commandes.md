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

**Arrêterl'entraînement**
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
