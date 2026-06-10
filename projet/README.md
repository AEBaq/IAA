# Lab 4 - Smart Parking (Reverse)
## Description
Système de parking autonome en marche arrière pour Duckiebot. Le robot suit la voie de manière autonome, détecte son spot de parking assigné via un AprilTag, exécute une manoeuvre de stationnement en marche arrière, attend 5 secondes, puis reprend le lane following.
## Architecture
```
LANE_FOLLOWING → PARKING_DETECTED → ALIGNING → REVERSE_PARKING → WAITING → EXIT → LANE_FOLLOWING
```
### Noeuds ROS
- `lane_following_node` : suivi de voie via modèle ResNet18 fine-tuné (lab2)
- `parking_node` : noeud principal, orchestre la state machine et les manoeuvres
- `state_machines` : gestion des transitions d'états

## Structure du projet
```
projet/
├── Dockerfile
├── dependencies-apt.txt
├── dependencies-py3.txt
├── dependencies-py3.dt.txt
├── launchers/
│   └── parking.sh
├── models/
│   └── best_finetuned_model.pt     ← à copier manuellement (non inclus dans le repo)
├── packages/
│   └── parking_pkg/
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── launch/
│       │   └── parking.launch
│       └── src/
│           ├── lane_following_node.py
│           ├── parking_node.py
│           └── state_machine.py
└── rapport/
    └── rapport.md
```

## Installation et déploiement 
1. Copier le modèle sur le robot
```bash
scp models/best_finetuned_model.pt duckie@d1.local:~/
```

2. Configurer l'ID du spot de parking. Modifier `targer_tag_id` dans `launchers/parking.sh` selon l'AprilTag assigné.
```bash
dt-exec roslaunch parking_pkg parking.launch \
robot_name:=${VEHICLE_NAME} \
target_tag_id:=TON_ID_ICI
```

3. Builder pour le robot
```bash
dts devel build -f
```

4. Lancer le système en pointant vers le robot
```bash
dts devel run -L parking -R d1
```

## Paramètres ajustables

Tous les paramètres sont modifiables dans `packages/parking_pkg/launch/parking.launch` sans recompiler :

| Paramètre | Défaut | Description |
|---|---|---|
| `target_tag_id` | `0` | ID de l'AprilTag du spot assigné |
| `base_speed` | `0.3` | Vitesse de base (lane following + sortie) |
| `reverse_speed` | `-0.25` | Vitesse de recul |
| `align_duration` | `1.0` | Durée de la rotation d'alignement (s) |
| `reverse_duration` | `2.5` | Durée de la marche arrière (s) |
| `exit_duration` | `2.5` | Durée de la sortie du parking (s) |
| `wait_duration` | `5.0` | Durée d'attente une fois garé (s) |

## Calibration

Les durées `align_duration`, `reverse_duration` et `exit_duration` sont en boucle ouverte et doivent être calibrées sur le robot physique. Procéder par itérations :

1. Lancer le système
2. Observer la manœuvre
3. Ajuster les durées dans `parking.launch`
4. Relancer sans rebuilder : `dts devel run -L parking -R d1`

## Dépannage

**Le modèle n'est pas trouvé au démarrage :**  
Vérifier que `best_finetuned_model.pt` est bien dans le dossier `models/`.

**Le topic AprilTag n'existe pas :**  
Vérifier le nom exact du topic avec `rostopic list | grep april` et adapter dans `parking_node.py`.

**Le robot ne suit pas bien la voie :**  
Le modèle a été fine-tuné sur le robot — s'assurer que la calibration des roues est correcte (`dts duckiebot calibrate_wheels d1`).