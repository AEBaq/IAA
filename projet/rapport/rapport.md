# Lab 4 - Smart Parking (Reverse)
Autrices : Emily Baquerizo & Kimberly Beyeler

Professeure : Marina Zapater

Assistants : Guillaume Chacun & Mehdi Akeddar

Classe : IAA-A

## Introduction
Ce laboratoire est le projet final du cours IAA. L'objectif est de déployer les modèles développés lors des labs précédents sur un DuckieBot physique, pour construire un système de conduite autonome complet.

Le projet choisi est P4 : Smart Parking (Reverse). L'idée est simple : le robot doit conduire de manière autonome sur la voie, reconnaître son spot de parking grâce à un AprilTag avec un identifiant unique, reculer pour se garer, attendre 5 secondes, puis repartir normalement.

Pour réaliser cela, on s'appuie sur plusieurs concepts vus en cours. Le système suit le pipeline modulaire (IAA-01) : capteur → perception → planification → contrôle, chaque bloc étant un nœud ROS indépendant. Le suivi de voie utilise l'apprentissage par imitation développé en Lab 2 (IAA-03). La détection du spot repose sur les AprilTags utilisés pour la localisation (IAA-08). La logique de décision est une machine à états finis (IAA-09). Et les manœuvres de parking sont des commandes en boucle ouverte (IAA-05).

## Architecture du système
### Vue d'ensemble
Le système suit le pipeline modulaire qui décompose le problème en blocs fonctionnels indépendants : 
```
entrée capteur -> percreption -> scene pardong -> planification de mission -> contrôle
```
Chaque bloc traite une abstraction de plus en plus haut niveau de l'environnement, ce qui garantit la modularité, l'interprétabilité et la possibilité d'entraîner chaque composant sépérement.

Dans notre implémentation, chaque bloc correspond à un noeud ROS indépendant qui communique via des topics : 
- `lane_following_node` : suivi de voie par apprentissage par imitation (modèle DuckieNet du Lab 2)
- `parking_node` : noeud principal orchestrant la machine à états et les manoeuvres de parking
- `state_machine` : gestion des transitions d'états de conduite

### Machine à états finis (FSM)
En conduite autonome, la planification de comportement (behaviour planning) consiste à sélectionner le comportement de conduite approprié en fonction de l'état courant du véhicule et de son environnement. Comme présenté en cours (IAA-09), cette couche est classiquement modélisée par une machine à états finis (FSM), dont les transitions sont gouvernées par les sorties des modules de perception.

La FSM démarre en état LANE_FOLLOWING, où le robot conduit normalement. Dès qu'un AprilTag correspondant à l'ID du spot assigné est détecté, le système passe en PARKING_DETECTED. Le robot avance alors encore un peu en ligne droite (APPROACHING) pour se positionner correctement avant la manœuvre, puis s'aligne face au spot (ALIGNING), recule pour se garer (REVERSE_PARKING), attend 5 secondes immobile (WAITING), puis repart en marche avant (EXIT) avant de reprendre le lane following.

Les transitions des états ALIGNING, REVERSE_PARKING, WAITING et EXIT sont déclenchées par des timers en boucle ouverte, sans retour capteur. C'est une hypothèse forte : on suppose que l'environnement est suffisamment contrôlé pour que des commandes de durée fixe produisent un comportement reproductible.

### Flux de données ROS
Le nœud lane_following_node s'abonne au topic `/d1/camera_node/image/compressed` et publie sur `/d1/wheels_driver_node/wheels_cmd`. Le nœud AprilTag de Duckietown publie ses détections sur `/d1/apriltag_detector_node/detections`, auquel le parking_node est abonné. Lorsque le tag cible est détecté, le parking_node désactive le lane following via un flag booléen et prend le contrôle exclusif des roues pour exécuter la séquence de manœuvres.

## Choix de conception
### Modèle de lane following
Le suivi de voie repose sur le modèle DuckieNet développé en Lab 2, basé sur l'apprentissage par immitation qui consiste à entraîner un modèle à reproduire le comportement d'un expert à partir de démonstration. L'idée fondamentale est que hard-coder des pratiques de conduite est difficile, il est plus eeficace d'adopter une approche data-driven et apprendre directement depuis des trajectoires expertes. Dans notre cas, l'expert est un mélange de conduite humaine à la mennete et de sortie algorithmiques de l'autopilot Duckietown. Le modèle prend une image RGB en entrée et produit directement les commandes de contrôle `[vel_left, vel_right]`, ce qui correspond à une approche end-to-end. L'architecture choisie est ResNet18 pré-entraîné dur ImageNet, fine-tuné sur les données du robot physique. Ce coix se justifie par plusieurs considérations liées au déploiement sur matériel embarqué : 
- Transfer learning : les featurs extraites des premières couches (texture, contoursm formes) sont directement réutilisables pour la reconnaissance de routes, dans nécessiter un entraînement from scratch coûteux.
- Tête de régression personnalisée : la tête de classification originale est remplacée par Linear(512->128) -> ReLU -> Dropout(0.3)->Linear(128,2), produisant directement [vel_left, vel_right].

Le fine-tuning a été effectué sur un dataset de 7198 échantillons collectés sur robot physique via le pachage ROS `data_collection`, afin de réduire le domain gap entre les images synthétiques d'entraînement et les images réelles. Seuls layer4 et fc ont été entraînés (gel partiel) avec un learning rate réduit à 1e-5 pour éviter l'effacement des features générales apprises sur ImageNet.

### Détection du spot de parking : AprilTag
L'identification du spot de parking assigné repose sur la détection d'AprilTagsm des marqueurs  visuels imprimés, similaire à un QR code, que la caméra du robot peut détecter et dont elle peut lire l'identifiant unique. Ils sont utilisés dans Duckietown pour corriger l'odométrie roues et localiser précisément le robot dams la carte. Dans notre cas, on exploite uniquement la capacité de détection et de lecture de l'identifiant : lorsque le robot voit un AprilTag dont l'ID correspond à son spot assigné, cela déclenche une phase d'approche pendant laquelle le robot avance encore un peu en ligne droite pour se positionner correctement par rapport au spot. Ce délai est nécessaire car la détection peut survenir à une certaine distance du tag, et manœuvrer trop tôt risquerait de mal positionner le robot par rapport à l'emplacement de parking.

### Manoeuvre de parking en boucle ouverte
Les phases de manœuvre (alignement, recul, sortie) sont implémentées en contrôle open-loop : des commandes de durée fixe sont envoyées aux roues sans retour capteur. Comme défini dans le cours, le contrôle en boucle ouverte ne nécessite pas de connaissance de la variable contrôlée, mais requiert une connaissance précise du plan d'action et ne peut pas compenser les perturbations inconnues, ce qui mène à des dérives. Ce choix se justifie par sa simplicité d'implémentation dans un environnement contrôlé, mais constitue la principale limitation de notre système pour atteindre un parking précis.

## Résultats 

### Performance du modèle de lane following
### Ce qui fonctionne

## Analyse et améliorations possibles

## Limites de l'approche open-loop pour le parking
## Limites de l'apprentissage par imitation pour le lane following
## Problèmes rencontrés
## Améliorations prioritaires

## Conclusion 