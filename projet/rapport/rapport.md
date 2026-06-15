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
La FSM démarre en état LANE_FOLLOWING, où le robot conduit normalement. Dès qu'un AprilTag correspondant à l'ID du spot assigné est détecté, le système passe en PARKING_DETECTED. Le robot avance encore un peu (APPROACHING) pour se positionner correctement par rapport au spot, puis exécute une rotation d'alignement avec des vitesses asymétriques sur les deux roues (ALIGNING), recule dans le spot (REVERSE_PARKING), attend 5 secondes immobile (WAITING), puis repart en marche avant (EXIT) avant de reprendre le lane following. Les vitesses et durées de chaque phase sont des paramètres ROS ajustables sans recompiler, ce qui permet de calibrer la manœuvre itérativement sur le robot physique.

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
Les phases de manœuvre (alignement, recul, sortie) fonctionnent en boucle ouverte : on envoie des commandes de durée fixe aux roues sans regarder ce que le robot fait réellement. Comme vu en cours, ce type de contrôle ne peut pas corriger les erreurs en cours de route, si le robot dérive à cause du sol ou d'un glissement de roue, il ne le détectera pas. C'est simple à mettre en place, mais ça limite la précision du parking.

## Résultats 
Dû aux divers imprévus dont nous avons discuté lors de la présentation, nous n'avons pas de résultat concret sur circuit à présenter...

## Problèmes rencontrés
Nous avons rencontré divers problèmes lors de ce projet. Pour commencer, notre laboratoire n'arrivant pas à suivre la ligne correctement, nous avons dû principalement réparer ce laboratoire avant de pouvoir continuer avec notre code. 

Ayant passé la majorité du temps du labo à réparer le robot le jour où nous souhaitions tester notre solution du labo 2, nous avons décidé de venir en début de semaine afin de tester cela ainsi que la suite du code pour le projet qui avait été écrit entre temps. Malheureusement, nous avons passé plusieurs heures à tenter de faire rouler le robot sans qu'il ne réponde à aucune de nos tentatives (Test de notre solution, test uniquement du lane following corrigé du labo 2, test lane following issue de duckiebot, test pour le faire avancé grâce au keyboard comme vu lors du labo introduction du robot).

Une fois que nous avons réussi à le faire avancé grâce à l'aide de Guillaume, nous voulions tout de même tester le robot avant la présentatîon. Cette fois-ci, nous avons eu des build qui prenaient un temps incommensurable. La VM qui crashait à plusieurs occasions n'a pas été non plus...

Une fois le build effectué, nos problèmes n'étaient pas fini puisque cette fois-ci, nous avions un message d'erreur annonçant que notre disque était plein et que nous devions réglé cela afin de pouvoir lancer notre job.

N'ayant plus le temps et surtout plus la force de continuer à réparer, nous avons décidé de s'arrêter là pour la phase de test.

## Améliorations prioritaires
Afin de tout de même compléter notre 1ère partie de l'implémentation, nous avons effectué une liste d'amélioration à mettre en place afin d'avoir notre version finale:
 1. Tester la distance de détection du tag afin de pouvoir adapter la réaction du robot après détection (continuer à avancer un peu pour s'aligner, ou s'arrêter directement, ainsi que la distance à couvrir jusqu'à la place)
 2. Tester le parking en épis afin de valider les différentes vitesses de roues pour la rotation ou adapter la vitesse des roues si nécessaire
 3. Mettre en place le redressement du parking afin de passer d'un parking en épis, à un parking latéral (Adaptation de la machine d'état, ajout d'une phase de redressement du robot pour finaliser le parking, et adapter la sortie)
 4. Gérer le parking du côté droite et gauche en détectant si le tag se trouve à gauche ou à droite (pour cela, nous pensions split l'image et regarder de quel côté se trouve le tag pour se parquer du même côté)

## Conclusion 
Pour finir, ce projet aurait pu être intéressant à finaliser si nous n'avions pas eu tout ces problèmes avec le robot. Le fait d'avoir qu'un seul pc pouvant communiquer avec le robot n'a pas forcément aidé (problème que nous avions tenté de régler avec l'aide de plusieurs assitants au labo 1 sans succès). Malgré cela, nous avons plusieurs idée de comment mettre cela en place, et nous regrettons de ne pas pouvoir les mettre entièrement en place. Nous avons tout de même pu travailler sur un projet final réunissant pas mal de compétence observé lors des différents laboratoires.

Nous souaitant dire un grand merci aux assitants pour leurs aides durant ces différents laboratoires ainsi qu'à Mme. Zapater pour ce cours. Malgré les couac, ce fut un plaisir d'assister à ce cours.
