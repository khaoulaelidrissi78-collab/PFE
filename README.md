# Cartographie du Carbone Organique du Sol au Maroc

Ce projet est développé dans le cadre de mon PFE portant sur la conception d'une plateforme Data Lake Serverless sur AWS basée sur une architecture Médaillon. Cette application web constitue la couche Gold de la plateforme, elle exploite le modèle de classification que j'ai entraîné sur les données transformées pour prédire et cartographier le carbone organique du sol.

## Contexte

J'ai construit un pipeline complet de traitement des données. À partir d'un fichier de parcelles agricoles, j'ai calculé les centroïdes en projection UTM, puis j'ai extrait pour chaque parcelle un ensemble de variables environnementales via Google Earth Engine : des indices satellitaires (NDVI, BSI), des variables topographiques (altitude, pente, TWI), des variables climatiques (précipitations, saisonnalité) et des propriétés du sol (argile, sable, limon, pH, azote, densité apparente). Ces données constituent le jeu d'entraînement de mon modèle.

## Fonctionnalités de l'application

L'application est organisée en quatre onglets. Le premier, Prédiction, permet d'estimer la classe de carbone organique d'une parcelle. L'utilisateur choisit l'emplacement en cliquant sur une carte interactive du Maroc, ce qui renseigne automatiquement la latitude, la longitude et la région. Il ajuste ensuite les caractéristiques de la parcelle à l'aide de curseurs regroupés par thème : indices satellitaires, topographie, climat et sol. La classe estimée (Faible ou Moyen) s'affiche immédiatement avec l'accuracy du modèle et se met à jour à chaque modification. Un message avertit l'utilisateur lorsque le point se trouve hors de la zone couverte par les données d'entraînement.

Le deuxième, Effet d'une variable sur la classe, montre comment la classe prédite évolue lorsqu'une seule variable varie, toutes les autres restant fixées. Il indique les valeurs à partir desquelles la parcelle passe d'une classe à l'autre.

Le troisième, Cartographie, présente une carte des classes de carbone sur l'ensemble du territoire marocain. Elle se recalcule à chaque modification des caractéristiques et indique la part du territoire classée dans chaque catégorie.

Le quatrième, Metrics du modèle, présente les performances du modèle, la frontière entre les classes, l'importance de chaque variable dans la prédiction et la méthodologie employée.

## Le modèle

J'ai entraîné plusieurs algorithmes et retenu le Random Forest, qui a donné les meilleurs résultats. Le modèle classe les parcelles en deux catégories, Faible et Moyen, séparées à la médiane de la teneur en carbone. Compte tenu des teneurs observées au Maroc, majoritairement faibles à modérées, j'ai choisi ces deux dénominations.

Pour l'entraînement, j'ai appliqué une zone tampon autour de la frontière entre les deux classes. Cette approche écarte les parcelles situées dans la zone de transition, où l'attribution de classe est ambiguë du fait de la nature continue du carbone du sol. J'ai validé le modèle par validation croisée à cinq plis, et il atteint une accuracy d'environ 0,92 avec toutes les métriques supérieures à 0,90.

## Structure du projet

Le projet est composé des fichiers suivants :

- app.py : contient le code de l'application Streamlit : l'interface, les quatre onglets, les cartes et les graphiques.
- modele_soc.pkl : contient le modèle entraîné, un pipeline scikit-learn qui associe une normalisation des données et un Random Forest. Le fichier contient aussi la liste des variables, la frontière entre les classes et les noms des classes.
- maroc_regions.geojson : contient les limites des douze régions du Maroc, issues de geoBoundaries. Il sert à afficher les régions sur les cartes, à identifier la région d'une parcelle et à délimiter le territoire cartographié.
- requirements.txt : liste les dépendances Python nécessaires, avec la version de scikit-learn fixée pour garantir la compatibilité avec le modèle.
- .streamlit/config.toml : définit le thème visuel de l'application ,mode clair et couleur principale verte.
- README.md décrit l'application et explique comment l'installer et la lancer.