# Projet Interpromo SID 2021

Ensemble des livrables du groupe G8 Innovation.

Notre groupe de projet s'était donné les objectifs suivants :

- Développer un moteur de recherche intelligent qui permettrait de faire de la recommandation d’articles à partir d’une requête et de signaux sur les utilisateurs.
- Conceptualiser les différentes sorties qu’on pourrait obtenir à partir de notre moteur de recherche, c’est à dire le contenu textuel des requêtes, les commentaires d’article, les feedbacks d’utilisateurs ou l’historique des clics par exemple, pour essayer de les valoriser et d’obtenir d’autres fonctionnalités intéressantes.

## Learning-to-rank

Sur la partie moteur de recherche, on utilise un modèle de Learning-to-rank développé avec la librairie PyTerrier. Le LTR caractérise un ensemble de techniques qui utilisent le ML supervisé pour résoudre des problèmes de classement. Appliqué à la recherche d’information, c’est ce qui permet d’ordonner les résultats de nos requêtes.

## Natural Language Proceesing

La partie NLP de notre projet s'est concentré sur la mise à profit des feedbacks des utilisateurs pour produire de la valeur ajoutée, notamment en récupérant le contenu textuel et des infos sur le comportement des utilisateurs à travers leurs interactions avec le moteur de recherche.
On a ainsi produit plusieurs preuves de concepts, notamment sur le moyen d’intégrer les commentaires utilisateurs à la construction de résumés abstractifs, d’améliorer les embeddings obtenus en pondérant les mots cités fréquemment dans les requêtes pour que les résultats affichés soient plus pertinents aux yeux des utilisateurs, et également de mettre à jour les mots-clés utilisés pour le scrapping en faisant du NLP et de l’extraction de features significatives pour quantifier les tendances au sein de l’actualité et conserver les informations les plus utiles pour chacune des veilles de scrapping au cours du temps.

--> Dans l’ensemble, on soutient l’idée que toutes ces informations obtenues via le moteur de recherche ont vraiment une valeur importante parce qu’elles nous permettraient de mieux comprendre l’utilisateur, et donc potentiellement de remodeler les services qu’on lui propose à travers l’intégralité de la chaîne de production, de la phase de scrapping jusqu’à son interaction avec l’interface web.


## Membres du groupe

Chef : Flora Estermann
Chef-adjoint : Katia Belaid
Chef qualité : Tanguy Daurat

Samba Seye
Célya Marcélo
Théo Saccareau
Damien Sonneville
Mélina Audiger
Jordi Mora Fernandez
