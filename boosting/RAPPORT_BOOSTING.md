# Rapport d'Analyse : Modélisation par Boosting (XGBoost) - Site de Sully

## 1. Justification et Pertinence du Modèle

Dans la continuité de la modélisation des crues de la Loire, nous avons déployé une approche basée sur le **Gradient Boosting** (algorithme **XGBoost**). Si les réseaux de neurones (MLP) sont d'excellents approximateurs universels, les méthodes d'ensemble basées sur les arbres de décision (Tree-based models) se révèlent souvent redoutables sur des données tabulaires structurées.

L'approche XGBoost construit séquentiellement des modèles "faibles" (arbres de décision peu profonds) où chaque nouvel arbre tente de corriger les erreurs (résidus) commises par les précédents. Cette stratégie itérative permet :
1.  **Une gestion fine des non-linéarités** : Capacité à modéliser des seuils de débordement ou des effets de saturation des sols.
2.  **Une robustesse aux valeurs aberrantes** : Meilleure tolérance au bruit que les modèles linéaires classiques.
3.  **Une interprétabilité accrue** : Contrairement à la "boîte noire" du Deep Learning, XGBoost permet de quantifier l'apport de chaque capteur dans la décision finale.

## 2. Méthodologie et Architecture

Le modèle a été entraîné selon un protocole de **Validation Croisée (Cross-Validation)** pour garantir que les performances observées ne sont pas dues au hasard d'un découpage favorable des données.

* **Pré-traitement** : Les données ont été normalisées pour harmoniser les échelles (débits en m³/s vs niveaux en m).
* **Algorithme** : XGBoost Regressor.
* **Fonction de coût** : Minimisation de l'erreur quadratique moyenne (RMSE).
* **Métriques d'évaluation** :
    * **R² (Coefficient de détermination)** : Pour mesurer la qualité de l'ajustement (proche de 1 = parfait).
    * **RMSE (Root Mean Square Error)** : Pour quantifier l'erreur moyenne en mètres.

## 3. Analyse des Performances

Les résultats obtenus sont excellents et surpassent légèrement ceux observés avec les autres approches sur certaines stations. Le modèle démontre une très grande fiabilité globale.

### Synthèse des Résultats par Station

| Station (Lieu) | R² Moyen (CV) | RMSE (Erreur Moyenne) | Performance |
| :--- | :--- | :--- | :--- |
| **Parc Château** | **0.9715** | **20.28 cm** | Excellente. Le modèle capture 97% de la variance du niveau d'eau. |
| **Centre Sully** | **0.9724** | **18.42 cm** | Très haute précision, erreur inférieure à 20cm. |
| **Gare de Sully** | **0.9533** | **13.80 cm** | Très bonne, avec une erreur absolue très faible. |
| **Caserne Pompiers** | **0.9639** | **5.78 cm** | Précision chirurgicale (< 6cm d'erreur moyenne). |

*Note : Les scores R² sont issus de la moyenne en validation croisée, assurant la robustesse du modèle face à de nouvelles données.*

Ces résultats confirment que XGBoost parvient à généraliser les dynamiques hydrauliques complexes du bassin versant sans sur-apprendre (overfitting), grâce à ses mécanismes de régularisation internes.

## 4. Interprétabilité et Importance des Variables

Un avantage majeur de ce modèle par rapport au Réseau de Neurones est la capacité d'extraire l'**Importance des Variables** (Feature Importance). Cela permet de valider la cohérence "physique" du modèle : nous pouvons vérifier que les prédictions ne sont pas basées sur des corrélations fallacieuses.

Les graphiques ci-dessous illustrent quelles variables (débits amont, pluviométrie, historique) pèsent le plus dans la prédiction pour chaque station :

![Importance Variables - Parc Chateau](boosting/importance_parc_chateau.png)
*Figure 1 : Importance des variables pour le Parc Château.*

![Importance Variables - Centre Sully](boosting/importance_centre_sully.png)
*Figure 2 : Importance des variables pour le Centre Sully.*

*(Les graphiques pour la Gare de Sully et la Caserne des Pompiers sont disponibles en annexe).*

Cette transparence est cruciale pour les décideurs : elle permet de comprendre, par exemple, si une alerte est déclenchée principalement par la montée de la Loire en amont ou par une accumulation locale de précipitations.

## 5. Critique : Limites du Modèle

Bien que les performances métriques (R², RMSE) soient supérieures ou égales à celles du MLP, le Boosting présente une limite théorique importante : l'**extrapolation**.

* **Problème des bornes** : Les modèles à base d'arbres ne peuvent pas prédire des valeurs supérieures à celles vues lors de l'entraînement. Si une crue millénale survient avec des niveaux d'eau jamais enregistrés historiquement, XGBoost aura tendance à plafonner ses prévisions au maximum connu.
* **Complémentarité** : C'est ici que le Réseau de Neurones (MLP) garde un intérêt, car il conserve une capacité (bien que risquée) à extrapoler des tendances linéaires au-delà des bornes historiques.

## 6. Conclusion

Le modèle **XGBoost** s'impose comme le candidat le plus performant en termes de précision pure et de stabilité sur le jeu de données actuel.
* Avec des scores **R² > 0.95** partout et jusqu'à **0.97** sur les points critiques.
* Une erreur moyenne contenue entre **5cm et 20cm**.

Il offre le meilleur compromis entre **performance prédictive** et **explicabilité**. Pour une mise en production (Système d'Alerte Précoce), nous recommandons d'utiliser ce modèle comme référence principale, tout en surveillant les bornes d'entrée pour détecter les situations hors-domaine ("Out of Distribution") où l'expertise humaine reste indispensable.