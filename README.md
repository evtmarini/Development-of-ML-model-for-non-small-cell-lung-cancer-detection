# NSCLC Classification Pipeline

Development of ML model for non-small cell lung cancer detection

# SYNOPSIS

Αυτό το repository περιέχει τον πλήρη κώδικα για την ανάπτυξη ενός Machine Learning pipeline που ταξινομεί υποτύπους μη-μικροκυτταρικού καρκίνου του πνεύμονα (NSCLC) 
με βάση ραδιομικά χαρακτηριστικά εξαγόμενα από αξονικές τομογραφίες (CT).

Το έργο αποτελεί μέρος της διπλωματικής εργασίας:
"Development of ML model for non-small cell lung cancer detection"

# Εισαγωγή

Ο καρκίνος του πνεύμονα είναι η κύρια αιτία θανάτων από καρκίνο παγκοσμίως (~19% το 2022).
Το NSCLC (Non-Small Cell Lung Cancer) αντιπροσωπεύει περίπου το 85% των περιπτώσεων και περιλαμβάνει:

* Αδενοκαρκίνωμα (Adenocarcinoma)
* Πλακώδες καρκίνωμα (Squamous Cell Carcinoma)
* Μεγαλοκυτταρικό καρκίνωμα (Large-Cell Carcinoma)

Η ακριβής διάκριση των υποτύπων είναι κρίσιμη για τη στοχευμένη θεραπεία και τη βελτίωση της πρόγνωσης των ασθενών.
Η Ραδιομική (Radiomics) προσφέρει μια μη επεμβατική προσέγγιση εξαγωγής ποσοτικών χαρακτηριστικών από ιατρικές εικόνες,
τα οποία μπορούν να χρησιμοποιηθούν σε προγνωστικά και διαγνωστικά μοντέλα μηχανικής μάθησης.

---

## Δομή Αρχείων

```
NSCLC Classification/
│
├── main.py
│
├── data/
│   ├── labeled_radiomics_features.csv
│   ├── radiomics_features.csv
│   └── radiomics features.xlsx
│
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── split_and_check.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── explainability.py
│   └── __init__.py
│
├── results/
│   ├── halving_results.csv
│   ├── halving_results.png
│   ├── split_report/
│   │   ├── heterogeneity_centers_heatmap.png
│   │   └── heterogeneity_labels_heatmap.png
│   ├── selected_features/
│   │   ├── selected_Boruta.csv
│   │   ├── selected_CorrSF.csv
│   │   ├── selected_HSIC-LASSO.csv
│   │   ├── selected_LASSO.csv
│   │   ├── selected_mRMR.csv
│   │   ├── selected_ReliefF.csv
│   │   ├── selected_RF-Importance.csv
│   │   ├── selected_RFE-SVM.csv
│   │   └── selected_SES.csv
│   └── results_explainability/
│       ├── shap_bar_plot.png
│       ├── shap_summary_plot.png
│       └── extended/
│           └── lime_example.html
```

---

## Περιγραφή Pipeline

Το pipeline αποτελείται από τα εξής στάδια:

| Στάδιο                     | Module                     | Περιγραφή                                                                                 |
| -------------------------- | -------------------------- | ----------------------------------------------------------------------------------------- |
| 1. Φόρτωση δεδομένων       | `src/load_data.py`         | Ανάγνωση αρχείων CSV, έλεγχος δεδομένων, encoding labels, διαχείριση ελλειπών τιμών       |
| 2. Προεπεξεργασία          | `src/preprocessing.py`     | Κανονικοποίηση, φιλτράρισμα με βάση τη διακύμανση, αφαίρεση συσχετισμένων χαρακτηριστικών |
| 3. Διαχωρισμός             | `src/split_and_check.py`   | Διαχωρισμός training/testing set, έλεγχος ετερογένειας                                    |
| 4. Επιλογή χαρακτηριστικών | `src/feature_selection.py` | Εφαρμογή CorrSF, Boruta, mRMR, ReliefF, RFE, LASSO, HSIC-LASSO, RF-importance, SES        |
| 5. Εκπαίδευση μοντέλων     | `src/models.py`            | Random Forest, Logistic Regression (L1), SVM (RBF kernel)                                 |
| 6. Αξιολόγηση              | `src/evaluation.py`        | HalvingGridSearchCV, Stratified K-Fold CV, ROC-AUC, Accuracy, F1-score                    |
| 7. Οπτικοποίηση            | `src/visualization.py`     | Confusion matrices, heatmaps, ROC curves, feature importance plots                        |
| 8. Ερμηνευσιμότητα         | `src/explainability.py`    | SHAP και LIME για ερμηνεία αποφάσεων του μοντέλου                                         |

---

# Δεδομένα

Τα δεδομένα περιλαμβάνονται στον φάκελο `data/` και αποτελούνται από:

* **Radiomic features** εξαγόμενα με το PyRadiomics από CT εικόνες.
* **labeled_radiomics_features.csv**: περιέχει τα χαρακτηριστικά με τις ετικέτες υποτύπων.
* **radiomics features.xlsx**: αρχείο αναφοράς με περιγραφή χαρακτηριστικών.

---


## Αποτελέσματα

Η απόδοση των μοντέλων αξιολογήθηκε μέσω **Stratified K-Fold Cross Validation** και **HalvingGridSearchCV**, για ταυτόχρονη επιλογή χαρακτηριστικών και βελτιστοποίηση υπερπαραμέτρων.
Εξετάστηκαν συνδυασμοί **Feature Selection (FS)** μεθόδων (SES, RFE-SVM, LASSO) και διαφόρων ταξινομητών.
Η μετρική αξιολόγησης ήταν το **F1-score**, που αποτυπώνει ισορροπία μεταξύ ευαισθησίας και ακρίβειας.

| Μέθοδος Επιλογής Χαρακτηριστικών | Μοντέλο                           | F1-score  |
| -------------------------------- | --------------------------------- | --------- |
| SES                              | XGBoost                           | **0.827** |
| SES                              | Random Forest                     | 0.821     |
| SES                              | Stacking Ensemble (RF + SVM + GB) | 0.822     |
| SES                              | LightGBM                          | 0.811     |
| RFE-SVM                          | SVM (RBF)                         | 0.745     |
| RFE-SVM                          | Random Forest                     | 0.733     |
| LASSO                            | Random Forest                     | 0.821     |
| LASSO                            | Stacking Ensemble (RF + SVM + GB) | 0.813     |
| LASSO                            | XGBoost                           | 0.785     |
| SES                              | Logistic Regression               | 0.608     |

### Ερμηνεία Αποτελεσμάτων

Η καλύτερη συνολική επίδοση επιτεύχθηκε με τη μέθοδο επιλογής χαρακτηριστικών **SES** και τον ταξινομητή **XGBoost**, με **F1-score = 0.827**.
Η προσέγγιση **LASSO + Random Forest** παρουσίασε επίσης ισχυρή απόδοση (F1 = 0.821), υποδεικνύοντας συνέπεια στην ικανότητα των δένδρων ενίσχυσης να διαχειρίζονται μη γραμμικές συσχετίσεις.
Η χρήση **Stacking Ensemble** (RF + SVM + GB) βελτίωσε την απόδοση συγκριτικά με τα επιμέρους μοντέλα, επιβεβαιώνοντας το όφελος της συνδυαστικής μάθησης.

### Βέλτιστες Υπερπαράμετροι

Για τα καλύτερα μοντέλα, οι υπερπαράμετροι που εντοπίστηκαν μέσω HalvingGridSearchCV ήταν:

* **XGBoost (SES):**

  * `n_estimators=800`, `max_depth=7`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.7`
* **Random Forest (SES):**

  * `n_estimators=300`, `max_depth=10`, `max_features='sqrt'`
* **LightGBM (SES):**

  * `num_leaves=63`, `n_estimators=800`, `learning_rate=0.1`
* **SVM (RBF, RFE-SVM):**

  * `C=100`, `gamma=0.01`, `pca__n_components=0.95`

Οι βελτιστοποιημένες αυτές ρυθμίσεις εξασφαλίζουν καλύτερη γενίκευση, μειώνοντας τον κίνδυνο υπερεκπαίδευσης σε σύνολα μικρού μεγέθους, όπως αυτά των ραδιομικών χαρακτηριστικών.

### Ερμηνευσιμότητα

Η ανάλυση **SHAP** και **LIME** εφαρμόστηκε στο βέλτιστο μοντέλο (**SES + XGBoost**) για να αποκαλύψει τη συνεισφορά κάθε χαρακτηριστικού στις προβλέψεις.
Τα **δέκα πιο σημαντικά χαρακτηριστικά** για τη διάκριση των υποτύπων NSCLC ήταν:

1. **GLCM_Correlation**
2. **GLSZM_GrayLevelNonUniformity**
3. **FirstOrder_Entropy**
4. **GLRLM_RunLengthNonUniformity**
5. **GLCM_Contrast**
6. **FirstOrder_MeanAbsoluteDeviation**
7. **GLDM_DependenceVariance**
8. **GLSZM_ZoneEntropy**
9. **Shape_SurfaceArea**
10. **FirstOrder_Kurtosis**

Η υφή και η ετερογένεια του ιστού αντικατοπτρίζουν βιολογικές διαφορές μεταξύ αδενοκαρκινωμάτων και πλακωδών καρκινωμάτων, γεγονός που εξηγεί τη διαγνωστική τους αξία.

### Οπτικοποιήσεις

* **`halving_results.png`**: συγκριτική απεικόνιση F1-score ταξινομητών ανά μέθοδο επιλογής χαρακτηριστικών.
* **`shap_summary_plot.png`** και **`shap_bar_plot.png`**: συνολική και ποσοτική συνεισφορά των σημαντικότερων χαρακτηριστικών.
* **`lime_example.html`**: τοπική εξήγηση προβλέψεων με βάση μεμονωμένα δείγματα.

### Συνοπτική Εκτίμηση

Το pipeline επιτυγχάνει **F1-score έως 0.83** με χρήση **SES feature selection και XGBoost**, αποδεικνύοντας ότι:

* Η κατάλληλη επιλογή ραδιομικών χαρακτηριστικών ενισχύει τη διαγνωστική ακρίβεια.
* Οι ensemble ταξινομητές υπερτερούν των γραμμικών μοντέλων.
* Οι τεχνικές ερμηνευσιμότητας (SHAP/LIME) επιβεβαιώνουν τη βιολογική συνάφεια των χαρακτηριστικών που επηρεάζουν περισσότερο την πρόβλεψη.



## Συμπεράσματα

Η παρούσα εργασία αποδεικνύει ότι τα ραδιομικά χαρακτηριστικά μπορούν να αξιοποιηθούν αποτελεσματικά για την ταξινόμηση υποτύπων NSCLC μέσω παραδοσιακών μοντέλων μηχανικής μάθησης.
Η ενσωμάτωση εργαλείων ερμηνευσιμότητας (SHAP, LIME) προσθέτει διαφάνεια και αξιοπιστία, καθιστώντας το pipeline χρήσιμο για κλινικές εφαρμογές.
