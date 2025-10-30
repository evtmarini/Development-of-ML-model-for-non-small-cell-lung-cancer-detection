# NSCLC Classification Pipeline

Development of ML model for non-small cell lung cancer detection

## ΣΥΝΟΨΗ

Αυτό το repository περιέχει τον πλήρη κώδικα και τα απαραίτητα scripts για την ανάπτυξη ενός Machine Learning pipeline που ταξινομεί υποτύπους μη-μικροκυτταρικού καρκίνου του πνεύμονα (NSCLC) 
με βάση ραδιομικά χαρακτηριστικά που εξάγονται από αξονικές τομογραφίες (CT).

Το έργο αναπτύχθηκε στο πλαίσιο της διπλωματικής εργασίας:
"Development of ML model for non-small cell lung cancer detection"
Μεταπτυχιακό Πρόγραμμα «Βιοπληροφορική και Νευροπληροφορική» – Ιόνιο Πανεπιστήμιο.

# Εισαγωγή
Ο καρκίνος του πνεύμονα αποτελεί την κύρια αιτία θανάτων από καρκίνο παγκοσμίως, ευθυνόμενος για περίπου 19% των σχετιζόμενων θανάτων το 2022.
Το NSCLC (Non-Small Cell Lung Cancer) αντιπροσωπεύει περίπου το 85% των περιπτώσεων και περιλαμβάνει τρεις βασικούς υποτύπους:

* Αδενοκαρκίνωμα (Adenocarcinoma)
* Πλακώδες καρκίνωμα (Squamous cell carcinoma)
* Μεγαλοκυτταρικό καρκίνωμα (Large-cell carcinoma)

Η διάκριση των υποτύπων αυτών είναι κρίσιμη για τη στοχευμένη θεραπεία και τη βελτίωση της πρόγνωσης των ασθενών.
Η Ραδιομική (Radiomics) προσφέρει μια μη επεμβατική μέθοδο εξαγωγής ποσοτικών χαρακτηριστικών από ιατρικές εικόνες, 
τα οποία μπορούν να αξιοποιηθούν για προγνωστικά και διαγνωστικά μοντέλα μηχανικής μάθησης.

# Δεδομένα
Τα δεδομένα προήλθαν από CT εικόνες ασθενών με διαγνωσμένο NSCLC και περιλαμβάνουν:

* Radiomic features: 1200+ αριθμητικά χαρακτηριστικά εξαγόμενα με το PyRadiomics (π.χ. first-order, GLCM, GLRLM, GLSZM, NGTDM, shape)
* Labels: Ετικέτες υποτύπων (Adeno / Squamous)

Το αρχείο labeled_radiomics_features.csv περιέχει τα κανονικοποιημένα δεδομένα που χρησιμοποιούνται για εκπαίδευση και αξιολόγηση των μοντέλων.

# Μεθοδολογία
Το pipeline αποτελείται από επτά στάδια, που αντιστοιχούν σε επιμέρους modules Python:

| Στάδιο               | Module                   | Περιγραφή                                                                               |
| -------------------- | ------------------------ | --------------------------------------------------------------------------------------- |
| 1. Data Loading      | src/load_data.py         | Φόρτωση αρχείων CSV, έλεγχος δεδομένων, encoding labels, διαχείριση missing values      |
| 2. Preprocessing     | src/preprocessing.py     | Κανονικοποίηση, φιλτράρισμα χαμηλής διακύμανσης, αφαίρεση συσχετισμένων χαρακτηριστικών |
| 3. Feature Selection | src/feature_selection.py | Επιλογή χαρακτηριστικών με CorrSF, Boruta, RFE, LASSO, RF-importance                    |
| 4. Modeling          | src/models.py            | Εκπαίδευση Random Forest, Logistic Regression (L1), SVM (RBF kernel)                    |
| 5. Evaluation        | src/evaluation.py        | GridSearchCV, Stratified K-Fold CV, ROC-AUC, Accuracy, F1-score                         |
| 6. Visualization     | src/visualization.py     | Παραγωγή γραφημάτων (ROC, Confusion Matrix, Feature Importance)                         |
| 7. Explainability    | src/explainability.py    | Εφαρμογή SHAP και LIME για ερμηνεία αποφάσεων του μοντέλου                              |

## Δομή Αρχείων

```
nsclc-classification-pipeline/
│
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
├── data/
│   └── labeled_radiomics_features.csv
│
├── results/
│   ├── ml_results.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── shap_summary_plot.png
│
├── main.py
└── README.md
```

## Τεχνολογίες

* Python 3.10+
* NumPy, Pandas – Διαχείριση δεδομένων
* scikit-learn – ML και αξιολόγηση μοντέλων
* PyRadiomics – Εξαγωγή ακτινομικών χαρακτηριστικών
* SHAP, LIME – Ερμηνευσιμότητα μοντέλων
* Matplotlib, Seaborn – Οπτικοποιήσεις

#  Εκτέλεση
1. Εκτέλεση του pipeline:

   ```
   python main.py
   ```
2. Τα αποτελέσματα (πίνακες, εικόνες, μετρικές) αποθηκεύονται στον φάκελο results/.

## Αποτελέσματα
* Καλύτερο μοντέλο: Logistic Regression (L1)
* Μέσες επιδόσεις (Stratified 5-Fold CV):

  * Accuracy: 0.86
  * ROC-AUC: 0.90
  * F1-score: 0.84
* Κύρια χαρακτηριστικά (Feature Importance):

  * GLCM_Correlation
  * FirstOrder_Entropy
  * GLSZM_GrayLevelNonUniformity
* Ερμηνευσιμότητα:
  Τα διαγράμματα SHAP δείχνουν ότι οι τιμές υφής και ομοιογένειας συμβάλλουν καθοριστικά στη διάκριση μεταξύ των υποτύπων.

# Συμπεράσματα
Η παρούσα μελέτη απέδειξε ότι τα ραδιομικά χαρακτηριστικά μπορούν να χρησιμοποιηθούν αποτελεσματικά για την ταξινόμηση υποτύπων NSCLC με παραδοσιακές ML μεθόδους.
Η ενσωμάτωση εργαλείων ερμηνευσιμότητας ενισχύει τη διαφάνεια και τη χρησιμότητα των μοντέλων σε κλινικά περιβάλλοντα.
