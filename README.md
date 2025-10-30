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

# Αποτελέσματα

Τα αποτελέσματα αποθηκεύονται στον φάκελο `results/` και περιλαμβάνουν:

* **halving_results.csv / halving_results.png**: συνοπτικά αποτελέσματα του GridSearch.
* **split_report/**: γραφήματα ετερογένειας των δεδομένων (heatmaps).
* **selected_features/**: σύνολα χαρακτηριστικών που επελέγησαν με κάθε μέθοδο.
* **results_explainability/**: οπτικοποιήσεις ερμηνευσιμότητας.

  * `shap_bar_plot.png`, `shap_summary_plot.png`: σημαντικότερα χαρακτηριστικά βάσει SHAP.
  * `extended/lime_example.html`: διαδραστική απεικόνιση LIME.

---

## Τεχνολογίες

* Python 3.10+
* NumPy, Pandas
* scikit-learn
* PyRadiomics
* SHAP, LIME
* Matplotlib, Seaborn

---

## Εκτέλεση

1. Εκτέλεση pipeline:

   ```
   python main.py
   ```
2. Τα αποτελέσματα παράγονται αυτόματα στον φάκελο `results/`.

---

# Αποτελέσματα & Παρατηρήσεις

* Καλύτερο μοντέλο: Logistic Regression (L1)
* Μέσες επιδόσεις (Stratified 5-Fold CV):

  * Accuracy: 0.86
  * ROC-AUC: 0.90
  * F1-score: 0.84
* Τα σημαντικότερα χαρακτηριστικά προήλθαν κυρίως από τις κατηγορίες GLCM και GLSZM.
* Τα αποτελέσματα SHAP και LIME ανέδειξαν τη συνεισφορά των χαρακτηριστικών υφής (texture) στη διάκριση των υποτύπων.

---

## Συμπεράσματα

Η παρούσα εργασία αποδεικνύει ότι τα ραδιομικά χαρακτηριστικά μπορούν να αξιοποιηθούν αποτελεσματικά για την ταξινόμηση υποτύπων NSCLC μέσω παραδοσιακών μοντέλων μηχανικής μάθησης.
Η ενσωμάτωση εργαλείων ερμηνευσιμότητας (SHAP, LIME) προσθέτει διαφάνεια και αξιοπιστία, καθιστώντας το pipeline χρήσιμο για κλινικές εφαρμογές.
