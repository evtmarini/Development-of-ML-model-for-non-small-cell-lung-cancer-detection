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

# Αποτελέσματα

Η απόδοση των μοντέλων αξιολογήθηκε με χρήση **Stratified K-Fold Cross Validation** και **HalvingGridSearchCV** για βελτιστοποίηση υπερπαραμέτρων.
Εξετάστηκαν διάφορες μέθοδοι επιλογής χαρακτηριστικών και ταξινομητές, μετρώντας την απόδοση μέσω του **F1-score**.

| Μέθοδος Επιλογής Χαρακτηριστικών | Μοντέλο                           | F1-score  |
| -------------------------------- | --------------------------------- | --------- |
| SES                              | XGBoost                           | **0.827** |
| SES                              | Random Forest                     | 0.821     |
| SES                              | Stacking Ensemble (RF + SVM + GB) | 0.822     |
| SES                              | LightGBM                          | 0.811     |
| RFE-SVM                          | Random Forest                     | 0.733     |
| SES                              | Logistic Regression               | 0.608     |
| SES                              | kNN                               | 0.531     |


### Ερμηνεία Αποτελεσμάτων

Τα αποτελέσματα δείχνουν ότι οι **ensemble μέθοδοι** (XGBoost, Random Forest, LightGBM) υπερέχουν σταθερά σε σχέση με τα γραμμικά μοντέλα, 
καθώς αποτυπώνουν αποτελεσματικά τις μη γραμμικές σχέσεις μεταξύ των ραδιομικών χαρακτηριστικών.
Η μέθοδος επιλογής χαρακτηριστικών **SES** εμφάνισε τη βέλτιστη απόδοση, αναδεικνύοντας τη σημασία της επιλογής στατιστικά ανεξάρτητων και διακριτικών χαρακτηριστικών.

# Ερμηνευσιμότητα

Η ανάλυση **SHAP** και **LIME** χρησιμοποιήθηκε για την ερμηνεία των αποφάσεων του καλύτερου μοντέλου (**SES + XGBoost**).
Τα αποτελέσματα έδειξαν ότι τα **δέκα πιο σημαντικά χαρακτηριστικά** για τη διάκριση των υποτύπων NSCLC είναι:

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

Τα χαρακτηριστικά αυτά αποτυπώνουν την ετερογένεια των όγκων στις περιοχές ενδιαφέροντος (ROI), συνδέονται με τη διαφοροποίηση της υφής, της ομοιογένειας και της πυκνότητας των ιστών, και αποτελούν γνωστούς ραδιομικούς δείκτες διαφοροποίησης μεταξύ αδενοκαρκινωμάτων και πλακωδών καρκινωμάτων.

### Οπτικοποιήσεις

* **`halving_results.png`**: σύγκριση επιδόσεων ταξινομητών ανά μέθοδο επιλογής χαρακτηριστικών.
* **`shap_summary_plot.png`** και **`shap_bar_plot.png`**: συνοπτική και ποσοτική συνεισφορά των σημαντικότερων χαρακτηριστικών.
* **`lime_example.html`**: διαδραστική ανάλυση ενός δείγματος με τοπική εξήγηση της πρόβλεψης του μοντέλου.

Συνολικά, το pipeline επιτυγχάνει **F1-score έως 0.83** με χρήση **SES feature selection** και **XGBoost**, 
επιβεβαιώνοντας την υψηλή προγνωστική ισχύ των ραδιομικών χαρακτηριστικών για τη διάκριση υποτύπων NSCLC.

---


## Συμπεράσματα

Η παρούσα εργασία αποδεικνύει ότι τα ραδιομικά χαρακτηριστικά μπορούν να αξιοποιηθούν αποτελεσματικά για την ταξινόμηση υποτύπων NSCLC μέσω παραδοσιακών μοντέλων μηχανικής μάθησης.
Η ενσωμάτωση εργαλείων ερμηνευσιμότητας (SHAP, LIME) προσθέτει διαφάνεια και αξιοπιστία, καθιστώντας το pipeline χρήσιμο για κλινικές εφαρμογές.
