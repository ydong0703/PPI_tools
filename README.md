# PPI Estimator Toolkit

Note: All for Mean estimator

### 1. `naive.py`
**Function:** `naive_estimator(y_labeled, ddof=0)`

- **Input:**  
  `y_labeled` — 1D array of labeled outcomes (length n)  
  `ddof` — degrees of freedom for variance (0 or 1)

- **Output:**  
  `{"est": mean of y_labeled, "sd": standard deviation / sqrt(n)}`


### 2. `ppi_base.py`
**Function:** `ppi_estimator(y_labeled, yhat_labeled, yhat_unlabeled)`

- **Input:**  
  - `y_labeled`: true labels (n,)  
  - `yhat_labeled`: predictions for labeled data (n,)  
  - `yhat_unlabeled`: predictions for unlabeled data (N-n,)

- **Output:**  
  `{"est": PPI estimate, "sd": estimated standard deviation}`


### 3. `ppi_safe.py`  (Safe PPI / PPI++)
**Function:** `ppi_pp_estimator(y_labeled, yhat_labeled, yhat_unlabeled, ddof=0)`

- **Input:**  
  same as above

- **Output:**  
  `{"est": PPI++ estimate, "sd": standard deviation, "omega": optimal weight}`


### 4. `sada.py`
**Function:** `sada_estimator(y_labeled, yhat_labeled, yhat_unlabeled, ddof=0, lambda_reg=0.0, use_pinv=False)`

- **Input:**  
  - `y_labeled`: true labels (n,)  
  - `yhat_labeled`: predictions for labeled data (n, K)  
  - `yhat_unlabeled`: predictions for unlabeled data (N-n, K)  
  - optional `lambda_reg` for ridge regularization  
  - `use_pinv` to use pseudo-inverse for ill-conditioned cases (not completed yet)

- **Output:**  
  `(theta_est, omega_vector, sd_est)`


## Notes
- All arrays should be NumPy arrays (`np.ndarray`).  
- `n` = number of labeled samples, `N` = total samples.  
- `ddof=0` uses n-0, `ddof=1` uses n-1
