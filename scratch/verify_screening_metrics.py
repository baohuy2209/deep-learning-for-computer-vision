import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

def verify_binormal_theoretical_milestones():
    print("=== THEORETICAL BINORMAL ROC MODEL (d' = 1.864 from Ref-AUC = 0.9063) ===")
    d_prime = 1.864
    
    # Formula: TPR = norm.cdf(d_prime + norm.ppf(FPR))
    # where FPR = 1 - Specificity
    targets = [
        (0.95, "Near current argmax (sub-optimal for screening)"),
        (0.90, "Intermediate benchmark"),
        (0.85, "Near 80% clinical screening target (HD_4 expected: 79.6%)"),
        (0.80, "Exceeds 80% clinical screening target (HD_4 expected: 84.7%)"),
        (0.75, "High sensitivity screening operating point (HD_4 expected: 88.3%)")
    ]
    
    for spec, note in targets:
        fpr = 1.0 - spec
        tpr = norm.cdf(d_prime + norm.ppf(fpr))
        print(f"Target Specificity: {int(spec*100)}% | Expected Sensitivity: {tpr*100:.1f}% | {note}")
        
    # Check assertions for 85% and 80%
    tpr_85 = norm.cdf(d_prime + norm.ppf(0.15))
    tpr_80 = norm.cdf(d_prime + norm.ppf(0.20))
    assert abs(tpr_85 - 0.796) < 0.005, f"tpr_85 mismatch: {tpr_85}"
    assert abs(tpr_80 - 0.847) < 0.005, f"tpr_80 mismatch: {tpr_80}"
    print("\n[PASS] Theoretical binormal milestones exactly match HD_4.pdf Page 1!")

def verify_empirical_operating_point_search():
    print("\n=== EMPIRICAL SPECIFICITY OPERATING POINT CALCULATION ===")
    np.random.seed(42)
    n_samples = 20000
    y_true = np.random.binomial(1, 0.45, n_samples)
    scores = np.where(y_true == 1, np.random.normal(1.864, 1.0, n_samples), np.random.normal(0.0, 1.0, n_samples))
    probs = 1.0 / (1.0 + np.exp(-scores))
    
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    spec = 1.0 - fpr
    roc_auc = auc(fpr, tpr)
    print(f"Empirical Ref-AUC: {roc_auc:.4f}")
    
    results = {}
    target_specificities = (0.80, 0.85, 0.90, 0.95)
    for target_sp in target_specificities:
        valid_idx = np.where(spec >= target_sp)[0]
        if len(valid_idx) == 0:
            sens_at_sp = 0.0
            thresh_at_sp = 1.0
        else:
            best_sub = valid_idx[np.argmax(tpr[valid_idx])]
            sens_at_sp = float(tpr[best_sub])
            thresh_at_sp = float(thresholds[best_sub])
        key_sens = f"Ref_Sensitivity_at_Spec{int(round(target_sp * 100))}"
        key_thresh = f"Ref_Threshold_at_Spec{int(round(target_sp * 100))}"
        results[key_sens] = sens_at_sp
        results[key_thresh] = thresh_at_sp
        print(f"  -> {key_sens}: {sens_at_sp*100:.1f}% (threshold tau = {thresh_at_sp:.4f})")
        
    assert results["Ref_Sensitivity_at_Spec85"] > 0.78, "Spec85 sensitivity failed threshold"
    assert results["Ref_Sensitivity_at_Spec80"] > 0.83, "Spec80 sensitivity failed threshold"
    print("[PASS] Empirical specificity search successfully computes all 4 targets!")

if __name__ == "__main__":
    verify_binormal_theoretical_milestones()
    verify_empirical_operating_point_search()
