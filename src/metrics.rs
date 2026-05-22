use ndarray::prelude::*;

pub struct ClassificationReport {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

pub fn calculate_f1(y_pred_probs: &Array2<f64>, y_true_onehot: &Array2<f64>) -> ClassificationReport {
    let n_classes = y_true_onehot.ncols();
    let mut f1_scores = Vec::new();
    let mut total_precision = 0.0;
    let mut total_recall = 0.0;

    let y_pred: Vec<usize> = y_pred_probs.rows().into_iter()
        .map(|row| row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0)
        .collect();

    let y_true: Vec<usize> = y_true_onehot.rows().into_iter()
        .map(|row| row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0)
        .collect();

    for class in 0..n_classes {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut fn_val = 0.0;

        for i in 0..y_true.len() {
            if y_true[i] == class && y_pred[i] == class { tp += 1.0; }
            else if y_true[i] != class && y_pred[i] == class { fp += 1.0; }
            else if y_true[i] == class && y_pred[i] != class { fn_val += 1.0; }
        }

        let precision = if (tp + fp) > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if (tp + fn_val) > 0.0 { tp / (tp + fn_val) } else { 0.0 };
        let f1 = if (precision + recall) > 0.0 { 2.0 * (precision * recall) / (precision + recall) } else { 0.0 };

        total_precision += precision;
        total_recall += recall;
        f1_scores.push(f1);
    }

    let macro_f1 = f1_scores.iter().sum::<f64>() / n_classes as f64;

    ClassificationReport {
        precision: total_precision / n_classes as f64,
        recall: total_recall / n_classes as f64,
        f1: macro_f1,
    }
}