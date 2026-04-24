use crate::{History, ObjFn};
use crate::gradient::{numerical_gradient, norm};

pub fn adam(
    f: ObjFn,
    x0: &[f64],
    lr: f64,
    beta1: f64,
    beta2: f64,
    max_iter: usize,
    tol: f64,
) -> History {
    let eps = 1e-8;
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut hist = History::new("Adam");
    hist.push(&x, f(&x));

    for t in 1..=max_iter {
        let g = numerical_gradient(f, &x);
        if norm(&g) < tol {
            break;
        }
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            x[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        let fx = f(&x);
        if !fx.is_finite() || x.iter().any(|xi| !xi.is_finite()) {
            hist.push(&x, fx);
            break;
        }
        hist.push(&x, fx);
    }
    hist
}