use crate::{History, ObjFn};
use crate::gradient::{numerical_gradient, norm};

pub fn sgd_momentum(
    f: ObjFn,
    x0: &[f64],
    lr: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
) -> History {
    let mut x = x0.to_vec();
    let mut v = vec![0.0; x.len()];
    let mut hist = History::new("SGD+Momentum");
    hist.push(&x, f(&x));

    for _ in 0..max_iter {
        let g = numerical_gradient(f, &x);
        if norm(&g) < tol {
            break;
        }
        for i in 0..x.len() {
            v[i] = beta * v[i] + g[i];
            x[i] -= lr * v[i];
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