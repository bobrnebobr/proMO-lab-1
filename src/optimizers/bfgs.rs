use crate::{History, ObjFn};
use crate::gradient::{numerical_gradient, norm};

fn line_search(f: ObjFn, x: &[f64], direction: &[f64], g: &[f64]) -> f64 {
    let c = 1e-4;
    let rho = 0.5;
    let mut alpha = 1.0;

    let fx = f(x);
    let g_dot_d: f64 = g.iter().zip(direction).map(|(a, b)| a * b).sum();

    for _ in 0..50 {
        let x_new: Vec<f64> = x.iter().zip(direction).map(|(xi, di)| xi + alpha * di).collect();
        if f(&x_new) <= fx + c * alpha * g_dot_d {
            return alpha;
        }
        alpha *= rho;
    }
    alpha
}

pub fn bfgs(f: ObjFn, x0: &[f64], max_iter: usize, tol: f64) -> History {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut h = identity(n);

    let mut hist = History::new("BFGS");
    hist.push(&x, f(&x));

    let mut g = numerical_gradient(f, &x);

    for _ in 0..max_iter {
        if norm(&g) < tol {
            break;
        }
        let d: Vec<f64> = (0..n).map(|i| -dot_row(&h, i, &g)).collect();

        let alpha = line_search(f, &x, &d, &g);
        let s: Vec<f64> = d.iter().map(|di| alpha * di).collect();
        let x_new: Vec<f64> = x.iter().zip(&s).map(|(xi, si)| xi + si).collect();
        let g_new = numerical_gradient(f, &x_new);
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();

        let ys: f64 = y.iter().zip(&s).map(|(a, b)| a * b).sum();

        if ys.abs() > 1e-10 {
            let rho = 1.0 / ys;
            h = bfgs_update(&h, &s, &y, rho);
        }

        x = x_new;
        g = g_new;
        hist.push(&x, f(&x));
    }
    hist
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

fn dot_row(m: &[Vec<f64>], i: usize, v: &[f64]) -> f64 {
    m[i].iter().zip(v).map(|(a, b)| a * b).sum()
}

fn bfgs_update(h: &[Vec<f64>], s: &[f64], y: &[f64], rho: f64) -> Vec<Vec<f64>> {
    let n = s.len();
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = if i == j { 1.0 } else { 0.0 } - rho * s[i] * y[j];
        }
    }
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = if i == j { 1.0 } else { 0.0 } - rho * y[i] * s[j];
        }
    }
    let ah = matmul(&a, h);
    let ahb = matmul(&ah, &b);
    let mut res = ahb;
    for i in 0..n {
        for j in 0..n {
            res[i][j] += rho * s[i] * s[j];
        }
    }
    res
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}