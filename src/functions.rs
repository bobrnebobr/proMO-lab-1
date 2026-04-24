use std::f64::consts::PI;

/// n-мерная функция Розенброка.
/// Глобальный минимум: x_i = 1, значение 0.
pub fn rosenbrock(x: &[f64]) -> f64 {
    assert!(x.len() >= 2);
    let mut s = 0.0;
    for i in 0..(x.len() - 1) {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        s += a * a + 100.0 * b * b;
    }
    s
}

/// n-мерная функция Растригина.
/// Глобальный минимум: x_i = 0, значение 0.
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
}

/// Функция Изома.
/// Глобальный минимум: (pi, pi), значение -1.
pub fn easom(x: &[f64]) -> f64 {
    assert_eq!(x.len(), 2);
    let (x1, x2) = (x[0], x[1]);
    let dx = x1 - PI;
    let dy = x2 - PI;
    -x1.cos() * x2.cos() * (-(dx * dx + dy * dy)).exp()
}

/// Функция из Desmos.
pub fn desmos(x: &[f64]) -> f64 {
    assert_eq!(x.len(), 2);
    let (xv, yv) = (x[0], x[1]);
    let d = 0.047;
    let t1 = xv * ((10.0 * yv).sin().round() + 2.0);
    let t2 = yv * ((7.0 * xv).sin().round() + 2.0);
    let a = t1 * t1 + yv - 10.0;
    let b = xv + t2 * t2 - 7.0;
    d * (a * a + b * b)
}