use crate::{History, ObjFn};
use rand::Rng;

pub fn simulated_annealing(
    f: ObjFn,
    domain: &[(f64, f64)],
    steps: usize,
) -> History {
    let mut rng = rand::thread_rng();

    let mut x: Vec<f64> = domain.iter().map(|(l, h)| 0.5 * (l + h)).collect();
    let mut fx = f(&x);

    let mut hist = History::new("SimulatedAnnealing");
    hist.push(&x, fx);

    let mut temp: f64 = 1.0;
    let cooling = (1e-5f64 / temp).powf(1.0 / steps as f64);

    for _ in 0..steps {
        let scale = temp * 2.0;
        let x_new: Vec<f64> = x.iter().enumerate().map(|(i, xi)| {
            let delta = rng.gen_range(-1.0..1.0) * scale;
            (xi + delta).clamp(domain[i].0, domain[i].1)
        }).collect();

        let f_new = f(&x_new);
        let accept = if f_new < fx {
            true
        } else {
            let p = ((fx - f_new) / temp).exp();
            rng.gen_bool(p.min(1.0).max(0.0))
        };

        if accept {
            x = x_new;
            fx = f_new;
        }
        hist.push(&x, fx);
        temp *= cooling;
    }
    hist
}