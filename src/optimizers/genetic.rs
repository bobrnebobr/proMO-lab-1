use crate::{History, ObjFn};
use rand::Rng;

pub fn genetic_algorithm(
    f: ObjFn,
    domain: &[(f64, f64)],
    pop_size: usize,
    generations: usize,
) -> History {
    let mut rng = rand::thread_rng();
    let dims = domain.len();

    let mut pop: Vec<(Vec<f64>, f64)> = (0..pop_size).map(|_| {
        let x: Vec<f64> = domain.iter().map(|(l, h)| rng.gen_range(*l..*h)).collect();
        let fx = f(&x);
        (x, fx)
    }).collect();

    let mut hist = History::new("Genetic");

    for _ in 0..generations {
        pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        hist.push(&pop[0].0, pop[0].1);

        pop.truncate(pop_size / 2);
        while pop.len() < pop_size {
            let p1 = &pop[rng.gen_range(0..pop.len())].0.clone();
            let p2 = &pop[rng.gen_range(0..pop.len())].0.clone();
            let alpha = rng.gen_range(0.0..1.0);
            let mut child: Vec<f64> = (0..dims).map(|i| alpha * p1[i] + (1.0 - alpha) * p2[i]).collect();
            if rng.gen_bool(0.1) {
                let d = rng.gen_range(0..dims);
                let shift = rng.gen_range(-0.1..0.1);
                child[d] = (child[d] + shift).clamp(domain[d].0, domain[d].1);
            }
            let fc = f(&child);
            pop.push((child, fc));
        }
    }
    pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    hist.push(&pop[0].0, pop[0].1);
    hist
}