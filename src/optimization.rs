use rand::RngExt;
use crate::creal::CReal;
use crate::interval::Interval;
use rand::rngs::ThreadRng;

pub struct Optimizer;

impl Optimizer {
    pub fn simulated_annealing<F>(f: F, domain: Vec<Interval>, steps: usize) -> Vec<f64>
    where F: Fn(Vec<CReal>) -> CReal {
        let mut rng = ThreadRng::default();

        let mut current_x: Vec<f64> = domain.iter().map(|i| i.center()).collect();
        let mut current_energy = f(current_x.iter().map(|&x| CReal::constant(x)).collect())
            .get_approx(2).center();

        let mut temp: f64 = 1.0;
        let min_temp: f64 = 0.00001;
        let cooling_rate = (min_temp / temp).powf(1.0 / steps as f64);

        for _ in 0..steps {
            let scale = temp * 2.0;

            let next_x: Vec<f64> = current_x.iter().enumerate().map(|(i, &x)| {
                let delta = rng.random_range(-1.0..1.0) * scale;
                (x + delta).clamp(domain[i].low, domain[i].high)
            }).collect();

            let next_energy = f(next_x.iter().map(|&x| CReal::constant(x)).collect())
                .get_approx(2).center();

            let delta_e = next_energy - current_energy;
            if delta_e < 0.0 || rng.random_bool((-delta_e / temp).exp().min(1.0)) {
                current_x = next_x;
                current_energy = next_energy;
            }

            temp *= cooling_rate;
        }
        current_x
    }

    pub fn genetic_algorithm<F>(f: F, domain: Vec<Interval>, pop_size: usize, generations: usize) -> Vec<f64>
    where F: Fn(Vec<CReal>) -> CReal {
        let mut rng = ThreadRng::default();
        let dims = domain.len();

        let mut population: Vec<Vec<f64>> = (0..pop_size).map(|_| {
            domain.iter().map(|i| rng.random_range(i.low..i.high)).collect()
        }).collect();

        for _ in 0..generations {
            population.sort_by(|a, b| {
                let fa = f(a.iter().map(|&x| CReal::constant(x)).collect()).get_approx(2).center();
                let fb = f(b.iter().map(|&x| CReal::constant(x)).collect()).get_approx(2).center();
                fa.partial_cmp(&fb).unwrap()
            });

            population.truncate(pop_size / 2);

            while population.len() < pop_size {
                let parent1 = &population[rng.random_range(0..population.len())];
                let parent2 = &population[rng.random_range(0..population.len())];

                let mut child: Vec<f64> = (0..dims).map(|i| {
                    if rng.random_bool(0.5) { parent1[i] } else { parent2[i] }
                }).collect();

                if rng.random_bool(0.1) {
                    let d = rng.random_range(0..dims);
                    child[d] = rng.random_range(domain[d].low..domain[d].high);
                }

                population.push(child);
            }
        }
        population[0].clone()
    }
}