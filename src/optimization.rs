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
        let mut current_fitness = f(current_x.iter().map(|&x| CReal::constant(x)).collect());

        let mut temp: f64 = 1.0;
        let cooling_rate = (0.00001f64 / temp).powf(1.0 / steps as f64);

        for _ in 0..steps {
            let scale = temp * 2.0;
            let next_x: Vec<f64> = current_x.iter().enumerate().map(|(i, &x)| {
                let delta = rng.random_range(-1.0..1.0) * scale;
                (x + delta).clamp(domain[i].low, domain[i].high)
            }).collect();

            let next_fitness = f(next_x.iter().map(|&x| CReal::constant(x)).collect());

            let is_better = next_fitness.compare_adaptive(&current_fitness, 15) == std::cmp::Ordering::Less;

            if is_better {
                current_x = next_x;
                current_fitness = next_fitness;
            } else {
                let de = next_fitness.get_approx(10).center() - current_fitness.get_approx(10).center();
                if rng.random_bool((-de / temp).exp().min(1.0)) {
                    current_x = next_x;
                    current_fitness = next_fitness;
                }
            }
            temp *= cooling_rate;
        }
        current_x
    }

    pub fn genetic_algorithm<F>(f: F, domain: Vec<Interval>, pop_size: usize, generations: usize) -> Vec<f64>
    where F: Fn(Vec<CReal>) -> CReal {
        let mut rng = ThreadRng::default();
        let dims = domain.len();

        let mut population: Vec<(Vec<f64>, CReal)> = (0..pop_size).map(|_| {
            let x: Vec<f64> = domain.iter().map(|i| rng.random_range(i.low..i.high)).collect();
            let creals = x.iter().map(|&val| CReal::constant(val)).collect();
            let fitness = f(creals);
            (x, fitness)
        }).collect();

        for _ in 0..generations {
            population.sort_by(|a, b| a.1.compare_adaptive(&b.1, 15));
            population.truncate(pop_size / 2);

            while population.len() < pop_size {
                let p1 = &population[rng.random_range(0..population.len())].0;
                let p2 = &population[rng.random_range(0..population.len())].0;

                let alpha = rng.random_range(0.0..1.0);

                let mut child_x: Vec<f64> = (0..dims).map(|i| {
                    alpha * p1[i] + (1.0 - alpha) * p2[i]
                }).collect();

                if rng.random_bool(0.1) {
                    let d = rng.random_range(0..dims);
                    let scale = 0.1;

                    let shift = rng.random_range(-1.0..1.0) * scale;
                    child_x[d] = (child_x[d] + shift).clamp(domain[d].low, domain[d].high);
                }

                let child_creals = child_x.iter().map(|&v| CReal::constant(v)).collect();
                let child_fitness = f(child_creals);

                population.push((child_x, child_fitness));
            }
        }
        population[0].0.clone()
    }
}
