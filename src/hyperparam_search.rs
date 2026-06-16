use std::cell::RefCell;
use crate::{
    data_loader::Dataset,
    model::{NeuralNetwork, cross_entropy},
    metrics,
    optimizers,
};

#[derive(Debug, Clone)]
pub struct HyperParams {
    pub hidden1: usize,
    pub hidden2: usize,
    pub lr: f64,
    pub f1: f64,
}

pub fn search(dataset: &Dataset, pop_size: usize, generations: usize) -> HyperParams {
    let domain: Vec<(f64, f64)> = vec![
        (4.0,  64.0),   // hidden1
        (4.0,  32.0),   // hidden2
        (1e-4,  0.1),   // lr
    ];

    let num_features = dataset.train_x.ncols();
    let num_classes  = dataset.train_y.ncols();

    // objective: принимает вектор [hidden1, hidden2, lr], возвращает -F1
    let objective = |params: &[f64]| -> f64 {
        let h1 = (params[0].round() as usize).max(1);
        let h2 = (params[1].round() as usize).max(1);
        let lr = params[2].clamp(1e-4, 0.5);

        let dims = vec![num_features, h1, h2, num_classes];
        let nn_cell = RefCell::new(NeuralNetwork::new(dims.clone()));

        let train_obj = |w: &[f64]| -> f64 {
            let mut nn = nn_cell.borrow_mut();
            nn.set_params(w);
            let pred = nn.forward(&dataset.train_x);
            cross_entropy(&pred, &dataset.train_y)
        };

        let init = NeuralNetwork::new(dims.clone()).get_params();
        let hist = optimizers::adam::adam(&train_obj, &init, lr, 0.9, 0.999, 50, 1e-6);

        let best_w = hist.get_best_x();
        let mut nn_eval = NeuralNetwork::new(dims);
        nn_eval.set_params(&best_w);
        let pred = nn_eval.forward(&dataset.test_x);
        let report = metrics::calculate_f1(&pred, &dataset.test_y);

        -report.f1
    };

    use rand::Rng;
    let mut rng = rand::thread_rng();

    // популяция
    let mut pop: Vec<(Vec<f64>, f64)> = (0..pop_size)
        .map(|_| {
            let x: Vec<f64> = domain.iter().map(|(l, h)| rng.gen_range(*l..*h)).collect();
            let fx = objective(&x);
            (x, fx)
        })
        .collect();

    println!("  [hyperparam search] старт, pop={}, gen={}", pop_size, generations);

    for generation in 0..generations {
        pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let best = &pop[0];
        let h1 = (best.0[0].round() as usize).max(1);
        let h2 = (best.0[1].round() as usize).max(1);
        let lr = best.0[2];
        println!("  gen {:2}/{}: best F1={:.4}  hidden=[{},{}]  lr={:.5}",
                 generation + 1, generations, -best.1, h1, h2, lr);

        pop.truncate(pop_size / 2);

        while pop.len() < pop_size {
            let p1 = pop[rng.gen_range(0..pop.len())].0.clone();
            let p2 = pop[rng.gen_range(0..pop.len())].0.clone();
            let alpha = rng.gen_range(0.0..1.0);
            let mut child: Vec<f64> = (0..domain.len())
                .map(|i| alpha * p1[i] + (1.0 - alpha) * p2[i])
                .collect();

            // мутация
            if rng.gen_bool(0.2) {
                let d = rng.gen_range(0..domain.len());
                let range = domain[d].1 - domain[d].0;
                let shift = rng.gen_range(-0.15..0.15) * range;
                child[d] = (child[d] + shift).clamp(domain[d].0, domain[d].1);
            }

            let fc = objective(&child);
            pop.push((child, fc));
        }
    }

    pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let best = &pop[0];

    HyperParams {
        hidden1: (best.0[0].round() as usize).max(1),
        hidden2: (best.0[1].round() as usize).max(1),
        lr:      best.0[2].clamp(1e-4, 0.5),
        f1:      -best.1,
    }
}
