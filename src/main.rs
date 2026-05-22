mod model;
mod data_loader;
mod metrics;
mod optimizers;
mod gradient;
mod plotters;

use std::cell::RefCell;
use model::{NeuralNetwork, cross_entropy};

pub type ObjFn<'a> = &'a dyn Fn(&[f64]) -> f64;

#[derive(Debug, Clone)]
pub struct History {
    pub points: Vec<Vec<f64>>,
    pub values: Vec<f64>,
    pub method: String,
}

impl History {
    pub fn new(method: &str) -> Self {
        Self { points: Vec::new(), values: Vec::new(), method: method.to_string() }
    }
    pub fn push(&mut self, x: &[f64], f: f64) {
        self.points.push(x.to_vec());
        self.values.push(f);
    }
    pub fn get_best_x(&self) -> Vec<f64> {
        let mut min_idx = 0;
        for (i, &val) in self.values.iter().enumerate() {
            if val < self.values[min_idx] { min_idx = i; }
        }
        self.points[min_idx].clone()
    }
}

fn compare_optimizers(name: &str, dataset: &data_loader::Dataset, hidden_size: usize) -> (f64, f64, Vec<History>) {
    println!("\n=== Тестирование на {} ===", name);

    let num_features = dataset.train_x.ncols();
    let num_classes = dataset.train_y.ncols();
    let dims = vec![num_features, hidden_size, num_classes];

    let nn = NeuralNetwork::new(dims.clone());
    let temp_nn = RefCell::new(NeuralNetwork::new(dims.clone()));

    let objective = |params: &[f64]| -> f64 {
        let mut nn_ref = temp_nn.borrow_mut();
        nn_ref.set_params(params);
        let pred = nn_ref.forward(&dataset.train_x);
        cross_entropy(&pred, &dataset.train_y)
    };

    println!("Запуск Adam...");
    let hist_adam = optimizers::adam::adam(
        &objective,
        &nn.get_params(),
        0.01, 0.9, 0.999, 200, 1e-6
    );

    println!("Запуск Genetic Algorithm...");
    let num_params = nn.get_params().len();
    let domain = vec![(-2.0, 2.0); num_params];
    let hist_genetic = optimizers::genetic::genetic_algorithm(
        &objective,
        &domain,
        40,
        100
    );

    let mut results = Vec::new();
    let hists = vec![hist_adam, hist_genetic];

    for history in &hists {
        let mut final_nn = NeuralNetwork::new(dims.clone());
        final_nn.set_params(&history.get_best_x());
        let test_pred = final_nn.forward(&dataset.test_x);
        let report = metrics::calculate_f1(&test_pred, &dataset.test_y);
        results.push(report.f1);
        println!("  > {}: F1 = {:.4}, Final Loss = {:.4}", history.method, report.f1, history.values.last().unwrap());
    }

    (results[0], results[1], hists)
}

fn main() {
    let d1 = data_loader::load_csv("src/data/dataset1.csv", 2, 2).expect("d1 load error");
    let d2 = data_loader::load_csv("src/data/dataset2.csv", 4, 2).expect("d2 load error");

    let (f1_a1, f1_g1, hists1) = compare_optimizers("Dataset 1 (d1)", &d1, 8);
    let (f1_a2, f1_g2, hists2) = compare_optimizers("Dataset 2 (d2)", &d2, 12);


    let hists_ref1: Vec<&History> = hists1.iter().collect();
    let _ = plotters::plot_loss_history(&hists_ref1, "loss_d1.png", "Loss Curve - Dataset 1");

    let hists_ref2: Vec<&History> = hists2.iter().collect();
    let _ = plotters::plot_loss_history(&hists_ref2, "loss_d2.png", "Loss Curve - Dataset 2");

    let best_f1_d1 = f1_a1.max(f1_g1);
    let best_f1_d2 = f1_a2.max(f1_g2);
    let current_score = 0.3 * best_f1_d1 + 0.3 * best_f1_d2;

    println!("\nТекущий балл: {:.4}", current_score);
}