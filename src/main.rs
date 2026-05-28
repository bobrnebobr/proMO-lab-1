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

const NUM_RUNS: usize = 1;
fn run_with_restarts(
    method_name: &str,
    make_history: &dyn Fn() -> History,
    dims: &[usize],
    dataset: &data_loader::Dataset,
) -> (f64, History) {
    let mut best_f1 = -1.0;
    let mut best_hist: Option<History> = None;

    for run in 0..NUM_RUNS {
        let hist = make_history();
        let mut final_nn = NeuralNetwork::new(dims.to_vec());
        final_nn.set_params(&hist.get_best_x());
        let test_pred = final_nn.forward(&dataset.test_x);
        let report = metrics::calculate_f1(&test_pred, &dataset.test_y);
        println!("    run {}/{}: F1 = {:.4}, loss = {:.4}",
                 run + 1, NUM_RUNS, report.f1, hist.values.last().unwrap());
        if report.f1 > best_f1 {
            best_f1 = report.f1;
            best_hist = Some(hist);
        }
    }
    println!("    {} лучший F1 = {:.4}", method_name, best_f1);
    (best_f1, best_hist.unwrap())
}

fn compare_optimizers(
    name: &str,
    dataset: &data_loader::Dataset,
    hidden1: usize,
    hidden2: usize,
) -> (f64, f64, Vec<History>) {
    println!("\n=== Тестирование на {} ===", name);

    let num_features = dataset.train_x.ncols();
    let num_classes  = dataset.train_y.ncols();
    let dims = vec![num_features, hidden1, hidden2, num_classes];
    println!("Архитектура: {:?}", dims);

    let dims_clone = dims.clone();
    let temp_nn = RefCell::new(NeuralNetwork::new(dims.clone()));
    let objective = |params: &[f64]| -> f64 {
        let mut nn_ref = temp_nn.borrow_mut();
        nn_ref.set_params(params);
        let pred = nn_ref.forward(&dataset.train_x);
        cross_entropy(&pred, &dataset.train_y)
    };

    let (f1_adam, hist_adam) = run_with_restarts(
        "Adam",
        &|| {
            let nn_fresh = NeuralNetwork::new(dims_clone.clone());
            optimizers::adam::adam(&objective, &nn_fresh.get_params(), 0.01, 0.9, 0.999, 50, 1e-6)
        },
        &dims,
        dataset,
    );

    let num_params = NeuralNetwork::new(dims.clone()).get_params().len();
    let domain = vec![(-2.0, 2.0); num_params];
    let (f1_genetic, hist_genetic) = run_with_restarts(
        "Genetic",
        &|| {
            optimizers::genetic::genetic_algorithm(&objective, &domain, 30, 50)
        },
        &dims,
        dataset,
    );

    (f1_adam, f1_genetic, vec![hist_adam, hist_genetic])
}

fn main() {
    let d1 = data_loader::load_csv("src/data/dataset1.csv", 2, 2).expect("d1 load error");
    let d2 = data_loader::load_csv("src/data/dataset2.csv", 4, 2).expect("d2 load error");

    let (f1_a1, f1_g1, hists1) = compare_optimizers("Dataset 1 (d1)", &d1, 16, 8);
    let (f1_a2, f1_g2, hists2) = compare_optimizers("Dataset 2 (d2)", &d2, 16, 8);

    let hists_ref1: Vec<&History> = hists1.iter().collect();
    let _ = plotters::plot_loss_history(&hists_ref1, "loss_d1.png", "Loss Curve - Dataset 1");

    let hists_ref2: Vec<&History> = hists2.iter().collect();
    let _ = plotters::plot_loss_history(&hists_ref2, "loss_d2.png", "Loss Curve - Dataset 2");

    let best_f1_d1 = f1_a1.max(f1_g1);
    let best_f1_d2 = f1_a2.max(f1_g2);

    /*let d3 = data_loader::load_csv("src/data/dataset3.csv", 2, 2).expect("d3 load error");
    let (f1_a3, f1_g3, hists3) = compare_optimizers("Dataset 3 (d3)", &d3, 16, 8);
    let hists_ref3: Vec<&History> = hists3.iter().collect();
    let _ = plotters::plot_loss_history(&hists_ref3, "loss_d3.png", "Loss Curve - Dataset 3");
     let best_f1_d3 = f1_a3.max(f1_g3);*/

    let current_score = 0.3 * best_f1_d1 + 0.3 * best_f1_d2;

    println!("Текущий скор: {:.4}", current_score);
}