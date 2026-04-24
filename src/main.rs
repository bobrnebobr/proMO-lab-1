use lab2::functions::{desmos, easom, rastrigin, rosenbrock};
use lab2::optimizers::{adam, bfgs, genetic_algorithm, sgd_momentum, simulated_annealing};
use lab2::plotting::{plot_convergence, plot_trajectories_2d};
use lab2::{History, ObjFn};
use std::f64::consts::PI;
use std::fs;

struct Task {
    name: &'static str,
    f: ObjFn,
    dim: usize,
    domain: Vec<(f64, f64)>,
    f_star: f64,
    plot_bounds: Option<(f64, f64, f64, f64)>,
    starts: Vec<(&'static str, Vec<f64>)>,
}

fn main() -> std::io::Result<()> {
    fs::create_dir_all("plots").ok();

    let tasks = vec![
        Task {
            name: "rosenbrock", f: rosenbrock, dim: 2,
            domain: vec![(-2.0, 2.0); 2], f_star: 0.0,
            plot_bounds: Some((-2.0, 2.0, -1.0, 3.0)),
            starts: vec![
                ("near", vec![0.9, 0.9]),
                ("mid",  vec![-1.2, 1.0]),
                ("far",  vec![-1.8, 1.8]),
            ],
        },
        Task {
            name: "rosenbrock", f: rosenbrock, dim: 5,
            domain: vec![(-2.0, 2.0); 5], f_star: 0.0,
            plot_bounds: None,
            starts: vec![
                ("near", vec![0.9; 5]),
                ("mid",  vec![-1.2; 5]),
                ("far",  vec![-1.8; 5]),
            ],
        },
        Task {
            name: "rastrigin", f: rastrigin, dim: 2,
            domain: vec![(-5.12, 5.12); 2], f_star: 0.0,
            plot_bounds: Some((-5.12, 5.12, -5.12, 5.12)),
            starts: vec![
                ("near", vec![0.3, 0.3]),
                ("mid",  vec![2.5, 2.5]),
                ("far",  vec![4.8, 4.8]),
            ],
        },
        Task {
            name: "rastrigin", f: rastrigin, dim: 5,
            domain: vec![(-5.12, 5.12); 5], f_star: 0.0,
            plot_bounds: None,
            starts: vec![
                ("near", vec![0.3; 5]),
                ("mid",  vec![2.5; 5]),
                ("far",  vec![4.8; 5]),
            ],
        },
        Task {
            name: "easom", f: easom, dim: 2,
            domain: vec![(-10.0, 10.0); 2], f_star: -1.0,
            plot_bounds: Some((-1.0, 8.0, -1.0, 8.0)),
            starts: vec![
                ("near", vec![PI - 0.3, PI - 0.3]),
                ("mid",  vec![1.5, 1.5]),
                ("far",  vec![6.0, 6.0]),
            ],
        },
        Task {
            name: "desmos", f: desmos, dim: 2,
            domain: vec![(-5.0, 5.0); 2], f_star: 0.0,
            plot_bounds: Some((-5.0, 5.0, -5.0, 5.0)),
            starts: vec![
                ("near", vec![2.9, 2.0]),
                ("mid",  vec![0.0, 0.0]),
                ("far",  vec![-4.0, -4.0]),
            ],
        },
    ];

    for task in &tasks {
        for (start_tag, x0) in &task.starts {
            println!("{} (dim={}, start={})", task.name, task.dim, start_tag);

            let runs: Vec<(&str, History)> = vec![
                ("sgd_momentum",        sgd_momentum(task.f, x0, 1e-3, 0.9, 10_000, 1e-6)),
                ("adam",                adam(task.f, x0, 1e-2, 0.9, 0.999, 10_000, 1e-6)),
                ("bfgs",                bfgs(task.f, x0, 1000, 1e-6)),
                ("simulated_annealing", simulated_annealing(task.f, &task.domain, 10_000)),
                ("genetic",             genetic_algorithm(task.f, &task.domain, 100, 200)),
            ];

            for (method, hist) in &runs {
                println!("{:20} iters={:6} f_final={:.6e}",method, hist.values.len().saturating_sub(1), hist.last_value());
            }

            let refs: Vec<(&str, &History)> = runs.iter().map(|(m, h)| (*m, h)).collect();

            let conv_path = format!("plots/conv_{}_d{}_{}.html", task.name, task.dim, start_tag);
            let conv_title = format!("Сходимость: {} (dim={}, start={})", task.name, task.dim, start_tag);
            plot_convergence(&conv_path, &conv_title, &refs, task.f_star);

            if let Some(bounds) = task.plot_bounds {
                let traj_path = format!("plots/traj_{}_{}.html", task.name, start_tag);
                let traj_title = format!("Траектории: {} (start={})", task.name, start_tag);
                plot_trajectories_2d(&traj_path, &traj_title, task.f, bounds, &refs);
            }

            println!();
        }
    }
    Ok(())
}