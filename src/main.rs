use lab1::{CReal, Interval, Optimizer};

fn rosenbrock(args: Vec<CReal>) -> CReal {
    let x = args[0].clone();
    let y = args[1].clone();
    let a = CReal::constant(1.0);
    let b = CReal::constant(100.0);

    let t1 = (a - x.clone()).pow2();
    let t2 = b * (y - x.pow2()).pow2();
    t1 + t2
}

fn main() {
    let domain = vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)];

    println!("--- Simulated Annealing ---");
    let sa_res = Optimizer::simulated_annealing(rosenbrock, domain.clone(), 100_000);
    println!("Result: x = {:.5}, y = {:.5}", sa_res[0], sa_res[1]);

    println!("\n--- Genetic Algorithm ---");
    let ga_res = Optimizer::genetic_algorithm(rosenbrock, domain, 1000, 100);
    println!("Result: x = {:.5}, y = {:.5}", ga_res[0], ga_res[1]);
}