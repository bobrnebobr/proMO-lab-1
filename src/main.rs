use lab1::{CReal, Interval, Optimizer};

fn rosenbrock(args: Vec<CReal>) -> CReal {
    let n = args.len();
    if n < 2 {
        panic!("Функция Розенброка требует минимум 2 измерения");
    }

    let mut total = CReal::constant(0.0);
    let one = CReal::constant(1.0);
    let hundred = CReal::constant(100.0);

    for i in 0..(n - 1) {
        let x_i = args[i].clone();
        let x_next = args[i + 1].clone();
        let term1 = (one.clone() - x_i.clone()).pow2();
        let term2 = hundred.clone() * (x_next - x_i.pow2()).pow2();

        total = total + term1 + term2;
    }
    total
}

fn main() {
    let domain = vec![Interval::new(-2.0, 2.0); 4];

    println!("--- Simulated Annealing ---");
    let sa_res = Optimizer::simulated_annealing(rosenbrock, domain.clone(), 100_000);
    println!("Result: {:?}", sa_res);

    println!("\n--- Genetic Algorithm ---");
    let ga_res = Optimizer::genetic_algorithm(rosenbrock, domain, 10_000, 1000);
    println!("Result: {:?}", ga_res);
}