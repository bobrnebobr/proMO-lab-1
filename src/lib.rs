pub mod functions;
pub mod gradient;
pub mod optimizers;
pub mod plotting;

pub type ObjFn = fn(&[f64]) -> f64;

#[derive(Debug, Clone)]
pub struct History {
    pub points: Vec<Vec<f64>>,
    pub values: Vec<f64>,
    pub method: String,
}

impl History {
    pub fn new(method: &str) -> Self {
        Self {
            points: Vec::new(),
            values: Vec::new(),
            method: method.to_string(),
        }
    }

    pub fn push(&mut self, x: &[f64], f: f64) {
        self.points.push(x.to_vec());
        self.values.push(f);
    }

    pub fn last_point(&self) -> &[f64] {
        self.points.last().expect("history is empty")
    }

    pub fn last_value(&self) -> f64 {
        *self.values.last().expect("history is empty")
    }
}