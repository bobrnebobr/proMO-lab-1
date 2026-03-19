#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
}

impl Interval {
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low <= high);
        Self { low, high }
    }

    pub fn width(&self) -> f64 {
        self.high - self.low
    }

    pub fn center(&self) -> f64 {
        (self.low + self.high) / 2.0
    }
}