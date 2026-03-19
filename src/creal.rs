use std::ops::{Add, Mul, Sub};
use std::sync::Arc;
use crate::interval::Interval;

#[derive(Clone)]
pub struct CReal {
    pub approx: Arc<dyn Fn(u32) -> Interval + Send + Sync>,
}

impl CReal {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(u32) -> Interval + 'static + Send + Sync,
    {
        Self { approx: Arc::new(f) }
    }

    pub fn get_approx(&self, n: u32) -> Interval {
        (self.approx)(n)
    }

    pub fn constant(val: f64) -> Self {
        Self::new(move |_| {
            Interval::new(val, val)
        })
    }

    pub fn pow2(&self) -> Self {
        let inner = self.clone();
        Self::new(move |n| {
            let i = inner.get_approx(n);
            let l2 = i.low * i.low;
            let h2 = i.high * i.high;
            if i.low <= 0.0 && i.high >= 0.0 {
                Interval::new(0.0, l2.max(h2))
            } else {
                Interval::new(l2.min(h2), l2.max(h2))
            }
        })
    }
}

impl Add for CReal {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(move |n| {
            let a = self.get_approx(n);
            let b = rhs.get_approx(n);
            Interval::new(a.low + b.low, a.high + b.high)
        })
    }
}

impl Sub for CReal {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(move |n| {
            let a = self.get_approx(n);
            let b = rhs.get_approx(n);

            if a.low - b.low < a.high - b.high {
                Interval::new(a.low - b.low, a.high - b.high)
            } else {
                Interval::new(a.high - b.high, a.low - b.low)
            }
        })
    }
}

impl Mul for CReal {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(move |n| {
            let a = self.get_approx(n);
            let b = rhs.get_approx(n);
            let p = [a.low * b.low, a.high * b.high, a.low * b.high, a.high * b.low];

            Interval::new(
                p.iter().copied().fold(f64::INFINITY, f64::min),
                p.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            )
        })
    }
}