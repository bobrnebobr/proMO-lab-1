use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(dims: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            let limit = (6.0 / (dims[i] + dims[i+1]) as f64).sqrt();
            layers.push(Layer {
                weights: Array2::random((dims[i], dims[i+1]), Uniform::new(-limit, limit)),
                biases: Array2::zeros((1, dims[i+1])),
            });
        }
        NeuralNetwork { layers }
    }
    
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut current = input.clone();
        let num_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            let z = current.dot(&layer.weights) + &layer.biases;
            if i < num_layers - 1 {
                current = z.mapv(|x| if x > 0.0 { x } else { 0.0 });
            } else {
                current = softmax(&z);
            }
        }
        current
    }
    
    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.weights.iter());
            params.extend(layer.biases.iter());
        }
        params
    }
    
    pub fn set_params(&mut self, params: &[f64]) {
        let mut offset = 0;
        for layer in &mut self.layers {
            let w_len = layer.weights.len();
            let b_len = layer.biases.len();
            
            let w_flat = Array1::from_vec(params[offset..offset+w_len].to_vec());
            layer.weights = w_flat.into_shape(layer.weights.raw_dim()).unwrap();
            offset += w_len;
            
            let b_flat = Array1::from_vec(params[offset..offset+b_len].to_vec());
            layer.biases = b_flat.into_shape(layer.biases.raw_dim()).unwrap();
            offset += b_len;
        }
    }
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    out
}

pub fn cross_entropy(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
    let epsilon = 1e-15;
    let loss = -(y_true * y_pred.mapv(|v| (v + epsilon).ln())).sum();
    loss / y_pred.nrows() as f64
}