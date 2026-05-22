use std::error::Error;
use ndarray::prelude::*;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Dataset {
    pub train_x: Array2<f64>,
    pub train_y: Array2<f64>,
    pub test_x: Array2<f64>,
    pub test_y: Array2<f64>,
}

pub fn load_csv(path: &str, num_features: usize, num_classes: usize) -> Result<Dataset, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut raw_data: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record.iter()
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();
        raw_data.push(row);
    }

    let mut rng = thread_rng();
    raw_data.shuffle(&mut rng);

    let n = raw_data.len();
    let total_cols = num_features + 1;

    let mut x_all = Array2::zeros((n, num_features));
    let mut y_all = Array2::zeros((n, num_classes));

    for (i, row) in raw_data.iter().enumerate() {
        for j in 0..num_features {
            x_all[[i, j]] = row[j];
        }

        let class_idx = row[num_features] as usize;
        if class_idx < num_classes {
            y_all[[i, class_idx]] = 1.0;
        }
    }

    for mut col in x_all.columns_mut() {
        let mean = col.mean().unwrap_or(0.0);
        let std = col.std(0.0);
        if std > 0.0 {
            col.mapv_inplace(|v| (v - mean) / std);
        }
    }


    let train_size = (n as f64 * 0.8) as usize;

    let (train_x_view, test_x_view) = x_all.view().split_at(Axis(0), train_size);
    let (train_y_view, test_y_view) = y_all.view().split_at(Axis(0), train_size);

    Ok(Dataset {
        train_x: train_x_view.to_owned(),
        train_y: train_y_view.to_owned(),
        test_x: test_x_view.to_owned(),
        test_y: test_y_view.to_owned(),
    })
}