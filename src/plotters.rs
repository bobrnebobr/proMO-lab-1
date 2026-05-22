use plotters::prelude::*;
use crate::History;

pub fn plot_loss_history(histories: &[&History], filename: &str, title: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = histories.iter()
        .flat_map(|h| h.values.iter())
        .fold(0.0, |a: f64, &b| a.max(b));

    let max_steps = histories.iter().map(|h| h.values.len()).max().unwrap_or(0);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..max_steps, 0.0..max_loss)?;

    chart.configure_mesh()
        .x_desc("Iteration / Generation")
        .y_desc("Loss (Error)")
        .draw()?;

    let colors = vec![&BLUE, &RED, &GREEN, &MAGENTA];

    for (i, hist) in histories.iter().enumerate() {
        let color = colors[i % colors.len()];
        chart.draw_series(LineSeries::new(
            hist.values.iter().enumerate().map(|(x, &y)| (x, y)),
            color,
        ))?
            .label(&hist.method)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}