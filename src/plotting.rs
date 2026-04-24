use crate::{History, ObjFn};
use plotly::{
    common::{Mode, Title, Marker, Line, Anchor, Font},
    layout::{Axis, AxisType, GridPattern, LayoutGrid, Annotation},
    Contour, Layout, Plot, Scatter,
};
use std::path::Path;

fn method_color(method: &str) -> &'static str {
    match method {
        "sgd_momentum"        => "#1f77b4",
        "adam"                => "#ff7f0e",
        "bfgs"                => "#2ca02c",
        "simulated_annealing" => "#d62728",
        "genetic"             => "#9467bd",
        _                     => "#7f7f7f",
    }
}

pub fn plot_convergence(
    path: impl AsRef<Path>,
    title: &str,
    runs: &[(&str, &History)],
    f_star: f64,
) {
    let mut plot = Plot::new();

    for (method, hist) in runs {
        let xs: Vec<usize> = (0..hist.values.len()).collect();
        let ys: Vec<f64> = hist.values.iter()
            .map(|&v| if v.is_finite() { (v - f_star).abs().max(1e-16) } else { f64::NAN })
            .collect();

        let trace = Scatter::new(xs, ys)
            .mode(Mode::Lines)
            .name(*method)
            .line(Line::new().color(method_color(method)).width(2.0));
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text(title))
        .x_axis(
            Axis::new()
                .title(Title::with_text("Итерация"))
                .range(vec![0.0, 1000.0]),
        )
        .y_axis(
            Axis::new()
                .title(Title::with_text("|f - f*|"))
                .type_(AxisType::Log)
                .range(vec![-6.0, 6.0]),
        );

    plot.set_layout(layout);
    plot.write_html(path.as_ref());
}

pub fn plot_trajectories_2d(
    path: impl AsRef<Path>,
    title: &str,
    f: ObjFn,
    bounds: (f64, f64, f64, f64),
    runs: &[(&str, &History)],
) {
    let (xmin, xmax, ymin, ymax) = bounds;
    let n = 120usize;
    let xs: Vec<f64> = (0..=n).map(|i| xmin + (xmax - xmin) * i as f64 / n as f64).collect();
    let ys: Vec<f64> = (0..=n).map(|j| ymin + (ymax - ymin) * j as f64 / n as f64).collect();

    let z: Vec<Vec<f64>> = ys.iter()
        .map(|&y| xs.iter().map(|&x| f(&[x, y])).collect())
        .collect();

    let mut plot = Plot::new();
    let mut annotations: Vec<Annotation> = Vec::new();

    let cols: f64 = 3.0;
    let rows: f64 = 2.0;

    for (idx, (method, hist)) in runs.iter().enumerate() {
        let axis_id = idx + 1;
        let x_axis = format!("x{}", axis_id);
        let y_axis = format!("y{}", axis_id);

        let col = (idx as f64) % cols;
        let row = (idx as f64 / cols).floor();
        let x_center = (col + 0.5) / cols;
        let y_top = 1.0 - row / rows;

        annotations.push(
            Annotation::new()
                .text(*method)
                .x_ref("paper")
                .y_ref("paper")
                .x(x_center)
                .y(y_top)
                .show_arrow(false)
                .x_anchor(Anchor::Center)
                .y_anchor(Anchor::Bottom)
                .font(Font::new().size(14)),
        );

        let contour_name = format!("{} (contour)", method);
        let contour = Contour::new(xs.clone(), ys.clone(), z.clone())
            .name(&contour_name)
            .show_scale(false)
            .x_axis(&x_axis)
            .y_axis(&y_axis);
        plot.add_trace(contour);

        let margin_x = (xmax - xmin) * 0.05;
        let margin_y = (ymax - ymin) * 0.05;
        let traj_x: Vec<f64> = hist.points.iter()
            .filter(|p| p.len() >= 2 && p[0].is_finite() && p[1].is_finite())
            .map(|p| p[0].clamp(xmin - margin_x, xmax + margin_x))
            .collect();
        let traj_y: Vec<f64> = hist.points.iter()
            .filter(|p| p.len() >= 2 && p[0].is_finite() && p[1].is_finite())
            .map(|p| p[1].clamp(ymin - margin_y, ymax + margin_y))
            .collect();

        if !traj_x.is_empty() {
            let traj = Scatter::new(traj_x.clone(), traj_y.clone())
                .mode(Mode::Lines)
                .name(method)
                .line(Line::new().color("red").width(2.0))
                .x_axis(&x_axis)
                .y_axis(&y_axis);
            plot.add_trace(traj);

            let start_name = format!("{} start", method);
            let start = Scatter::new(vec![traj_x[0]], vec![traj_y[0]])
                .mode(Mode::Markers)
                .name(&start_name)
                .marker(Marker::new().color("green").size(10))
                .x_axis(&x_axis)
                .y_axis(&y_axis)
                .show_legend(false);
            plot.add_trace(start);

            let last = traj_x.len() - 1;
            let end_name = format!("{} end", method);
            let finish = Scatter::new(vec![traj_x[last]], vec![traj_y[last]])
                .mode(Mode::Markers)
                .name(&end_name)
                .marker(Marker::new().color("red").size(12))
                .x_axis(&x_axis)
                .y_axis(&y_axis)
                .show_legend(false);
            plot.add_trace(finish);
        }
    }

    let grid = LayoutGrid::new()
        .rows(2)
        .columns(3)
        .pattern(GridPattern::Independent);

    let layout = Layout::new()
        .title(Title::with_text(title))
        .grid(grid)
        .height(900)
        .annotations(annotations);

    plot.set_layout(layout);
    plot.write_html(path.as_ref());
}