use std::{fmt::Write, path::Path};

use itertools::Itertools;
use serde::Deserialize;

fn main() {
    let pc = Computer::new(
        "../measure/results.json",
        "AMD Ryzen 5600X (6x2 cores @ 4.6GHz)",
        "Ubuntu 20.04.4 LTS",
    );
    let latex = format!(
        r#"
    \centering
    \begin{{minipage}}{{0.5\textwidth}}
        \centering
        {}
    \end{{minipage}}\hfill
    \begin{{minipage}}{{0.5\textwidth}}
        \centering
        {}
    \end{{minipage}}
    "#,
        pc.gen_latex_table(false),
        pc.gen_latex_table(true)
    );
    println!("```latex\n{latex}\n```");
    std::fs::write("../report/tables.tex", &latex).unwrap();
}

#[derive(Debug)]
struct Computer {
    cpu_name: String,
    gpu_name: String,
    os: String,
    wgpu_backend: String,
    measurements: Measurements,
}

#[derive(Debug, Deserialize)]
struct Measurements {
    #[serde(rename = "GPU brightness/contrast (compute pass)")]
    gpu_compute: Vec<Point>,
    #[serde(rename = "GPU brightness/contrast (render pass)")]
    gpu_render: Vec<Point>,
    #[serde(rename = "CPU brightness/contrast (f32)")]
    cpu_f32: Vec<Point>,
    #[serde(rename = "CPU brightness/contrast (u8)")]
    cpu_u8: Vec<Point>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Point {
    size: usize,
    duration_secs: f64,
}

impl Computer {
    fn new(file_path: impl AsRef<Path>, cpu_name: &str, os_name: &str) -> Self {
        #[derive(Debug, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct FileContents {
            gpu_name: String,
            backend: String,
            measurements: Measurements,
        }

        let contents: FileContents =
            serde_json::from_str(&std::fs::read_to_string(file_path).unwrap()).unwrap();

        Self {
            gpu_name: contents.gpu_name,
            cpu_name: cpu_name.to_owned(),
            os: os_name.to_owned(),
            wgpu_backend: contents.backend,
            measurements: contents.measurements,
        }
    }

    fn gen_latex_table(&self, just_gpu: bool) -> String {
        let mut lines = vec![
            (
                "GPU compute",
                "blue",
                "triangle",
                &self.measurements.gpu_compute,
            ),
            (
                "GPU render",
                "blue",
                "square",
                &self.measurements.gpu_render,
            ),
        ];

        if !just_gpu {
            lines.push(("CPU (f32)", "red", "triangle", &self.measurements.cpu_f32));
            lines.push(("CPU (u8)", "red", "square", &self.measurements.cpu_u8));
        }

        // Compute max size of the diagram
        let mut max_size = 0;
        let mut max_duration = 0.0;
        for (_name, _col, _mark, pts) in &lines {
            for pt in *pts {
                max_size = max_size.max(pt.size);
                if pt.duration_secs > max_duration {
                    max_duration = pt.duration_secs;
                }
            }
        }

        let x_axis = Axis::new(max_size as f64);
        let y_axis = Axis::new(max_duration);

        let mut latex = format!(
            r#"\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.9\textwidth,
    title={{ {} }},
    xlabel={{ Image Size ({} pixels) }},
    ylabel={{ Average Duration ({}s) }},
    xmin=0, xmax={},
    ymin=0, ymax={},
    xtick={{ {} }},
    ytick={{ {} }},
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed,
]"#,
            if just_gpu {
                "Just GPU compute"
            } else {
                "CPU and GPU compute"
            },
            x_axis.factor_name,
            y_axis.shorthand,
            x_axis.max,
            y_axis.max,
            x_axis.tick_string(),
            y_axis.tick_string()
        );

        for (name, colour, mark, points) in &lines {
            let point_str = points
                .iter()
                .map(|pt| {
                    format!(
                        "({}, {:.5})",
                        pt.size as f64 / x_axis.factor,
                        pt.duration_secs / y_axis.factor
                    )
                })
                .join("");
            write!(
                latex,
                "
\\addplot[color={colour}, mark={mark}]
    coordinates {{ {point_str} }};
    \\addlegendentry{{ {name} }}"
            )
            .unwrap();
        }

        latex.push_str(
            "

\\end{axis}
\\end{tikzpicture}",
        );

        latex
    }
}

#[derive(Debug)]
struct Axis {
    tick_step: f64,
    max: f64,
    factor: f64,
    shorthand: &'static str,
    factor_name: &'static str, // e.g. million
}

impl Axis {
    fn new(max_value: f64) -> Self {
        // First, determine the factor by which we should reduce our values
        let (factor, shorthand, factor_name) = get_factor(max_value);
        let tick_step = get_tick_step(max_value / factor, 7.0);
        let max = (max_value / factor / tick_step).ceil() * tick_step;

        Self {
            tick_step,
            max,
            factor,
            factor_name,
            shorthand,
        }
    }

    fn tick_string(&self) -> String {
        let mut tick = 0.0;
        let mut s = String::new();
        while tick <= self.max {
            if tick != 0.0 {
                s.push_str(", ");
            }
            s.push_str(&tick.to_string());
            tick += self.tick_step;
        }
        s
    }
}

fn get_factor(max_value: f64) -> (f64, &'static str, &'static str) {
    for (factor, shorthand, name) in [
        (1e6, "M", "million"),
        (1e3, "k", "thousand"),
        (1.0, "", ""),
        (1e-3, "m", "micro"),
        (1e-6, "µ", "milli"),
    ] {
        if max_value > factor {
            return (factor, shorthand, name);
        }
    }
    (1e-9, "n", "nano")
}

fn get_tick_step(max_value: f64, ideal_num_steps: f64) -> f64 {
    let mut best_step = 1.0;
    let mut best_num_steps = f64::MAX;

    for power in -3..=3 {
        for digit in [1.0, 2.0, 5.0] {
            let step = digit * 10.0f64.powi(power);
            let num_steps = max_value / step as f64;
            if (num_steps - ideal_num_steps).abs() < (best_num_steps - ideal_num_steps).abs() {
                best_step = step;
                best_num_steps = num_steps;
            }
        }
    }
    best_step
}
