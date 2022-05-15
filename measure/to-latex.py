#!/usr/bin/env python3

import json

# CONFIGURATION

# Maps wgpu's GPU names to `(short name, line colour)`
gpu_data = {
    "NVIDIA GeForce GTX 1060 6GB": ("GTX 1060 6GB", "green")
}

# Names of tests who's input size is measured in bytes rather than pixels
test_case_data = {
    "GPU->CPU buffer copy": (True, "GPU$\\rightarrow$CPU buffer copy"),
    "CPU->GPU buffer copy": (True, "CPU$\\rightarrow$GPU buffer copy")
}



# READ JSON FILES

class Table:
    def __init__(self, name: str) -> None:
        self.name = name
        self.is_bytes = False
        self.lines = []

        if name in test_case_data:
            is_bytes, latex_name = test_case_data[name]
            self.is_bytes = is_bytes
            self.name = latex_name

class Line:
    def __init__(self, gpu_name, backend, points, colour) -> None:
        self.gpu_name = gpu_name
        self.backend = backend
        self.points = points
        self.colour = colour

class Point:
    def __init__(self, size, duration_secs) -> None:
        self.size = size
        self.duration_secs = duration_secs

def add_lines(tables, file_path):
    results = json.loads(open(file_path).read())

    # Shorten GPU name if needed
    gpu_name = results["gpu_name"]
    short_name, colour = gpu_data[gpu_name]
    gpu_name = short_name or gpu_name

    for table_name in results["measurements"]:
        # Create a new `Line` for this system's results
        points = [Point(p["size"], p["duration_secs"]) for p in results["measurements"][table_name]]
        line = Line(gpu_name, results["backend"], points, colour)

        if table_name not in tables:
            tables[table_name] = Table(table_name)
        tables[table_name].lines.append(line)

# Read results and convert them from `map<system, map<test, results>>`
#                                 to `map<test, map<system, results>>` 
tables = {} # Indexed by name
add_lines(tables, "results.json")



# Emit latex

def get_factor(max_val):
    for factor, shorthand, name in [
        (1e6 , "M", "million"),
        (1e3 , "k", "thousand"),
        (1   , "" , ""),
        (1e-3, "m", "micro"),
        (1e-6, "µ", "milli")
    ]:
        if max_val > factor:
            return (factor, shorthand, name)
    return (1e-9, "n", "nano")

def get_tick_steps(max_val, ideal_num_steps = 7):
    best_step = 1
    best_num_steps = 1000000

    for power in range(-3, 4):
        for digit in [1, 2, 5]:
            step = digit * 10 ** power
            num_steps = max_val // step
            if abs(num_steps - ideal_num_steps) < abs(best_num_steps - ideal_num_steps):
                best_step = step
                best_num_steps = num_steps
    return best_step

def get_ticks_and_max(max_val):
    step = get_tick_steps(max_val)

    ticks = []
    i = 0
    while True:
        ticks.append(i)
        if i >= max_val:
            return ",".join([str(t) for t in ticks]), i
        i += step

def gen_latex_string(table):
    max_size = max((p.size for l in table.lines for p in l.points))
    max_duration = max((p.duration_secs for l in table.lines for p in l.points))
    size_factor, size_prefix, size_factor_name = get_factor(max_size)
    dur_factor, dur_prefix, _dur_factor_name = get_factor(max_duration)

    size_tick_str, size_max = get_ticks_and_max(max_size / size_factor)
    dur_tick_str, dur_max = get_ticks_and_max(max_duration / dur_factor)

    if table.is_bytes:
        xlabel = f"Data Size ({ size_prefix }B)"
    else:
        xlabel = f"Image Size ({ size_factor_name } pixels)"

    latex_string = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{{ table.name }}},
    xlabel={{{ xlabel }}},
    ylabel={{Average Duration ({ dur_prefix }s)}},
    xmin=0, xmax={ size_max },
    ymin=0, ymax={ dur_max },
    xtick={{{ size_tick_str }}},
    ytick={{{ dur_tick_str }}},
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed,
]
"""

    for line in table.lines:
        point_str = "".join((
            f"({ p.size / size_factor }, { p.duration_secs / dur_factor })"
            for p in line.points
        ))

        latex_string += f"""
\\addplot[color={ line.colour }, mark=square]
    coordinates {{{ point_str }}};
    \\legend{{{line.gpu_name} ({line.backend})}}"""

    latex_string += """

\\end{axis}
\\end{tikzpicture}"""

    return latex_string

# WRITE LATEX TO AN AUX FILE, READY TO BE IMPORTED

with open("../report/tables.tex", "w") as f:
    for t_name in tables:
        f.write(gen_latex_string(tables[t_name]))
        f.write("\n\n")
