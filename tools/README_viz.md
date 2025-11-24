# Visualization tools (v16)

This directory contains helper scripts to generate visualizations and reports for the V16 pipeline.

Files
- `run_viz_dataset_cli.py` : CLI batch visualizer. Usage:

```bash
python3 tools/run_viz_dataset_cli.py --data-dirs /path/to/json_dir1 /path/to/json_dir2 \
  --model-path /path/to/model.pth --out-root /path/to/output_root --top-k 5 --workers 4
```

Notes:
- Each worker process loads the model once. Workers >1 will spawn processes and each will load the model; adjust `--workers` to available GPUs/CPUs.
- Output structure: `--out-root/<basename_of_data_dir>/*.png` and `*_bad_points.txt`. An aggregated CSV is written to `--out-root/bad_points_summary_dataset.csv` when using the original dataset script.

- `generate_report.py` : Generate a Markdown report from the aggregated CSV.

```bash
python3 tools/generate_report.py --csv /work/hwc/SPARSE/figs_val/v16_dataset/bad_points_summary_dataset.csv --out /work/hwc/SPARSE/figs_val/v16_dataset/bad_points_report.md --top 30
```

If you want me to integrate `generate_report.py` into a single HTML report with embedded PNG thumbnails, I can extend it.
