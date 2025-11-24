#!/usr/bin/env python3
"""
Generate a Markdown report from the aggregated CSV. The report includes:
 - Top N worst points overall
 - Per-JSON summary (max, mean error) and link to generated PNG
 - Save report to `bad_points_report.md` in the output folder
"""
import os, csv, argparse
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='/work/hwc/SPARSE/figs_val/v16_dataset/bad_points_summary_dataset.csv')
    p.add_argument('--out', default='/work/hwc/SPARSE/figs_val/v16_dataset/bad_points_report.md')
    p.add_argument('--top', type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        print('CSV not found:', args.csv); return
    rows = []
    per_json = defaultdict(list)
    with open(args.csv, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                err = float(r['error_pixels'])
            except:
                continue
            rows.append({'json_file': r['json_file'], 'point_id': r['point_id'], 'error_pixels': err, 'source_log': r.get('source_log','')})
            per_json[r['json_file']].append(err)
    rows_sorted = sorted(rows, key=lambda x: x['error_pixels'], reverse=True)
    md_lines = []
    md_lines.append('# V16 Bad Points Report')
    md_lines.append('')
    md_lines.append('Generated from: `'+args.csv+'`')
    md_lines.append('')
    md_lines.append('## Top {} worst points (overall)'.format(args.top))
    md_lines.append('')
    md_lines.append('| Rank | JSON file | point_id | error_pixels | PNG |')
    md_lines.append('|---:|---|---|---:|---|')
    for i, item in enumerate(rows_sorted[:args.top]):
        json_file = item['json_file']
        pid = item['point_id']
        err = item['error_pixels']
        # PNG path relative
        png_rel = os.path.join(os.path.basename(os.path.dirname(args.csv)), json_file.replace('.json','.png'))
        # If PNG in subdir (aligned_data_for_training_int...), link to png under the same folder
        png_link = png_rel
        md_lines.append('| {} | {} | {} | {:.4f} | [{}]({}) |'.format(i+1, json_file, pid, err, 'png', png_link))
    md_lines.append('')
    md_lines.append('## Per-JSON summary')
    md_lines.append('')
    md_lines.append('| JSON file | count_bad | max_err | mean_err | PNG |')
    md_lines.append('|---|---:|---:|---:|---|')
    import statistics
    for jf, errs in sorted(per_json.items()):
        c = len(errs)
        mx = max(errs)
        mean = statistics.mean(errs)
        png_rel = os.path.join(os.path.basename(os.path.dirname(args.csv)), jf.replace('.json','.png'))
        md_lines.append('| {} | {} | {:.4f} | {:.4f} | [{}]({}) |'.format(jf, c, mx, mean, 'png', png_rel))
    md_lines.append('')
    md_lines.append('---')
    md_lines.append('\n*Report generated automatically.*')
    with open(args.out, 'w', encoding='utf-8') as wf:
        wf.write('\n'.join(md_lines))
    print('Wrote report to', args.out)

if __name__=='__main__':
    main()
