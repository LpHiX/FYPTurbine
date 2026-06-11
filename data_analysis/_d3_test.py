"""Quick check: throat-clearing prediction sensitivity to effective throat D3.
Run:  python _d3_test.py     (delete after reading — exploration only)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thesis_figures as tf

for d3mm in (3.8, 3.0, 2.8):
    tf.D3 = d3mm / 1000
    print(f"--- D3 = {d3mm} mm ---")
    for c in tf.hq_throat_clear():
        r = c["q_clear_lps"] / c["q_pred_lps"]
        print(f"  {c['run']:4s} video Q_clear={c['q_clear_lps']:.3f} l/s   "
              f"Lock pred={c['q_pred_lps']:.3f} l/s   ratio={r:.2f}")
