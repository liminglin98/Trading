macro_model/
============
Archived monthly MACRO factor-model line of work (SETTLED: no tradeable
vol-timing / VRP alpha -- the model's durable use is calibrated WING-SIZING).
Originals of all old scripts + their out/ artifacts + the IV caches live here.

Data anchor: scripts now sit one level below analysis/, so the
Path(__file__).resolve() ROOT anchor was bumped from parents[1] to parents[2]
to keep reaching Trading/Data and Trading/qmt_export.

Run any script from inside this folder, e.g.:  python vrp_universe.py
