"""T3 DAGSNet seed41, same frozen B21 recipe; quota/build/launch/collect."""
import sys
from scripts import run_b21_full as T
from scripts import run_b20_full as B
from scripts import run_w20 as W

T.OUT = W.ROOT / 'runs/t3-s41'
T.ART = W.ROOT / 'artifacts/t3-s41'
B.OUT, B.ART = T.OUT, T.ART
W.OUT, W.ART = T.OUT, T.ART
B.REF[41] = 'CBB41 artifacts/cbb-s41/mh-k-cbb-s41-r50'
T.ARMS = [('T341', 41, T.tier('[[64,64],[64,32],[48,16]]'), 50, 11.5)]
if __name__ == '__main__':
    assert sys.argv[1] in {'quota', 'launch', 'collect'}, 'audit uses CBB41, not CBB42'
    T.main()
