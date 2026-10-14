"""Full-data version of the approved V8 factorial, all seven arms train two DAGSNet models."""
from mh.v8_hybrid import Method as PilotMethod, ARMS
from mh.model_packet import Codec


class Method(PilotMethod):
    def __init__(self, run, initialize=True):
        super().__init__(run, initialize)
        self.codec = Codec(self.proxy[0], f'full:{self.arm}:{run.a.seed}')
