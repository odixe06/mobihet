"""Uncompressed fp32 reference, keeping exact A optimizer/BN/private-KD recipe."""
from mh.v2_methods import Method as V2Method
from mh.v2_math import pack_vector,unpack


class Method(V2Method):
    def aggregate(self,r):
        total,n=self.aggregate_sum.finish(); self.last_mean=total/n
        packet=pack_vector('dense',self.global_vector+self.last_mean,
                           base=self.version,version=r,half=False)
        self.global_vector=unpack(packet,self.R.dev)['values']; self.latest_packet=packet
        for k in range(self.R.K): self.record(packet,r,k,'down','proxy_update')
        self.version=r
