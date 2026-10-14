"""Coherent DAGSNet parameter/BN snapshots, including integer BN counters."""
import hashlib
import struct
import zlib
import numpy as np
import torch

HEADER = struct.Struct('<4sBBHIII16sI')


class Codec:
    def __init__(self, model, context):
        self.context = hashlib.sha256(context.encode()).digest()[:16]
        self.fields = [(name, tuple(value.shape), value.numel(), value.dtype,
                        value.is_floating_point()) for name, value in model.state_dict().items()]
        self.size = sum(n for _, _, n, _, floating in self.fields if floating)

    def encode(self, state, base, version, half=False):
        chunks = []
        for name, shape, n, dtype, floating in self.fields:
            value = state[name].detach().cpu().numpy()
            assert tuple(value.shape) == shape
            value = value.astype('<f2' if half and floating else '<f4' if floating else '<i8')
            assert np.isfinite(value).all(), 'nonfinite DAGSNet snapshot'
            chunks.append(value.tobytes())
        body = b''.join(chunks)
        return HEADER.pack(b'MHMD', 1, int(half), 0, base, version, len(body), self.context,
                           zlib.crc32(body)) + body

    def decode(self, packet, device='cpu', expected_base=None):
        magic, schema, half, reserved, base, version, length, context, crc = HEADER.unpack_from(packet)
        body = packet[HEADER.size:]
        assert magic == b'MHMD' and schema == 1 and half in (0, 1) and reserved == 0
        assert context == self.context and length == len(body) and zlib.crc32(body) == crc
        assert expected_base is None or base == expected_base
        state, offset = {}, 0
        for name, shape, n, dtype, floating in self.fields:
            wire = '<f2' if half and floating else '<f4' if floating else '<i8'
            value = np.frombuffer(body, wire, n, offset).copy().reshape(shape)
            assert np.isfinite(value).all()
            state[name] = torch.from_numpy(value).to(device=device, dtype=dtype)
            offset += n * np.dtype(wire).itemsize
        assert offset == length
        return state, base, version


class Sum:
    """Trusted aggregation emulator; the algorithm consumes only a cohort sum."""
    def __init__(self, codec, base, device):
        self.codec, self.base, self.device = codec, base, device
        self.seen, self.total = set(), None

    def add(self, client, packet):
        assert client not in self.seen
        state, _, version = self.codec.decode(packet, self.device, self.base)
        assert version == self.base + 1
        if self.total is None:
            self.total = {name: value.double() if value.is_floating_point() else value.clone()
                          for name, value in state.items()}
        else:
            for name, value in state.items():
                if value.is_floating_point():
                    self.total[name] += value.double()
                else:
                    self.total[name] = torch.maximum(self.total[name], value)
        self.seen.add(client)

    def finish(self, clients):
        assert self.seen == set(range(clients)) and clients >= 5
        return {name: (value / clients).float() if value.is_floating_point() else value
                for name, value in self.total.items()}
