"""
detector.py: YOLOv8 person detector with software fault injection.

Fault model (software emulation of hardware errors, as in PyTorchFI / TensorFI style studies):
  activation fault  single bit flip in one element of one Conv2d output, for one inference call
                    (transient error in a MAC unit, pipeline register or on-chip buffer)
  weight fault      single bit flip in one Conv2d weight that stays until the weight is restored
                    (persistent error in weight memory, lasts until the weights are reloaded)
Bits are flipped in the storage format the model actually runs in (FP16 by default), so bit 15 is
the sign, bits 10-14 the exponent and bits 0-9 the mantissa.

The model is the Ultralytics predictor's own module after Conv+BN fusion and FP16 conversion,
i.e. the network as it would be deployed. Hooks sit on every nn.Conv2d and are used to:
  inject   flip bits in the conv output (before the activation function)
  record   per-layer min / max of the conv output (activation range profiling, range detector)
  clip     clamp each conv output to its profiled range (range restriction, Ranger-style baseline)

FakeDetector is a small numpy stand-in with the same interface. It is only used by the
self-test (tests/run_selftest.py) so the pipeline can be checked without a GPU or MOT17.
"""
import zlib

import numpy as np

EMPTY = np.zeros((0, 5), np.float32)


def block_index(name):
    """'model.model.12.cv1.conv' -> 12 (Ultralytics block index), -1 if none."""
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def _xor_bits(flat, idx, bits):
    """Flip bit `bits[k]` of element `idx[k]` in a contiguous 1-D float tensor, in place."""
    import torch
    if flat.dtype in (torch.float16, torch.bfloat16):
        iv, nb = flat.view(torch.int16), 16
    elif flat.dtype == torch.float32:
        iv, nb = flat.view(torch.int32), 32
    else:
        raise TypeError(f"unsupported dtype {flat.dtype}")
    masks = [(1 << b) - ((1 << nb) if b == nb - 1 else 0) for b in bits]  # two's-complement pattern
    m = torch.tensor(masks, dtype=iv.dtype, device=iv.device)
    iv[idx] = torch.bitwise_xor(iv[idx], m)


def _find_net(root, path, max_depth=5):
    """Find the torch module the Ultralytics predictor actually runs. Its location differs between
    Ultralytics versions (predictor.model, predictor.model.model, a backend object, ...), so search
    breadth-first through Ultralytics objects for the outermost nn.Module that holds Conv2d weights."""
    import torch
    queue, seen = [(root, path, 0)], set()
    while queue:
        obj, p, d = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module) and any(isinstance(m, torch.nn.Conv2d) for m in obj.modules()):
            return obj, p
        if d >= max_depth:
            continue
        items = list(getattr(obj, "__dict__", {}).items())
        if isinstance(obj, torch.nn.Module):
            items += list(obj._modules.items())
        for k, v in items:
            if isinstance(v, torch.nn.Module) or type(v).__module__.startswith("ultralytics"):
                queue.append((v, f"{p}.{k}", d + 1))
    raise RuntimeError("could not find the network inside the Ultralytics predictor")


class FaultyYOLO:
    def __init__(self, weights="yolov8s.pt", imgsz=960, half=True, device="0", conf=0.05, iou=0.45, classes=(0,)):
        import logging
        import warnings

        import torch
        from ultralytics import RTDETR, YOLO
        try:  # newer Ultralytics prints a deprecation warning for `half` on every call
            from ultralytics.utils import LOGGER
            LOGGER.setLevel(logging.ERROR)
        except Exception:
            pass
        warnings.filterwarnings("ignore", message=".*half.*")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.torch = torch
        self.yolo = (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, classes=list(classes), device=device, verbose=False)
        if half:
            self.kw["half"] = True
        blank = np.zeros((1080, 1920, 3), np.uint8)
        self.yolo.predict(blank, **self.kw)  # builds the predictor: fused, converted, on device
        net, self.net_path = _find_net(self.yolo.predictor, "predictor")
        self.net = net
        self.convs = [(n, m) for n, m in net.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        self.names = [n for n, _ in self.convs]
        self.L = len(self.convs)
        self.dtype = str(self.convs[0][1].weight.dtype).replace("torch.", "")
        self.nbits = 16 if "16" in self.dtype else 32
        self._sites, self._record, self._clip = {}, False, None
        self._bounds = None
        # Keep only layers whose forward() actually runs. Some Linear modules are never called as modules
        # (e.g. nn.MultiheadAttention.out_proj, used through F.multi_head_attention_forward in RT-DETR),
        # so their hooks never fire and activation faults cannot be injected there.
        self._register_hooks()
        self._record = True
        self._calls = [0] * self.L
        self.yolo.predict(blank, **self.kw)
        self._record = False
        self.n_multi_call = sum(1 for c in self._calls if c > 1)
        called = [i for i in range(self.L) if self._amin[i] is not None]
        self.n_skipped = self.L - len(called)
        if self.n_skipped:
            self.convs = [self.convs[i] for i in called]
            self.names = [n for n, _ in self.convs]
            self.L = len(self.convs)
            self._register_hooks()
        self.w_numel = np.array([m.weight.numel() for _, m in self.convs], np.int64)
        self.infer(blank, record=True)
        self.act_numel = np.array(self._numel, np.int64)  # at 1080p input
        self.meta = {"weights": weights, "net_path": self.net_path, "imgsz": imgsz, "half": half, "dtype": self.dtype, "conf": conf,
                     "nms_iou": iou, "n_conv": self.L, "n_layers_not_called": self.n_skipped,
                     "n_layers_called_repeatedly": self.n_multi_call, "act_elems_1080p": int(self.act_numel.sum()),
                     "weight_elems": int(self.w_numel.sum()), "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "cpu"}
        try:
            import ultralytics
            self.meta["ultralytics"] = ultralytics.__version__
        except Exception:
            pass

    def _register_hooks(self):
        for h in getattr(self, "_handles", []):
            h.remove()
        self._handles = [m.register_forward_hook(self._make_hook(i)) for i, (_, m) in enumerate(self.convs)]
        self._amin, self._amax, self._numel = [None] * self.L, [None] * self.L, [0] * self.L
        self._calls = [0] * self.L

    def _make_hook(self, i):
        torch = self.torch

        def hook(mod, inp, out):
            # A module can run several times in one forward pass (RT-DETR reuses e.g. the query position
            # MLP in every decoder layer). Faults go into the first call only (one transient upset), and
            # recorded ranges cover all calls, so clamping to them never changes a clean output.
            c = self._calls[i]
            self._calls[i] = c + 1
            sites = self._sites.get(i) if c == 0 else None
            if sites:
                out = out.contiguous().clone()
                flat = out.view(-1)
                n = flat.numel()
                idx = torch.tensor([min(int(u * n), n - 1) for u, _ in sites], device=out.device)
                _xor_bits(flat, idx, [b for _, b in sites])
            if self._clip is not None:
                out = torch.clamp(out, float(self._clip[0][i]), float(self._clip[1][i]))
            if self._record:
                if c == 0 or self._amin[i] is None:
                    self._numel[i] = out.numel()
                    self._amin[i], self._amax[i] = out.amin(), out.amax()
                else:
                    self._amin[i] = torch.minimum(self._amin[i], out.amin())
                    self._amax[i] = torch.maximum(self._amax[i], out.amax())
            return out
        return hook

    def set_bounds(self, lo, hi):
        self._bounds = (np.asarray(lo, float), np.asarray(hi, float))

    def infer(self, img, key=None, act_sites=None, record=False, clip=False):
        """Returns (dets, stats). dets: float32 [N,5] (x1,y1,x2,y2,score), or None if inference raised
        (a detected, unrecoverable error). stats: {'amin','amax'} per conv layer when record=True."""
        self._sites = {}
        for layer, u, bit in act_sites or ():
            self._sites.setdefault(int(layer), []).append((float(u), int(bit)))
        self._record = record
        self._calls = [0] * self.L
        self._clip = self._bounds if clip else None
        if clip and self._bounds is None:
            raise RuntimeError("clip=True needs set_bounds() first")
        try:
            r = self.yolo.predict(img, **self.kw)[0]
            b = r.boxes
            dets = (np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]).astype(np.float32)
                    if len(b) else EMPTY.copy())
        except Exception:
            dets = None
        finally:
            self._sites, self._clip = {}, None
        stats = None
        if record:
            t = self.torch
            stats = {"amin": t.stack([a.float() for a in self._amin]).cpu().numpy(),
                     "amax": t.stack([a.float() for a in self._amax]).cpu().numpy()}
            self._record = False
        return dets, stats

    def flip_weight(self, layer, u, bit):
        """Flip one weight bit in place. Calling again with the same arguments restores it exactly."""
        w = self.convs[layer][1].weight
        assert w.is_contiguous()
        with self.torch.no_grad():
            flat = w.data.view(-1)
            n = flat.numel()
            idx = min(int(u * n), n - 1)
            _xor_bits(flat, self.torch.tensor([idx], device=flat.device), [int(bit)])
        return idx

    def weight_checksum(self):
        """Exact checksum of all conv weights (used to verify that injected faults were undone)."""
        h = 0
        for _, m in self.convs:
            h = zlib.crc32(m.weight.detach().cpu().numpy().tobytes(), h)
        return h


class FakeDetector:
    """Numpy stand-in for tests. Golden output is a deterministic synthetic crowd; faults perturb it
    with severities that depend on the flipped bit, so every code path downstream gets exercised."""

    def __init__(self, n_layers=23):
        self.names = [f"model.model.{i}.conv" for i in range(n_layers)]
        self.L, self.nbits, self.dtype = n_layers, 16, "float16"
        self.act_numel = np.linspace(4e6, 2e5, n_layers).astype(np.int64)
        self.w_numel = np.linspace(1e3, 2e5, n_layers).astype(np.int64)
        self._wf, self._bounds = {}, None
        self.meta = {"weights": "fake", "dtype": self.dtype, "n_conv": n_layers}

    @staticmethod
    def _h(*a):
        return zlib.crc32(repr(a).encode())

    def _scene(self, seq, f):
        rng = np.random.default_rng(self._h(seq))
        n = 18
        x0, y0 = rng.uniform(100, 1700, n), rng.uniform(300, 700, n)
        vx, vy = rng.normal(0, 2.0, n), rng.normal(0, 0.5, n)
        h = rng.uniform(80, 260, n)
        alive = (f + rng.integers(0, 400, n)) % 400 < 330  # people enter and leave
        x, y = x0 + vx * f, y0 + vy * f
        alive &= (x > 0) & (x < 1900)
        fr = np.random.default_rng(self._h(seq, f))
        cx = x + fr.normal(0, 1.5, n); cy = y + fr.normal(0, 1.5, n)
        hh = h * (1 + fr.normal(0, 0.02, n)); ww = 0.41 * hh
        sc = np.clip(rng.uniform(0.55, 0.95, n) + fr.normal(0, 0.05, n), 0.06, 0.99)
        occl = fr.random(n) < 0.04
        sc[occl] = fr.uniform(0.1, 0.45, occl.sum())
        d = np.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, sc], 1)[alive]
        return d.astype(np.float32)

    def _apply(self, d, layer, bit, rng, clip, persistent_key=None):
        """Severity grows with bit position; exponent bits can create ghosts or remove people."""
        sev = 0.0
        if bit <= 9 or rng.random() < 0.35:
            return d, 0.0
        if clip and bit >= 13 and rng.random() < 0.6:
            return d, 0.0
        if bit in (10, 11, 12) and len(d):
            k = rng.integers(len(d)); d[k, :4] += rng.normal(0, 2 + 3 * (bit - 9), 2).repeat(2); sev = 1.0
        elif bit == 13 and len(d):
            k = rng.integers(len(d)); d[k, [0, 2]] += 120; sev = 10.0
        elif bit == 14:
            if rng.random() < 0.5 and len(d):
                keep = np.ones(len(d), bool); keep[rng.choice(len(d), min(3, len(d)), replace=False)] = False
                d = d[keep]
            else:
                pr = np.random.default_rng(self._h(persistent_key)) if persistent_key else rng
                cx, cy = pr.uniform(300, 1600), pr.uniform(400, 800)
                d = np.vstack([d, [[cx - 40, cy - 100, cx + 40, cy + 100, 0.9]]]).astype(np.float32)
            sev = 1e3
        elif bit == 15 and len(d):
            d = d[1:]; sev = 1e2
        return d, sev

    def ground_truth(self, seq, f):
        """Noise-free boxes and identities of the synthetic crowd (for the self-test's ground truth)."""
        rng = np.random.default_rng(self._h(seq))
        n = 18
        x0, y0 = rng.uniform(100, 1700, n), rng.uniform(300, 700, n)
        vx, vy = rng.normal(0, 2.0, n), rng.normal(0, 0.5, n)
        h = rng.uniform(80, 260, n)
        alive = (f + rng.integers(0, 400, n)) % 400 < 330
        x, y = x0 + vx * f, y0 + vy * f
        alive &= (x > 0) & (x < 1900)
        w = 0.41 * h
        return [(i + 1, x[i] - w[i] / 2, y[i] - h[i] / 2, w[i], h[i]) for i in np.flatnonzero(alive)]

    def infer(self, img, key=None, act_sites=None, record=False, clip=False):
        seq, f = key
        d = self._scene(seq, f).copy()
        amax = np.full(self.L, 8.0, np.float32); amin = -amax.copy()
        hits = [(l, b, self._h(seq, f, l, u, b), None) for l, u, b in act_sites or ()]
        hits += [(l, b, self._h(seq, f, l, i, b), (l, i, b)) for (l, i), b in self._wf.items()]
        for layer, bit, hs, pk in hits:
            rng = np.random.default_rng(hs)
            d, sev = self._apply(d, layer, bit, rng, clip, pk)
            if sev and not clip:
                amax[layer:] *= 1 + sev
        stats = {"amin": amin, "amax": amax} if record else None
        return d.astype(np.float32), stats

    def set_bounds(self, lo, hi):
        self._bounds = (lo, hi)

    def flip_weight(self, layer, u, bit):
        idx = min(int(u * self.w_numel[layer]), self.w_numel[layer] - 1)
        k = (layer, idx)
        if k in self._wf:
            del self._wf[k]
        else:
            self._wf[k] = bit
        return idx

    def weight_checksum(self):
        return self._h(sorted(self._wf.items()))


def make_detector(backend="yolo", **kw):
    if backend == "fake":
        return FakeDetector()
    import common as C
    kw.setdefault("classes", tuple(C.CLASSES))
    return FaultyYOLO(**kw)
