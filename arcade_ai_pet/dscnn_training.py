"""Host DS-CNN construction/export. Training recipe and device integration are separate."""
import numpy as np
from .dscnn_model import SHAPES, STRIDES, GROUPS, validate
from .dscnn_quantize import fold_batch_norm, quantize_layer


def build_model():
    import torch
    class Dscnn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(shape[1]*groups, shape[0], shape[2], stride=stride,
                                    padding=shape[2]//2, groups=groups, bias=False),
                    torch.nn.BatchNorm2d(shape[0]), torch.nn.ReLU6())
                for shape, stride, groups in zip(SHAPES[:5], STRIDES, GROUPS, strict=True)])
            self.head = torch.nn.Linear(3137, 7)

        def forward(self, values):
            if values.ndim != 2 or values.shape[1] != 2113:
                raise ValueError('flat 2113-value logmel vectors required')
            image = values[:,:2048].reshape(-1,1,64,32)
            for block in self.blocks:
                image = block(image)
            return self.head(torch.cat((image.flatten(1), values[:,2048:]), dim=1))
    return Dscnn()


def quantize(model):
    if model.training:
        raise ValueError('BN export requires evaluation mode')
    layers = []
    for block in model.blocks:
        conv, bn = block[0], block[1]
        weight, bias = fold_batch_norm(conv.weight.detach().cpu().numpy(),
            np.zeros(conv.out_channels), bn.running_mean.detach().cpu().numpy(),
            bn.running_var.detach().cpu().numpy(), bn.weight.detach().cpu().numpy(),
            bn.bias.detach().cpu().numpy(), eps=bn.eps)
        layers.append(quantize_layer(weight, bias))
    layers.append(quantize_layer(model.head.weight.detach().cpu().numpy(), model.head.bias.detach().cpu().numpy()))
    validate(layers)
    return layers


from time import monotonic
import math
from .dscnn_model import inputs, scores, weights_bytes
from .domain_training import _inputs, validation_rank, select_threshold, shortlist


def fit(x, y, vx, vy, groups, *, epochs=30, batch_size=256, seed=42,
        device='cpu', max_steps=8000, max_seconds=900):
    import torch
    started = monotonic()
    if (type(epochs) is not int or not 1 <= epochs <= 1000 or
        type(batch_size) is not int or not 1 <= batch_size <= 1024 or
        type(seed) is not int or not 0 <= seed < 2**31 or
        type(max_steps) is not int or not 1 <= max_steps <= 1000000 or
        type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 86400 or
        device not in ('cpu', 'mps')):
        raise ValueError('bounded local CNN training configuration required')
    _inputs(vy, groups)
    for values, labels in ((x,y), (vx,vy)):
        inputs(values)
        if (not isinstance(labels, np.ndarray) or labels.shape != (len(values),) or
            not np.issubdtype(labels.dtype, np.integer) or set(labels.tolist()) != set(range(7))):
            raise ValueError('all seven aligned training/validation classes required')
    if ((len(y) + batch_size - 1) // batch_size) * epochs > max_steps:
        raise ValueError('requested epochs exceed the step budget')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('requested local MPS device unavailable')
    def check_time():
        if monotonic() - started > max_seconds:
            raise TimeoutError('CNN training deadline exceeded; no partial candidate')
    check_time()
    torch.set_num_threads(2); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    model = optimizer = tx = ty = None
    try:
        model = build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
        counts = np.bincount(y, minlength=7)
        weight = 1 / np.maximum(counts, 1); weight /= weight.mean()
        lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device))
        tx = torch.from_numpy(x).to(device)
        ty = torch.from_numpy(y.astype(np.int64)).to(device)
        checkpoints, steps = [], 0
        first_loss = last_loss = None
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(y)).to(device)
            for start in range(0, len(y), batch_size):
                check_time()
                indexes = order[start:start+batch_size]
                optimizer.zero_grad()
                loss = lossfn(model(tx[indexes]), ty[indexes])
                loss.backward(); optimizer.step()
                if device == 'mps':
                    torch.mps.synchronize()  # Bound in-flight work and deadline observations.
                last_loss = float(loss.detach().cpu())
                if not math.isfinite(last_loss):
                    raise ValueError('nonfinite CNN loss')
                if first_loss is None:
                    first_loss = last_loss
                steps += 1
                check_time()
            if epoch % 5 == 0 or epoch == epochs - 1:
                model.eval(); layers = quantize(model)
                logits = scores(vx, layers)
                checkpoints.append(dict(layers=layers, epoch=epoch+1,
                                        rank=validation_rank(vy, logits.argmax(1), groups)))
                check_time()
        candidates = shortlist(checkpoints)
        best = best_key = None
        for row in candidates:
            check_time()
            logits = scores(vx, row['layers'])
            threshold, reports, passed = select_threshold(vy, logits, groups)
            key = (passed, min(r['macro_f1'] for r in reports), sum(r['macro_f1'] for r in reports), -threshold, -row['epoch'])
            if best_key is None or key > best_key:
                best_key = key
                best = dict(layers=row['layers'], selected_epoch=row['epoch'], checkpoint_threshold_q10=threshold)
            check_time()
        best.update(seed=seed, epochs=epochs, batch_size=batch_size, training_device=device,
                    training_steps=steps, checkpoint_candidates_evaluated=len(candidates),
                    first_batch_loss=first_loss, final_batch_loss=last_loss,
                    fit_elapsed_seconds=monotonic()-started, weights_bytes=weights_bytes(best['layers']),
                    selection_metric='calibrated_worst_development_domain_profile')
        return best
    finally:
        del model, optimizer, tx, ty
        if device == 'mps':
            torch.mps.empty_cache()
