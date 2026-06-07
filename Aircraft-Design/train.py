import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from testloss import TestLoss

FIELD_NAMES = ['Cp', 'Rho', 'U', 'V', 'W', 'Pressure']


def _to_device(batch, device):
    x, fx, y = batch
    return x.to(device), fx.to(device), y.to(device)


def evaluate(model, loader, device, denorm_std=None):
    """Return overall relative-L2 error and per-field relative-L2 error.

    If ``denorm_std`` (the per-field std used to normalize the targets) is
    given, errors are computed on the de-normalized fields; otherwise on the
    normalized fields. Relative L2 is scale-invariant, so this only matters
    when comparing fields of very different magnitude in the per-field report.
    """
    model.eval()
    myloss = TestLoss(size_average=True)
    overall = 0.0
    n_fields = len(FIELD_NAMES)
    per_field = np.zeros(n_fields)
    n = 0
    with torch.no_grad():
        for batch in loader:
            x, fx, y = _to_device(batch, device)
            out = model(x, fx=fx)
            if denorm_std is not None:
                s = torch.as_tensor(denorm_std, device=device)
                out_d, y_d = out * s, y * s
            else:
                out_d, y_d = out, y
            overall += myloss(out_d, y_d).item()
            for c in range(n_fields):
                per_field[c] += myloss(out_d[..., c], y_d[..., c]).item()
            n += 1
    return overall / n, per_field / n


def save_checkpoint(path, epoch, model, optimizer, scheduler):
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}, path)


def train_model(model, train_ds, val_ds, device, args, coef_norm, save_path):
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=True)
    denorm_std = coef_norm['values_std'] if coef_norm is not None else None

    # Resume from an existing checkpoint so a re-submitted (or requeued) job
    # continues from the last completed epoch with its optimizer/scheduler state.
    start_epoch = 0
    if args.resume and os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            print(f'Resuming from {save_path} at epoch {start_epoch}', flush=True)
        else:
            print(f'Found legacy weights-only checkpoint at {save_path}; '
                  f'loading weights but restarting optimizer/scheduler', flush=True)
            model.load_state_dict(ckpt)

    if start_epoch >= args.epochs:
        print(f'Checkpoint already at epoch {start_epoch} >= {args.epochs}; nothing to do.', flush=True)
        return model

    print(f'Training: {len(train_ds)} train / {len(val_ds)} val cases, '
          f'epochs {start_epoch}..{args.epochs - 1}', flush=True)
    start = time.time()
    for ep in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, fx, y = _to_device(batch, device)
            optimizer.zero_grad()
            out = model(x, fx=fx)
            loss = myloss(out, y)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if ep % args.val_iter == 0 or ep == args.epochs - 1:
            val_err, per_field = evaluate(model, val_loader, device, denorm_std)
            pf = '  '.join(f'{n}:{e:.4f}' for n, e in zip(FIELD_NAMES, per_field))
            print(f'Epoch {ep:4d}  train_relL2 {train_loss:.5f}  '
                  f'val_relL2 {val_err:.5f}  [{pf}]', flush=True)
        else:
            print(f'Epoch {ep:4d}  train_relL2 {train_loss:.5f}', flush=True)

        # Checkpoint every epoch so an interrupted job loses at most one epoch.
        save_checkpoint(save_path, ep, model, optimizer, scheduler)

    print(f'Done in {time.time() - start:.1f}s. Saved -> {save_path}', flush=True)
    return model
