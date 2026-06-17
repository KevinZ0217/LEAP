# Copyright note: Adapted from filipradenovic/revisitop (python/evaluate.py).
# https://github.com/filipradenovic/revisitop

import numpy as np


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    ranks : zero-based ranks of positive images
    nres : number of positive images
    """
    nimgranks = len(ranks)
    ap = 0.0
    recall_step = 1.0 / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.0

    return ap


def compute_map(ranks, gnd, kappas=None):
    """
    ranks: shape (db_size, n_queries) — argsort order of DB indices per query.
    gnd: list of dicts with 'ok' and 'junk' keys (see fast_dinov2 / revisitop eval).
    """
    if kappas is None:
        kappas = []

    mean_map = 0.0
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]["ok"])

        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]["junk"])
        except Exception:
            qgndj = np.empty(0)

        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        ap = compute_ap(pos, len(qgnd))
        mean_map = mean_map + ap
        aps[i] = ap

        pos = pos + 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    mean_map = mean_map / (nq - nempty)
    if nq - nempty > 0:
        pr = pr / (nq - nempty)
    else:
        pr = pr * 0.0

    return mean_map, aps, pr, prs
