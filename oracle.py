import numpy as np
import torch
from torch import optim
from .constraints import constraint_loss

def evaluate_constraint(ins, targets, constraint, net, rollout_func):
    domains = constraint.domains(ins, targets)
    z_batches = general_attack(ins, targets, constraint, domains,
                               100, net, rollout_func)

    _, pos_losses , is_satisfied = constraint_loss(constraint, ins, targets, z_batches, net, rollout_func)

    return torch.mean(pos_losses), torch.mean(is_satisfied.to(torch.float))


def general_attack(input_batch, target_batch, constraint, domains, num_iters, net, rollout_func):
    if len(domains) == 0:
        return None

    z_best = [dom.sample() for dom in domains]
    for z in z_best:
        z.requires_grad = True

    optimizer = optim.SGD(z_best, lr=0.01, momentum=0.99, nesterov=True)

    assert z_best[0].ndim == 3

    for _ in range(num_iters):

        neg_losses, _, _ = constraint_loss(constraint, input_batch, target_batch, z_best, net, rollout_func)

        optimizer.zero_grad()
        avg_neg_loss = torch.mean(neg_losses)
        avg_neg_loss.backward()
        optimizer.step()
        
        # Projected gradient descent
        for i, dom in enumerate(domains):
            z_best[i].data = dom.project(z_best[i])

    return z_best # TODO: support multiple retries
