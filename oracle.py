import numpy as np
import torch
from .constraints import constraint_loss

def evaluate_constraint(ins, targets, constraint, net, rollout_func):
    learning_rate = 0.01
    num_iters = 50
    domains = constraint.domains(ins, targets)
    z_batches = general_attack(ins, targets, constraint, domains,
                               num_iters, learning_rate, net, rollout_func)

    _, pos_losses , is_satisfied = constraint_loss(constraint, ins, targets, z_batches, net, rollout_func)

    return torch.mean(pos_losses), torch.mean(is_satisfied.to(torch.float))


def general_attack(input_batch, target_batch, constraint, domains, num_iters, learning_rate, net, rollout_func):
    if len(domains) == 0:
        return None

    z_best = [dom.sample() for dom in domains]
    assert z_best[0].ndim == 3



    for _ in range(num_iters):

        z_current = [z.clone() for z in z_best]
        for z in z_current:
            z.requires_grad = True

        neg_losses, _, _ = constraint_loss(constraint, input_batch, target_batch, z_current, net, rollout_func)

        avg_neg_loss = torch.mean(neg_losses)
        avg_neg_loss.backward()

        # Projected gradient descent
        for i, dom in enumerate(domains):
            updated_z = z_best[i] - learning_rate * z_current[i].grad
            z_best[i] = dom.project(updated_z)

        return z_best # TODO: support multiple retries
