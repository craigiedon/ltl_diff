import numpy as np
import torch
from torch import optim
from .constraints import constraint_loss

def evaluate_constraint(ins, targets, constraint, net, rollout_func, adversarial):
    domains = constraint.domains(ins, targets)

    if adversarial:
        z_batches = general_attack(ins, targets, constraint, domains,
                                   1, net, rollout_func)
    else:
        z_batches = ins

    _, pos_losses , is_satisfied = constraint_loss(constraint, ins, targets, z_batches, net, rollout_func)

    return torch.mean(pos_losses), torch.mean(is_satisfied.to(torch.float))


def general_attack(input_batch, target_batch, constraint, domains, num_iters, net, rollout_func):
    if len(domains) == 0:
        return None

    z_best = [dom.sample() for dom in domains]
    for z in z_best:
        z.requires_grad = True


    assert z_best[0].ndim == 2

    # optimizer = optim.SGD(z_best, lr=0.01, momentum=0.99, nesterov=True)

    # for _ in range(num_iters):

    #     z_view = torch.stack(z_best).transpose(1, 0)
    #     neg_losses, _, _ = constraint_loss(constraint, input_batch, target_batch, z_view, net, rollout_func)

    #     optimizer.zero_grad()
    #     avg_neg_loss = torch.mean(neg_losses)
    #     avg_neg_loss.backward()
    #     optimizer.step()
        
    #     Projected gradient descent
    #     for i, dom in enumerate(domains):
    #         z_best[i].data = dom.project(z_best[i])

    return torch.stack(z_best).transpose(1,0) #TODO: Support multiple retries
