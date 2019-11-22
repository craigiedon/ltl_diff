from . import ltldiff as ltd
import torch
from .domains import Box

def constraint_loss(constraint, ins, targets, zs, net, rollout_func):
    cond = constraint.condition(zs, ins, targets, net, rollout_func)
    
    neg_losses = ltd.Negate(cond).loss(0)
    losses = cond.loss(0)
    sat = cond.satisfy(0)
        
    return neg_losses, losses, sat


class EventuallyReach:
    def __init__(self, p, epsilon):
        assert p.dim() == 1, "D"
        self.p = p
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        z_in = zs[0]
        weights = net(z_in)
        rollout_traj = rollout_func(z_in[:, 0], z_in[:, 1], weights)[0]

        return ltd.Eventually(
            ltd.EQ(
                ltd.TermDynamic(rollout_traj),
                ltd.TermStatic(self.p),
            ),
            rollout_traj.shape[1]
        )

    def domains(self, ins, targets):
        low_ins = ins - self.epsilon
        high_ins = ins + self.epsilon
        return [Box(low_ins, high_ins)]

class StayInZone:
    def __init__(self, min_bound, max_bound, epsilon):
        assert min_bound.dim() == 1, "Num of dims min bound should be 1: (Spatial Dims)"
        assert max_bound.dim() == 1, "Num of dims for max bound should be 1: (Spatial Dims)"

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        z_ins = zs[0]
        weights = net(z_ins)
        rollout_traj = rollout_func(weights)
        rollout_term = ltd.TermDynamic(rollout_traj)

        return ltd.Always(
            ltd.And([
                ltd.GEQ(rollout_term, ltd.TermStatic(self.min_bound)),
                ltd.LEQ(rollout_term, ltd.TermStatic(self.max_bound))
            ]),
            rollout_traj.shape[1]
        )

    def domains(self, ins, targets):
        low_ins = ins - self.epsilon
        high_ins = ins + self.epsilon
        return [Box(low_ins, high_ins)]


class LipchitzContinuous:
    def __init__(self, smooth_thresh, epsilon):
        self.smooth_thresh = smooth_thresh
        self.epsilon = epsilon

    def constraint(self, zs, ins, targets, net, rollout_func):
        z_ins = zs[0]
        weights_x = net(ins)
        weights_z = net(z_ins)

        rollout_x = rollout_func(weights_x)
        rollout_z = rollout_func(weights_z)

        rollout_diffs = ltd.TermDynamic(torch.abs(rollout_x - rollout_z))
        input_diffs = ltd.TermStatic(self.smooth_thresh * torch.abs(ins - zs))

        return ltd.Always(
            ltd.LEQ(rollout_diffs, input_diffs),
            rollout_x.shape[1]
        )

    def domains(self, ins, targets):
        return [Box(ins - self.epsilon, ins + self.epsilon)]


class DontTipEarly:
    def __init__(self, fixed_orientation, tip_point, min_dist, epsilon):
        assert tip_point.dim() == 2 # N X D
        self.tip_point = tip_point
        self.min_dist = min_dist
        self.fixed_orientation = fixed_orientation
        self.epsilon = epsilon

    def constraint(self, zs, ins, targets, net, rollout_func):
        z_ins = zs[0]
        weights = net(z_ins)
        rollout = rollout_func(weights)

        # Assuming final dimensions are x, y, theta (rotation)
        rotation_terms = ltd.TermStatic(rollout[:, :, 2])

        # Measuring the distance across dimensions...
        dist_to_tip = ltd.TermDynamic(torch.norm(rollout - self.tip_point[:, None, :], dim=2))

        return ltd.Always(
            ltd.Implication(
                ltd.GT(dist_to_tip, self.min_dist),
                ltd.EQ(self.fixed_orientation, rotation_terms)
            ),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        low_in = ins - self.epsilon
        high_in = ins + self.epsilon
        return [Box(low_in, high_in)]



class MoveSlowly:
    def __init__(self, max_velocity, epsilon):
        self.max_velocity = max_velocity
        self.epsilon = epsilon

    def constraint(self, zs, ins, targets, net, rollout_func):
        z_ins = zs[0]
        weights = net(z_ins)
        rollout = rollout_func(weights)

        displacements = torch.zeros_like(rollout)
        displacements[:, 1:, :] = rollout[:, 1:, :] - rollout[:, :-1, :] # i.e., v_t = x_{t + 1} - x_t
        velocities = ltd.TermDynamic(torch.norm(displacements, dim=2))

        return ltd.Always(
            ltd.LT(velocities, ltd.TermStatic(self.max_velocity)),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        low_in = ins - self.epsilon
        high_in = ins + self.epsilon
        return [Box(low_in, high_in)]


class AvoidPoint:
    def __init__(self, point, min_dist, epsilon):
        assert point.dim() == 2 # N X D
        self.point = point
        self.min_dist = min_dist
        self.epsilon = epsilon

    def constraint(self, zs, ins, targets, net, rollout_func):
        z_ins = zs[0]
        weights = net(z_ins)
        rollout = rollout_func(weights)

        dist_to_point = ltd.TermDynamic(torch.norm(rollout - self.point[:, None, :], dim=2))
        return ltd.Always(
            ltd.GT(dist_to_point, ltd.TermStatic(self.min_dist)),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        low_in = ins - self.epsilon
        high_in = ins + self.epsilon
        return [Box(low_in, high_in)]
