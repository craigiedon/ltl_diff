from . import ltldiff as ltd
import torch
from .domains import Box

def constraint_loss(constraint, ins, targets, zs, net, rollout_func):
    cond = constraint.condition(zs, ins, targets, net, rollout_func)
    
    neg_losses = ltd.Negate(cond).loss(0)
    losses = cond.loss(0)
    sat = cond.satisfy(0)
        
    return neg_losses, losses, sat

def fully_global_ins(ins, epsilon):
    low_ins = ins - epsilon
    high_ins = ins + epsilon
    return [Box(low_ins[:, i], high_ins[:, i]) for i in range(ins.shape[1])]


class EventuallyReach:
    def __init__(self, reach_ids, epsilon):
        # The index of object in start info that you need to eventually reach
        self.reach_ids = reach_ids 
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        rollout_term = ltd.TermDynamic(rollout_traj)

        reach_constraints = []
        for reach_id in self.reach_ids:
            reach_point_batch = zs[:, reach_id]
            reach_constraints.append(ltd.Eventually(
                ltd.EQ(rollout_term, ltd.TermStatic(reach_point_batch)),
                rollout_traj.shape[1]
            ))
        
        return ltd.And(reach_constraints)

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

class StayInZone:
    def __init__(self, min_bound, max_bound, epsilon):
        assert min_bound.dim() == 1, "Num of dims min bound should be 1: (Spatial Dims)"
        assert max_bound.dim() == 1, "Num of dims for max bound should be 1: (Spatial Dims)"

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        return ltd.Always(
            ltd.And([
                ltd.GEQ(rollout_term, ltd.TermStatic(self.min_bound)),
                ltd.LEQ(rollout_term, ltd.TermStatic(self.max_bound))
            ]),
            rollout_traj.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class LipchitzContinuous:
    def __init__(self, smooth_thresh, epsilon):
        self.smooth_thresh = smooth_thresh
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights_x = net(ins)
        weights_z = net(zs)

        rollout_x = rollout_func(ins[:, 0], ins[:, -1], weights_x)[0]
        rollout_z = rollout_func(zs[:, 0], zs[:, -1], weights_z)[0]

        rollout_diffs = ltd.TermDynamic(torch.abs(rollout_x - rollout_z))
        input_diffs = ltd.TermStatic(self.smooth_thresh * torch.abs(ins - zs))

        return ltd.Always(
            ltd.LEQ(rollout_diffs, input_diffs),
            rollout_x.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class DontTipEarly:
    def __init__(self, orientation_lb, orientation_ub, tip_id, min_dist, epsilon):
        self.tip_id = tip_id
        self.min_dist = min_dist
        self.orientation_lb = orientation_lb
        self.orientation_ub = orientation_ub
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        # Assuming xyz-rpy
        rotation_terms = ltd.TermDynamic(rollout[:, :, 3:])

        # Measuring the distance across dimensions...
        tip_point = zs[:, self.tip_id, 0:2]

        dist_to_tip = ltd.TermDynamic(torch.norm(rollout[:, :, 0:2] - tip_point[:, None, :], dim=2, keepdim=True))

        return ltd.Always(
            ltd.Implication(
                ltd.Or([
                    ltd.GEQ(dist_to_tip, ltd.TermStatic(self.min_dist)),
                    ltd.LEQ(ltd.TermDynamic(rollout[:, :, 2:3]), ltd.TermStatic(zs[:, 1, 2:3]))
                ]),
                ltd.And([
                    ltd.GEQ(rotation_terms, ltd.TermStatic(self.orientation_lb)),
                    ltd.LEQ(rotation_terms, ltd.TermStatic(self.orientation_ub))
                ])
            ),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)



class MoveSlowly:
    def __init__(self, max_velocity, epsilon):
        self.max_velocity = max_velocity
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        displacements = torch.zeros_like(rollout)
        displacements[:, 1:, :] = rollout[:, 1:, :] - rollout[:, :-1, :] # i.e., v_t = x_{t + 1} - x_t
        velocities = ltd.TermDynamic(torch.norm(displacements, dim=2, keepdim=True))

        return ltd.Always(
            ltd.LT(velocities, ltd.TermStatic(self.max_velocity)),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class AvoidPoint:
    def __init__(self, point_id, min_dist, epsilon):
        self.point_id = point_id
        self.min_dist = min_dist
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        point = zs[:, self.point_id]

        dist_to_point = ltd.TermDynamic(
            torch.norm(rollout - point[:, None, :], dim=2, keepdim=True)
        )

        return ltd.Always(
            ltd.GT(dist_to_point, ltd.TermStatic(self.min_dist)),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

class AvoidAndPatrolConstant:
    def __init__(self, av_point, patrol_point, min_avoid_dist, epsilon):
        self.av_point = av_point
        self.patrol_point = patrol_point
        self.epsilon = epsilon
        self.min_avoid_dist = min_avoid_dist

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]


        dist_avoid = ltd.TermDynamic(
            torch.norm(rollout[:, :, 0:2] - self.av_point, dim=2, keepdim=True)
        )

        return ltd.And([
            ltd.Always(
                ltd.GT(dist_avoid, ltd.TermStatic(self.min_avoid_dist)),
                rollout.shape[1]
            ),
            ltd.Eventually(
                ltd.EQ(ltd.TermDynamic(rollout[:, :, 0:2]), ltd.TermStatic(self.patrol_point)),
                rollout.shape[1]
            )
        ])

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)
