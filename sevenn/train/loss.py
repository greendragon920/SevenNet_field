import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import sevenn._const as CONST
import sevenn._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name: str,
        unit: Optional[str] = None,
        criterion: Optional[Callable] = None,
        ref_key: Optional[str] = None,
        pred_key: Optional[str] = None,
        use_weight: bool = False,
        ignore_unlabeled: bool = True,
    ) -> None:
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.use_weight = use_weight
        self.ignore_unlabeled = ignore_unlabeled

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion: Callable) -> None:
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        return pred, ref, None

    def _ignore_unlabeled(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        data_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        unlabeled = torch.isnan(ref)
        pred = pred[~unlabeled]
        ref = ref[~unlabeled]
        if data_weights is not None:
            data_weights = data_weights[~unlabeled]
        return pred, ref, data_weights

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        """
        Function that return scalar
        """
        if self.criterion is None:
            raise NotImplementedError('LossDefinition has no criterion.')
        pred, ref, w_tensor = self._preprocess(batch_data, model)

        if self.ignore_unlabeled:
            pred, ref, w_tensor = self._ignore_unlabeled(pred, ref, w_tensor)

        if len(pred) == 0:
            assert self.ref_key is not None
            return torch.zeros(1, device=batch_data[self.ref_key].device)

        ref = ref.to(dtype=pred.dtype)
        if w_tensor is not None:
            w_tensor = w_tensor.to(dtype=pred.dtype)

        loss = self.criterion(pred, ref)
        if self.use_weight:
            loss = torch.mean(loss * w_tensor)
        return loss


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name: str = 'Energy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.ENERGY,
        pred_key: str = KEY.PRED_TOTAL_ENERGY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name: str = 'Force',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(w_tensor, 3)

        return pred, ref, w_tensor


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name: str = 'Stress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        pred = torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,))
        ref = torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 6)

        return pred, ref, w_tensor


class BECLoss(LossDefinition):
    """
    Loss for Born effective charges, Z*_{i,ab} = d(dipole_a)/dr_{ib}, per atom
    """

    def __init__(
        self,
        name: str = 'BEC',
        unit: str = 'e',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.BEC,
        pred_key: str = KEY.PRED_BEC,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(w_tensor, 9)

        return pred, ref, w_tensor


class SusceptibilityLoss(LossDefinition):
    """
    Loss for the electronic susceptibility chi = eps_inf - 1, per graph
    """

    def __init__(
        self,
        name: str = 'Susceptibility',
        unit: str = '',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.SUSCEPTIBILITY,
        pred_key: str = KEY.PRED_SUSCEPTIBILITY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 9)

        return pred, ref, w_tensor


def _closest_lattice_shift(c: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Integer n minimizing ``||(c - n) Q||``, one row at a time.

    Rounding c is only the nearest lattice point for a well conditioned cell,
    so the basis is triangularised with QR and searched depth first, pruning
    any branch already longer than the best found. The result is exact for any
    full rank cell. The shift is an integer and carries no gradient, so the
    search runs detached and the caller applies it with tensor ops.
    """
    c_cpu = c.detach().double().cpu()
    Q_cpu = Q.detach().double().cpu()
    shifts = torch.zeros_like(c_cpu)

    for i in range(c_cpu.shape[0]):
        if not torch.isfinite(c_cpu[i]).all():
            continue  # unlabeled row, the caller drops it anyway
        r = torch.linalg.qr(Q_cpu[i].transpose(0, 1).contiguous())
        y = r.Q.transpose(0, 1) @ (c_cpu[i] @ Q_cpu[i])
        rr = r.R
        sign = torch.where(torch.diagonal(rr) < 0, -1.0, 1.0)
        rr, y = sign.unsqueeze(-1) * rr, sign * y

        cur = [0, 0, 0]
        for level in range(2, -1, -1):
            off = sum(float(rr[level, j]) * cur[j] for j in range(level + 1, 3))
            cur[level] = round((float(y[level]) - off) / float(rr[level, level]))
        best = list(cur)
        resid = rr @ torch.tensor(best, dtype=rr.dtype) - y
        best_sq = float(resid @ resid)

        def search(level, dist_sq, cur=cur):
            nonlocal best, best_sq
            if level < 0:
                best, best_sq = list(cur), dist_sq
                return
            diag = float(rr[level, level])
            off = sum(float(rr[level, j]) * cur[j] for j in range(level + 1, 3))
            center = (float(y[level]) - off) / diag
            radius = math.sqrt(max(best_sq - dist_sq, 0.0)) / abs(diag)
            lo, hi = math.ceil(center - radius), math.floor(center + radius)
            for z in sorted(range(lo, hi + 1), key=lambda v: abs(v - center)):
                step = diag * z + off - float(y[level])
                nxt = dist_sq + step * step
                if nxt <= best_sq:
                    cur[level] = z
                    search(level - 1, nxt)
            cur[level] = 0

        search(2, 0.0)
        shifts[i] = torch.tensor(best, dtype=c_cpu.dtype)

    return shifts.to(device=c.device, dtype=c.dtype)


def fold_polarization_difference(
    pred: torch.Tensor, ref: torch.Tensor, cell: torch.Tensor
) -> torch.Tensor:
    """Return ``pred - ref`` reduced onto the shortest branch of the
    polarization lattice ``Q = cell / |Omega|``, shape (n, 3).

    Rows whose reference is NaN come back NaN.
    """
    pred = pred.reshape(-1, 3)
    ref = ref.reshape(-1, 3)
    cell = cell.reshape(-1, 3, 3)

    vol = torch.det(cell).abs().clamp_min(1e-8)
    Q = cell / vol.view(-1, 1, 1)  # rows are the lattice vectors over |Omega|

    d = pred - ref
    c = torch.linalg.solve(Q.transpose(1, 2), d.unsqueeze(-1)).squeeze(-1)
    shift = _closest_lattice_shift(c, Q)
    return d - torch.einsum('ni,nij->nj', shift, Q)


class PolarizationLoss(LossDefinition):
    """
    Loss for the polarization P = -dF/dE / volume, per graph, in e/Angstrom^2

    The difference is folded onto the polarization lattice before the criterion
    sees it; the reported metric folds too.
    """

    def __init__(
        self,
        name: str = 'Polarization',
        unit: str = 'e/Ang^2',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.POLARIZATION,
        pred_key: str = KEY.PRED_POLARIZATION,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        ref = batch_data[self.ref_key].reshape(-1, 3)
        folded = fold_polarization_difference(
            batch_data[self.pred_key], ref, batch_data[KEY.CELL]
        )
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 3)

        # the criterion gets a pair whose difference is already folded
        return (ref + folded).reshape(-1), ref.reshape(-1), w_tensor


class L2Regularization(LossDefinition):
    """
    L2 regularization for task-specific (modal) parameters.
    Regularizes the last weight view of modal-specific IrrepsLinear layers,
    which corresponds to the modal input dimension.
    """

    def __init__(
        self,
        name: str,
        module_keys: List[str],
        reg_modal_only: bool = True,
    ):
        super().__init__(
            name=name,
            unit=None,
            criterion=None,
            ref_key=None,
            pred_key=None,
        )
        self.module_keys = module_keys
        self.reg_modal_only = reg_modal_only

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        device = batch_data['x'].device
        ret = torch.tensor([0.0], device=device)
        for module_key in self.module_keys:
            module = model._modules[module_key]  # type: ignore
            reg_params = list(module._modules['linear'].weight_views())[-1]
            reg_loss = torch.sum(torch.pow(reg_params, 2))
            ret = ret + reg_loss
        return 0.5 * ret  # penalty = (1/2)||w||^2

    def get_cosine(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ):
        cosine_list = []
        for module_key in self.module_keys:
            module = model._modules[module_key]  # type: ignore
            reg_params = list(module._modules['linear'].weight_views())[-1]
            dot = torch.dot(reg_params[0], reg_params[1])
            norm = torch.norm(reg_params[0]) * torch.norm(reg_params[1])
            cosine_list.append(dot / norm)
        ret = torch.tensor(
            [sum(cosine_list) / len(cosine_list)],
            device=batch_data['x'].device,
        )
        return ret


def get_modal_regularization(
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
) -> Optional[Tuple[LossDefinition, float]]:
    reg_params = config.get(KEY.REG_PARAM, {})

    modal_param = reg_params.get('modal', {})
    if not modal_param or not config.get(KEY.USE_MODALITY, False):
        return None

    if not model:
        raise ValueError('modal reg is requested but model is not given.')

    module_keys_to_reg = []
    for module_key in list(model._modules.keys()):
        for (
            use_modal_module_key,
            modal_module_name,
        ) in CONST.IMPLEMENTED_MODAL_MODULE_DICT.items():
            if (
                not config[use_modal_module_key]
                or modal_module_name not in module_key
            ):
                continue
            elif modal_module_name == 'reduce_input_to_hidden':
                continue
            module_keys_to_reg.append(module_key)

    return (
        L2Regularization('L2_modal', module_keys_to_reg, reg_modal_only=True),
        float(modal_param.get(KEY.REG_WEIGHT, 1e-5)),
    )


def make_loss_info_dict_from_config(config: Dict[str, Any]):
    # this is for backward compatibility
    loss_info_dict = {}
    loss_type = config.get(KEY.LOSS, 'mse').lower()
    loss_param = config.get(KEY.LOSS_PARAM, {})
    for key in ['energy', 'force', 'stress', 'bec', 'susceptibility', 'polarization']:
        loss_info_dict[key] = {}
        # loss_weight not initialized here.
        loss_info_dict[key].update(
            {KEY.LOSS_TYPE: loss_type, KEY.LOSS_PARAM: loss_param}
        )

    return loss_info_dict


def get_loss_functions_from_config(
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
) -> List[Tuple[LossDefinition, float]]:
    from sevenn.train.optim import loss_dict
    from sevenn.train.reewc.loss import get_ewc_loss

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss_info_dict = config.get(KEY.LOSS, 'mse')
    if isinstance(loss_info_dict, str):
        loss_info_dict = make_loss_info_dict_from_config(config)

    loss_function_cls_dict = {
        'energy': PerAtomEnergyLoss,
        'force': ForceLoss,
        'stress': StressLoss,
        'bec': BECLoss,
        'susceptibility': SusceptibilityLoss,
        'polarization': PolarizationLoss,
    }
    loss_weights = {
        'energy': config.get(KEY.ENERGY_WEIGHT, 1.0),
        'force': config[KEY.FORCE_WEIGHT],
        'stress': config[KEY.STRESS_WEIGHT],
        'bec': config[KEY.BEC_WEIGHT],
        'susceptibility': config[KEY.SUSCEPTIBILITY_WEIGHT],
        'polarization': config[KEY.POLARIZATION_WEIGHT],
    }

    use_weight = config.get(KEY.USE_WEIGHT, False)
    commons = {'use_weight': use_weight}

    keys = ['energy', 'force']
    if config[KEY.IS_TRAIN_STRESS]:
        keys += ['stress']
    if config[KEY.IS_TRAIN_BEC]:
        keys += ['bec']
    if config[KEY.IS_TRAIN_SUSCEPTIBILITY]:
        keys += ['susceptibility']
    if config[KEY.IS_TRAIN_POLARIZATION]:
        keys += ['polarization']

    for key in keys:
        loss_info = loss_info_dict.get(key, {})
        loss_param = loss_info.get(KEY.LOSS_PARAM, {})
        loss_weight = loss_info.get(KEY.LOSS_WEIGHT, loss_weights[key])
        if (loss_type := loss_info.get(KEY.LOSS_TYPE, 'mse').lower()) == 'l2mae':
            if key == 'energy':
                raise NotImplementedError('L2MAE not implemented for energy.')
            else:
                loss_param.update({'prop': key})

        loss_cls = loss_dict[loss_type]
        if use_weight:
            loss_param['reduction'] = 'none'
        criterion = loss_cls(**loss_param)
        loss_function_cls = loss_function_cls_dict[key]
        loss_function = loss_function_cls(criterion=criterion, **commons)
        loss_functions.append((loss_function, loss_weight))

    if addi_loss := get_ewc_loss(config):
        loss_functions.append(addi_loss)
    if addi_loss := get_modal_regularization(config, model):
        loss_functions.append(addi_loss)

    return loss_functions
