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


class _CartesianTensorLoss(LossDefinition):
    """
    Loss for a 3x3 Cartesian tensor target, per atom or per graph.

    Both field-response predictions are already Cartesian -- Z* comes out of
    ``d(dipole_a)/dr_ib`` and chi out of ``d(dipole_a)/dE_b`` -- so unlike
    SevenNet-Polar, which emits BEC in irreps form and has to round-trip the
    reference through ``e3nn.io.CartesianTensor``, the reference and prediction
    are directly comparable. The base ``_preprocess`` flattening is all we need.

    ``get_loss`` is scaled by 9 because flattening divides the mean by 9*N
    instead of N, which would otherwise shrink these gradients ninefold
    relative to the energy/force terms.
    """

    n_component = 9

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            weight = batch_data[KEY.DATA_WEIGHT][self.name.lower()]
            if self.is_per_atom:
                weight = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(weight, self.n_component)

        return pred, ref, w_tensor

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        loss = super().get_loss(batch_data, model)
        # The x9 undoes the componentwise mean that flattening introduces, so
        # the weight refers to a whole tensor rather than one component. L2MAE
        # already reduces per tensor (it takes the norm over the 9 components),
        # so applying it there would count the same factor twice.
        from sevenn.train.optim import L2MAE
        if isinstance(self.criterion, L2MAE):
            return loss
        return loss * float(self.n_component)


class BECLoss(_CartesianTensorLoss):
    """
    Loss for Born effective charges, Z*_{i,ab} = d(dipole_a)/dr_{ib}, per atom.
    """

    is_per_atom = True

    def __init__(
        self,
        name: str = 'BornEffectiveCharges',
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


class PolarizabilityLoss(_CartesianTensorLoss):
    """
    Loss for the electronic polarizability chi = eps_inf - 1, per graph.

    The model's chi is symmetric by construction (Maxwell reciprocity), while
    the MP-Dielectrics labels are only symmetric to ~7e-2. That antisymmetric
    part is unfittable by design; it sets a floor on the achievable RMSE rather
    than a bias, since a symmetric prediction is the least-squares optimum for
    an almost-symmetric target.
    """

    is_per_atom = False

    def __init__(
        self,
        name: str = 'Polarizability',
        unit: str = '',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.POLARIZABILITY,
        pred_key: str = KEY.PRED_POLARIZABILITY,
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
    for key in ['energy', 'force', 'stress', 'bec', 'polarizability']:
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
        'polarizability': PolarizabilityLoss,
    }
    loss_weights = {
        'energy': config.get(KEY.ENERGY_WEIGHT, 1.0),
        'force': config[KEY.FORCE_WEIGHT],
        'stress': config[KEY.STRESS_WEIGHT],
        'bec': config.get(KEY.BEC_WEIGHT, 1.0),
        'polarizability': config.get(KEY.POLARIZABILITY_WEIGHT, 0.15),
    }

    use_weight = config.get(KEY.USE_WEIGHT, False)
    commons = {'use_weight': use_weight}

    keys = ['energy', 'force']
    if config[KEY.IS_TRAIN_STRESS]:
        keys += ['stress']
    if config.get(KEY.IS_TRAIN_BEC, False):
        keys += ['bec']
    if config.get(KEY.IS_TRAIN_POLARIZABILITY, False):
        keys += ['polarizability']

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
