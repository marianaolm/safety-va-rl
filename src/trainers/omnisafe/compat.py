import inspect
import torch.optim.lr_scheduler as _lrs
from torch.optim.lr_scheduler import LinearLR as _LinearLR


def patch_linear_lr():
    if "verbose" not in inspect.signature(_LinearLR.__init__).parameters:

        class LinearLRCompat(_LinearLR):
            def __init__(
                self,
                optimizer,
                start_factor: float = 1.0 / 3,
                end_factor: float = 1.0,
                total_iters: int = 5,
                last_epoch: int = -1,
                verbose=None,
            ):
                super().__init__(
                    optimizer,
                    start_factor=start_factor,
                    end_factor=end_factor,
                    total_iters=total_iters,
                    last_epoch=last_epoch,
                )

        _lrs.LinearLR = LinearLRCompat
