from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(model, optimizer_cls, optimizer_kwargs, data_loader, dp_sigma):
    # 1. 替换不支持的模块（如 BatchNorm）
    model = ModuleValidator.fix(model)

    # 2. 重新初始化优化器
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # 3. 接入 DP
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_sigma,
        max_grad_norm=MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA