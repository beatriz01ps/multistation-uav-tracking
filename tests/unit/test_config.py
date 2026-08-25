from pathlib import Path

from config.loader import load_config
from config.models import AppConfig
from models.enums import AssociationMode, FusionStrategyName, MotionModelName


def test_default_config_matches_spec_example_values():
    config = AppConfig()
    assert config.tracking.tentative_confirmation_hits == 3
    # Desvio deliberado do exemplo da especificacao (3.0/6.0/10.0): esses
    # tres campos sao explicitamente parametros de configuracao a ajustar
    # experimentalmente, nao valores fixos. Recalibrados pra cobrir
    # oclusoes/dropouts mais longos sem perder a identidade do track.
    assert config.tracking.coasting_timeout_seconds == 10.0
    assert config.tracking.lost_timeout_seconds == 20.0
    assert config.tracking.deletion_timeout_seconds == 32.0
    assert config.association.mode is AssociationMode.FULL_STATE
    # Desvio deliberado do exemplo da especificacao (0.99): correcao tipo
    # Bonferroni pro aumento de frequencia das estacoes (2-10Hz -> 20-60Hz,
    # ver AssociationConfig.chi_square_probability).
    assert config.association.chi_square_probability == 0.9983
    # Desvio deliberado do exemplo da especificacao (que sugeria
    # constant_velocity): o tracker fica SEMPRE pronto pra extrapolar uma
    # manobra durante COASTING, ja que nao da pra saber de antemao se um
    # alvo vai curvar ou nao - ver FilterConfig.motion_model.
    assert config.filter.motion_model is MotionModelName.COORDINATED_TURN
    assert config.fusion.strategy is FusionStrategyName.INFORMATION


def test_load_config_from_repo_default_yaml():
    path = Path(__file__).resolve().parents[2] / "src" / "config" / "default.yaml"
    config = load_config(path)
    assert config == AppConfig()


def test_load_config_none_returns_defaults():
    assert load_config(None) == AppConfig()
