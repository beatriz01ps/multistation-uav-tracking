"""Dois UAVs inicialmente separados que cruzam trajetorias. Avalia
association accuracy, ID switches, fragmentacao e RMSE via o simulador +
evaluation/metrics.py."""

import numpy as np

from config.models import AppConfig
from evaluation.metrics import run_and_evaluate
from simulation.simple_simulator import UavScenario, run_simulation
from simulation.trajectories import straight_line
from simulation.virtual_station import VirtualStation


def test_crossing_uavs_keep_correct_identity():
    rng = np.random.default_rng(7)

    uav1_trajectory = straight_line(np.array([-50.0, 0.0, 100.0, 10.0, 0.0, 0.0]), duration_s=10.0, dt=1.0)
    uav2_trajectory = straight_line(np.array([50.0, 2.0, 100.0, -10.0, 0.0, 0.0]), duration_s=10.0, dt=1.0)

    uavs = [
        UavScenario(name="UAV_1", trajectory=uav1_trajectory),
        UavScenario(name="UAV_2", trajectory=uav2_trajectory),
    ]
    stations = [
        VirtualStation("station_a", position_std=1.5, velocity_std=0.3, rng=rng),
        VirtualStation("station_b", position_std=1.5, velocity_std=0.3, rng=rng),
    ]

    simulation = run_simulation(uavs, stations, scenario_id="crossing_test")
    report = run_and_evaluate(simulation, AppConfig())

    assert report["num_real_uavs"] == 2
    assert report["association_accuracy"] >= 0.9
    assert report["id_switches"] == 0
    assert report["position_rmse_m"] < 10.0
    for uav_name, fragments in report["track_fragmentation"].items():
        assert fragments == 1, f"{uav_name} fragmentou em {fragments} global tracks"
