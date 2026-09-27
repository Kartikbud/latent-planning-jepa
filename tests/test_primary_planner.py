"""CPU regression tests for split-primary planning; no simulator or checkpoint needed.

Run: python -m unittest discover -s tests -p test_primary_planner.py -v
"""

import ast
import contextlib
import io
import json
from collections import deque
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference.mpc_parallel import Planner


def definitions(source):
    return {
        node.name: node for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def notebook_definitions():
    notebook = json.loads((ROOT / "notebooks/run_primary.ipynb").read_text())
    nodes = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            line for line in cell["source"] if not line.startswith(("%", "!"))
        )
        compile(source, f"notebook cell {i}", "exec")
        nodes.update(definitions(source))
    return nodes


def make_planner(cls=Planner, **kwargs):
    options = dict(N=100, K=10, T=4, seed=42)
    options.update(kwargs)
    return cls("cpu", nn.Identity(), nn.Identity(), **options)


class QuadraticPlanner(Planner):
    """Each gripper schedule requires an opposite motion; record actual candidates."""

    def score_actions(self, zi, a, actions, zg, apply_bounds=True):
        if not hasattr(self, "batches"):
            self.batches = []
            self.cost_batches = []
        costs = (actions[:, 0, 0] - actions[:, 0, 6]).square()
        self.batches.append(actions.clone())
        self.cost_batches.append(costs.clone())
        return costs


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.zi = torch.zeros(1, 192)
        self.a = torch.empty(0, 35)
        self.goal = torch.zeros(192)

    def test_notebook_and_scripts_are_identical(self):
        notebook = notebook_definitions()
        for path, names in (
            ("src/inference/mpc_parallel.py", ("Planner",)),
            ("src/inference/run_primary.py", ("run_rollout", "held_gripper_schedule")),
        ):
            source = definitions((ROOT / path).read_text())
            for name in names:
                self.assertEqual(ast.dump(source[name]), ast.dump(notebook[name]))

    def test_smoothing_preserves_variance_and_crosses_block_boundaries(self):
        planner = make_planner(noise_correlation=0.7)
        noise = planner._sample_noise(5000).reshape(5000, 25, 7)
        self.assertAlmostEqual(float(noise.var()), 1.0, delta=0.04)
        # Primitive steps 4 and 5 straddle two different model blocks.
        correlation = np.corrcoef(noise[:, 4, 0], noise[:, 5, 0])[0, 1]
        self.assertAlmostEqual(correlation, 0.7, delta=0.04)
        self.assertFalse(np.array_equal(noise[:, 4], noise[:, 5]))
        independent = make_planner(noise_correlation=0)._sample_noise(5000).reshape(5000, 25, 7)
        self.assertAlmostEqual(np.corrcoef(independent[:, 4, 0], independent[:, 5, 0])[0, 1], 0, delta=0.05)

    def test_shift_discards_one_block_and_preserves_action_order(self):
        planner = make_planner()
        plan = np.arange(175, dtype=np.float32).reshape(5, 35)
        shifted = planner.shifted_plan(plan)
        np.testing.assert_array_equal(shifted[:4], plan[1:])
        np.testing.assert_array_equal(shifted[-1], plan[-1])
        shifted[0, 0] = -100
        self.assertEqual(plan[1, 0], 35)

    def test_schedule_searches_do_not_mix_motion_elites(self):
        planner = make_planner(QuadraticPlanner, N=101, T=5, fresh_fraction=0, momentum=0)
        schedules = np.stack([np.full((5, 5), -1.), np.full((5, 5), 1.)])
        result = planner.CEM(self.zi, self.a, self.goal, gripper_schedules=schedules)
        for batch in planner.batches[:-1]:
            self.assertEqual(len(batch), 101)  # N remains the total, even when not divisible.
            self.assertTrue(torch.all(batch[:51, :, planner.gripper_indices] == -1))
            self.assertTrue(torch.all(batch[51:, :, planner.gripper_indices] == 1))
        final = planner.batches[-1]
        self.assertLess(final[0, 0, 0].item(), -0.8)
        self.assertGreater(final[1, 0, 0].item(), 0.8)
        result_cost = float((result[0, 0] - result[0, 6]) ** 2)
        self.assertLessEqual(result_cost, min(float(c.min()) for c in planner.cost_batches) + 1e-7)

    def test_best_sample_survives_a_bad_average(self):
        class TwoWells(Planner):
            def _sample_noise(self, count):
                noise = np.zeros((count, self.H, self.action_dim), np.float32)
                noise[1, :, 0], noise[2, :, 0] = -1, 1
                return noise

            def score_actions(self, zi, a, actions, zg, apply_bounds=True):
                return (actions[:, 0, 0].square() - 1).square()

        planner = make_planner(TwoWells, N=3, K=2, T=2, fresh_fraction=0, momentum=0)
        result = planner.CEM(self.zi, self.a, self.goal)
        self.assertEqual(abs(result[0, 0]), 1)

    def test_warm_mean_and_fresh_recovery_proposals_are_both_scored(self):
        planner = make_planner(QuadraticPlanner, N=20, T=1, fresh_fraction=0.2)
        warm = np.full((5, 35), 50., np.float32)
        planner.CEM(self.zi, self.a, self.goal, initial_mean=warm)
        samples = planner.batches[0].numpy()
        np.testing.assert_array_equal(samples[0], warm)
        self.assertTrue(np.all(np.abs(samples[-4:]) < 10))
        self.assertTrue(np.all(samples[1:-4] > 40))

    def test_bounds_closed_gripper_and_repeatability(self):
        options = dict(N=30, T=2, action_low=np.full(35, -0.5), action_high=np.full(35, 0.5))
        schedules = np.full((1, 5, 5), 1., np.float32)
        for separate in (True, False):
            p1 = make_planner(QuadraticPlanner, **options, separate_gripper_searches=separate)
            p2 = make_planner(QuadraticPlanner, **options, separate_gripper_searches=separate)
            first = p1.CEM(self.zi, self.a, self.goal, gripper_schedules=schedules)
            second = p2.CEM(self.zi, self.a, self.goal, gripper_schedules=schedules)
            np.testing.assert_array_equal(first, second)
            raw = first.reshape(5, 5, 7)
            self.assertTrue(np.all(np.abs(raw[..., :6]) <= 0.5))
            self.assertTrue(np.all(raw[..., -1] == 1))

    def test_actual_split_networks(self):
        from architectures.robot_encoder import RobotEncoder
        from architectures.predictor import PredictorNetwork

        encoder = RobotEncoder().eval()
        predictor = PredictorNetwork(d_action=35, window=3).eval()
        with torch.no_grad():
            z = encoder(torch.rand(1, 3, 3, 84, 84), torch.rand(1, 3, 3, 84, 84), torch.zeros(1, 3, 9))[0]
        planner = Planner("cpu", encoder, predictor, N=20, K=4, T=1, seed=42,
                          raw_action_mean=np.linspace(-.2, .2, 7), raw_action_std=np.linspace(.3, .9, 7))
        closed = (np.ones((1, 5, 5)) - planner.raw_action_mean[-1]) / planner.raw_action_std[-1]
        for context in (1, 3):
            for schedules in (planner.single_switch_gripper_schedules(), closed):
                plan = planner.CEM(z[:context], torch.zeros(context - 1, 35), z[-1],
                                   initial_mean=np.zeros((5, 35)), gripper_schedules=schedules)
                raw = planner.denormalize_action_blocks(plan)
                self.assertEqual(raw.shape, (5, 5, 7))
                self.assertTrue(np.isfinite(raw).all())
                if len(schedules) == 1:
                    np.testing.assert_allclose(raw[..., -1], 1., atol=1e-6)


class RolloutTests(unittest.TestCase):
    def run_mock(self, success_at=None, done_at=None, warm_start=True):
        class Env:
            def __init__(self): self.actions = []
            def step(self, action):
                self.actions.append(action.copy())
                return len(self.actions), 0, len(self.actions) == done_at, {}
            def _check_success(self):
                return success_at is not None and len(self.actions) >= success_at

        env = Env()
        outer = self

        class RecordingPlanner(Planner):
            def __init__(self):
                super().__init__("cpu", nn.Identity(), nn.Identity())
                self.calls = []
                self.previous = None

            def CEM(self, zi, a, zg, initial_mean=None, gripper_schedules=None):
                self._prepare_inputs(zi, a, zg)
                t = len(env.actions)
                outer.assertEqual(zi[-1, 0].item(), t)
                if len(a):
                    expected = np.asarray(env.actions[-len(a) * 5:]).reshape(len(a), 35)
                    np.testing.assert_allclose(a.numpy(), expected)
                if not warm_start or t in (0, 200):
                    outer.assertIsNone(initial_mean)
                else:
                    np.testing.assert_array_equal(initial_mean, self.shifted_plan(self.previous))
                self.calls.append(t)
                raw = np.zeros((5, 5, 7), np.float32)
                raw[:, :, 0] = np.arange(1, 26).reshape(5, 5) / 100
                raw[:, :, -1] = -1 if t < 200 else 1
                if t >= 200:
                    outer.assertTrue(np.all(gripper_schedules == 1))
                self.previous = raw.reshape(5, 35)
                return self.previous.copy()

        planner = RecordingPlanner()
        frames = []
        class Writer:
            def append_data(self, frame): frames.append(frame)
        namespace = dict(torch=torch, np=np, deque=deque, FRAME_SKIP=5, WINDOW=3,
                         PREDICTOR_ACTION_DIM=35, reset_to_eval_state=lambda env, entry: 0,
                         render_frame=lambda env: 0, set_geom_groups=lambda env: None,
                         encode_observation=lambda enc, obs, device, stats: torch.full((192,), float(obs)))
        source = definitions((ROOT / "src/inference/run_primary.py").read_text())
        for name in ("postprocess_action", "held_gripper_schedule", "run_rollout"):
            exec(compile(ast.Module(body=[source[name]], type_ignores=[]), name, "exec"), namespace)
        entry = dict(name="demo_0", start=torch.zeros(192), stages=[
            dict(name=name, goal=torch.ones(192), frame=i) for i, name in enumerate(("grasp", "lift"))
        ])
        settings = {name: dict(budget=budget, threshold=0., use_threshold=False)
                    for name, budget in (("grasp", 200), ("lift", 100))}
        with contextlib.redirect_stdout(io.StringIO()):
            result = namespace["run_rollout"](env, planner, None, entry, "cpu", {}, Writer(), settings,
                                               warm_start=warm_start)
        self.assertEqual(len(frames), len(env.actions) + 1)
        return result, planner.calls, np.asarray(env.actions)

    def test_full_rollout_warm_resets_history_and_order(self):
        for enabled in (True, False):
            result, calls, actions = self.run_mock(warm_start=enabled)
            self.assertEqual(result, (False, 300))
            self.assertEqual(calls, list(range(0, 300, 5)))
            np.testing.assert_allclose(actions[:, 0], np.tile(np.arange(1, 6) / 100, 60))
            self.assertTrue(np.all(actions[:200, -1] == -1))
            self.assertTrue(np.all(actions[200:, -1] == 1))

    def test_early_success_and_termination(self):
        for step in (3, 203):
            self.assertEqual(self.run_mock(success_at=step)[0], (True, step))
        self.assertEqual(self.run_mock(done_at=7)[0], (False, 7))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
