import unittest
from bochemian.bo.optimizer import (
    BotorchOptimizer,
)  # replace with your actual module name
import torch
from bochemian.utils import instantiate_class


class TestBotorchOptimizer(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.optimizer = BotorchOptimizer(
            design_space=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            surrogate_model_config=BotorchOptimizer.default_surrogate_model_config(),
            acq_function_config=BotorchOptimizer.default_acq_function_config(),
            train_x=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            train_y=torch.tensor([[0.5], [0.6]]),
            heldout_x=torch.tensor([[0.7, 0.8], [0.9, 1.0], [0.1, 0.3]]),
        )
        print("YOOOO", self.optimizer.heldout_x.shape)
        print(torch.tensor([[0.7, 0.8], [0.9, 1.0], [0.1, 0.2]]).shape)

    def test_init(self):
        # Check that the optimizer was initialized correctly

        surrogate_model = instantiate_class(
            self.optimizer.surrogate_model_config,
            train_x=self.optimizer.train_x,
            train_y=self.optimizer.train_y,
        )
        acq_function = instantiate_class(
            self.optimizer.acq_function_config, model=surrogate_model
        )

        self.assertIsNotNone(surrogate_model)
        self.assertIsNotNone(acq_function)

        # Check the initial heldout_y tensor
        self.assertEqual(self.optimizer.heldout_y.shape, (3, 1))
        self.assertTrue(torch.isnan(self.optimizer.heldout_y).all())

    def test_lie_to_me(self):
        # Test different strategies
        for strategy in ["cl_min", "cl_mean", "cl_max", "kriging"]:
            y_lie = self.optimizer.lie_to_me(
                torch.tensor([[0.1, 0.2]]), self.optimizer.train_y, strategy=strategy
            )
            self.assertIsNotNone(y_lie)

        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.optimizer.lie_to_me(
                torch.tensor([[0.1, 0.2]]), self.optimizer.train_y, strategy="invalid"
            )

    def test_ask(self):
        # Test asking for a single point
        candidates = self.optimizer.ask(n_points=1)
        self.assertEqual(len(candidates), 1)

        # Test asking for multiple points
        candidates = self.optimizer.ask(n_points=2)
        self.assertEqual(len(candidates), 2)

        # Test that it correctly handles surrogate_model being None
        self.optimizer.surrogate_model = None
        self.optimizer.train_x = None
        self.optimizer.train_y = None
        # TODO gotta check this one
        # candidates = self.optimizer.ask(n_points=1)
        # self.assertEqual(candidates, [])

    def test_tell(self):
        # Test telling with lie=False
        self.assertEqual(self.optimizer.heldout_x.shape[0], 3)

        self.optimizer.tell(
            torch.tensor([[0.1, 0.3]]), torch.tensor([[0.5]]), lie=False
        )

        # Check that the point was removed from heldout_x and heldout_y
        self.assertEqual(self.optimizer.heldout_x.shape[0], 2)
        self.assertEqual(self.optimizer.heldout_y.shape[0], 2)

        # Test telling with lie=True
        self.optimizer.tell(torch.tensor([[0.7, 0.8]]), torch.tensor([[0.5]]), lie=True)
        # The point should not be removed from heldout_x and heldout_y
        self.assertEqual(self.optimizer.heldout_x.shape[0], 2)
        self.assertEqual(self.optimizer.heldout_y.shape[0], 2)

    def test_next_evaluations(self):
        # Test with heldout_x set
        candidates, acq_vals = self.optimizer.next_evaluations(
            self.optimizer.acq_function
        )
        self.assertIsNotNone(candidates)
        self.assertIsNotNone(acq_vals)

        # Test without heldout_x set
        self.optimizer.heldout_x = None
        candidates, acq_vals = self.optimizer.next_evaluations(
            self.optimizer.acq_function, bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        )
        self.assertIsNotNone(candidates)
        self.assertIsNotNone(acq_vals)

    def test_default_surrogate_model_config(self):
        # Check that it returns a valid config
        config = BotorchOptimizer.default_surrogate_model_config()
        self.assertIsInstance(config, dict)

    def test_default_acq_function_config(self):
        # Check that it returns a valid config
        config = BotorchOptimizer.default_acq_function_config()
        self.assertIsInstance(config, dict)


if __name__ == "__main__":
    unittest.main()
