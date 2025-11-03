"""
Quick Demo: ESMSEU on Split-MNIST Continual Learning.
Run: python example.py
Expected: ~70-80% final avg acc, <10% forgetting.
"""

import torch.nn as nn
from esmseu_continual.continual_trainer import ESMSEUContinualTrainer, create_split_mnist_tasks
from esmseu_continual.regularizer import ESMSEURegularizer

# Model (extendable to any nn.Module)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(self.flatten(x))

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Setup
    model = SimpleMLP()
    tasks = create_split_mnist_tasks(n_tasks=5)
    reg = ESMSEURegularizer(lambda1=0.1, lambda2=0.01, m=2)

    # Train
    trainer = ESMSEUContinualTrainer(
        model=model,
        tasks=tasks,
        reg=reg,
        epochs=2,  # Quick; increase for full
        lr=0.001,
        batch_size=64,
    )
    metrics = trainer.train()

    print("Results:")
    print(f"Avg Forgetting: {metrics['avg_forgetting']:.4f}")
    print(f"Est. Reduction: {metrics['forgetting_reduction']}% (vs. baseline)")
    print("Final Task Accs:", {k: f"{v:.4f}" for k, v in metrics['final_metrics'].items()})
