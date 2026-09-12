import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd.functional import jvp

class SimpleNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z, r, t):
        net_input = torch.cat([z, t, t-r], dim=-1)
        return self.net(net_input)

def sample_gaussian_2d(n_samples=1000, mean=[0, 0], std=1.0, device='cpu'):
    return torch.normal(mean[0], std, (n_samples, 2), device=device)

def sample_spiral_2d(n_samples=1000, noise=0.1, device='cpu'):
    t = torch.linspace(0, 4*torch.pi, n_samples, device=device)
    r = t / (2*torch.pi)
    x = r * torch.cos(t) + torch.normal(0, noise, (n_samples,), device=device)
    y = r * torch.sin(t) + torch.normal(0, noise, (n_samples,), device=device)
    return torch.stack([x, y], dim=1)

def sample_t_r(batch_size=256, device='cpu'):
    if torch.rand(1).item() < 1.0:
        samples = torch.sigmoid(torch.normal(-0.4, 1.0, (batch_size, 2), device=device))
        # enforce t > r
        t = torch.max(samples[:, 0], samples[:, 1]).unsqueeze(1)
        r = torch.min(samples[:, 0], samples[:, 1]).unsqueeze(1)
    else:
        t = torch.sigmoid(torch.normal(-0.4, 1.0, (batch_size, 1), device=device))
        r = t.clone()
    
    return t, r

def create_training_data(model, n_samples=1000, device='cpu'):
    e = sample_gaussian_2d(n_samples, device=device)
    x = sample_spiral_2d(n_samples, device=device)

    t, r = sample_t_r(n_samples, device=device)

    z = (1 - t) * x + t * e
    v = e - x

    u, dudt = jvp(
        func=model,
        inputs=(z, r, t),
        v=(v, torch.zeros_like(r), torch.ones_like(t)),
        create_graph=True
    )

    u_tgt = v - (t - r) * dudt
    u_tgt = u_tgt.detach()
    
    return z, r, t, u_tgt

def train_mean_flow(model, n_epochs=1000, batch_size=256, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    tic = time.time()
    for epoch in range(n_epochs):
        z, r, t, u_tgt = create_training_data(model, batch_size, device)
        
        optimizer.zero_grad()
        
        predicted_velocity = model(z, r, t)
        loss = criterion(predicted_velocity, u_tgt)

        adp_wt = (loss + 0.01) ** 1.0
        loss = loss / adp_wt.detach()
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, time: {time.time() - tic:.2f}s")
            tic = time.time()
    
    return losses

def sample_mean_flow(model, noise, n_steps=100):
    device = next(model.parameters()).device
    with torch.no_grad():
        x = noise.to(device)
        dt = 1.0 / n_steps
        
        trajectory = [x.clone().cpu()]
        
        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i-1) * dt, device=device).unsqueeze(1)
            t = torch.full((x.shape[0],), i * dt, device=device).unsqueeze(1)
            velocity = model(x, r, t)
            x = x - velocity * dt
            trajectory.append(x.clone().cpu())
    
    return torch.stack(trajectory)

def setup_subplot(ax, title, x_lim=(-2.5, 2.5), y_lim=(-2.5, 2.5)):
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def plot_source_data(axes, x0, x1_target, x_lim, y_lim):
    axes[0, 0].scatter(x0[:, 0].cpu(), x0[:, 1].cpu(), alpha=0.6, s=20)
    setup_subplot(axes[0, 0], 'Source: 2D Gaussian', x_lim, y_lim)
    
    axes[0, 1].scatter(x1_target[:, 0].cpu(), x1_target[:, 1].cpu(), alpha=0.6, s=20, color='red')
    setup_subplot(axes[0, 1], 'Target: Spiral', x_lim, y_lim)

def plot_trajectory(ax, model, noise, x_lim, y_lim):
    trajectory = sample_mean_flow(model, noise, n_steps=64)
    for i in range(0, len(trajectory), max(1, len(trajectory)//10)):
        traj_data = trajectory[i].numpy()
        ax.scatter(traj_data[:100, 0], traj_data[:100, 1], 
                  alpha=0.4, s=5, c=plt.cm.viridis(i/len(trajectory)))
    setup_subplot(ax, 'Transformation Trajectory', x_lim, y_lim)

def plot_step_comparisons(axes, model, noise, step_counts, x_lim, y_lim):
    for i, n_steps in enumerate(step_counts):
        trajectory = sample_mean_flow(model, noise, n_steps=n_steps)
        x1_generated = trajectory[-1].numpy()
        
        axes[1, i].scatter(x1_generated[:, 0], x1_generated[:, 1], alpha=0.6, s=20)
        setup_subplot(axes[1, i], f'{n_steps} steps', x_lim, y_lim)

def visualize_transformation_with_steps(model, n_samples=1000):
    device = next(model.parameters()).device
    noise = sample_gaussian_2d(n_samples, device=device)
    target = sample_spiral_2d(n_samples, device=device)
    
    step_counts = [1, 4, 16, 64]
    x_lim = (-2.5, 2.5)
    y_lim = (-2.5, 2.5)
    
    _, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    plot_source_data(axes, noise, target, x_lim, y_lim)
    plot_trajectory(axes[0, 2], model, noise, x_lim, y_lim)
    axes[0, 3].axis('off')
    
    plot_step_comparisons(axes, model, noise, step_counts, x_lim, y_lim)
    
    plt.tight_layout()
    plt.savefig('figures/mean_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("Training Mean Flow: 2D Gaussian → Spiral")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleNet()
    
    print("Training model...")
    losses = train_mean_flow(model, n_epochs=10000, batch_size=512, device=device)
    
    print("Visualizing transformation with different step counts...")
    visualize_transformation_with_steps(model)
    
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('figures/mean_flow_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Done! Check the generated plots.")

if __name__ == "__main__":
    main()