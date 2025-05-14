import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from QuantConnect import *
from QuantConnect.Algorithm import *
import numpy as np

class StockLevelFactorLearning(nn.Module):
    """Module for learning stock-level irrationality factors using cointegration attention."""
    def __init__(self, num_stocks, embedding_dim, lambda1=0.1, lambda_rho=0.01):
        super(StockLevelFactorLearning, self).__init__()
        self.stock_embeddings = nn.Embedding(num_stocks, embedding_dim)
        self.beta = nn.Parameter(torch.randn(num_stocks, num_stocks))
        self.rho = nn.Parameter(torch.zeros(num_stocks))
        self.lambda1 = lambda1
        self.lambda_rho = lambda_rho

    def forward(self, prices):
        """Compute virtual rational prices and irrationality factors."""
        T, I = prices.shape
        beta = self.beta.clone()
        beta.diagonal().zero_()
        embeddings = self.stock_embeddings.weight
        w = torch.mm(embeddings, embeddings.T)
        w.diagonal().fill_(-float('inf'))
        exp_w = torch.exp(w)
        sum_exp_w = exp_w.sum(dim=1, keepdim=True)
        ATT = exp_w / sum_exp_w
        p_expanded = prices.unsqueeze(1)
        beta_expanded = beta.unsqueeze(0)
        candidate_prices = beta_expanded * p_expanded
        virtual_rational_prices = (ATT.unsqueeze(0) * candidate_prices).sum(dim=2)
        u = virtual_rational_prices - prices
        return virtual_rational_prices, u

    def loss(self, prices, virtual_rational_prices, u):
        """Compute combined loss: regression and stationary regularization."""
        T, I = prices.shape
        regression_loss = F.mse_loss(prices, virtual_rational_prices)
        u_lagged = torch.cat([torch.zeros(1, I, device=u.device), u[:-1]], dim=0)
        residual = u - self.rho * u_lagged
        stationary_loss = F.mse_loss(residual, torch.zeros_like(residual))
        rho_penalty = torch.clamp(torch.abs(self.rho) - 1, min=0).pow(2).sum()
        total_loss = regression_loss + self.lambda1 * stationary_loss + self.lambda_rho * rho_penalty
        return total_loss

class UMIForecastingModel(nn.Module):
    """Module for forecasting stock returns using Transformers and graph attention."""
    def __init__(self, num_stocks, feature_dim, d_model, nhead, num_layers, market_factor_dim, hidden_dim):
        super(UMIForecastingModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.gat = GATConv(d_model, d_model)
        total_input_dim = d_model + d_model + market_factor_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, stock_features, market_factor, edge_index):
        """Predict stock returns for each stock."""
        x = self.input_proj(stock_features)
        transformer_out = self.transformer_encoder(x)
        c_t_minus_1 = transformer_out[-1]
        gat_out = self.gat(c_t_minus_1, edge_index)
        market_factor = market_factor.repeat(I, 1)
        combined = torch.cat([c_t_minus_1, gat_out, market_factor], dim=1)
        prediction = self.mlp(combined).squeeze(1)
        return prediction

    def train_step(self, stock_features, market_factor, true_returns, edge_index):
        """Compute training loss: MSE + RankIC."""
        predictions = self.forward(stock_features, market_factor, edge_index)
        mse_loss = F.mse_loss(predictions, true_returns)
        rank_pred = predictions.argsort().argsort().float()
        rank_true = true_returns.argsort().argsort().float()
        pred_std = rank_pred.std()
        true_std = rank_true.std()
        if pred_std > 0 and true_std > 0:
            corr = ((rank_pred - rank_pred.mean()) * (rank_true - rank_true.mean())).mean() / (pred_std * true_std)
        else:
            corr = torch.tensor(0.0, device=predictions.device)
        rankic_loss = -corr
        total_loss = mse_loss + 0.1 * rankic_loss
        return total_loss

class UMIModel(nn.Module):
    """Complete UMI model integrating stock-level and forecasting components."""
    def __init__(self, num_stocks, feature_dim, d_model, nhead, num_layers, market_factor_dim, hidden_dim, embedding_dim):
        super(UMIModel, self).__init__()
        self.stock_level = StockLevelFactorLearning(num_stocks, embedding_dim)
        self.forecasting = UMIForecastingModel(num_stocks, feature_dim, d_model, nhead, num_layers, market_factor_dim, hidden_dim)

    def forward(self, prices, market_factor, edge_index):
        """Forward pass for the entire UMI model."""
        _, u = self.stock_level(prices)
        stock_features = torch.stack([prices, u], dim=2)
        predictions = self.forecasting(stock_features, market_factor, edge_index)
        return predictions

    def factor_learning_step(self, prices):
        """Step 1: Train stock-level factors."""
        virtual_rational_prices, u = self.stock_level(prices)
        loss = self.stock_level.loss(prices, virtual_rational_prices, u)
        return loss

    def forecasting_training_step(self, prices, market_factor, true_returns, edge_index):
        """Step 2: Train forecasting model with pre-learned factors."""
        with torch.no_grad():
            _, u = self.stock_level(prices)
        stock_features = torch.stack([prices, u], dim=2)
        loss = self.forecasting.train_step(stock_features, market_factor, true_returns, edge_index)
        return loss

class UMIModelAlgorithm(QCAlgorithm):
    """QuantConnect algorithm integrating the UMI model for stock return forecasting."""
    def __init__(self):
        self.model = None
        self.symbols = []
        self.lookback = 30
        self.num_stocks = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Initialize(self):
        """Initialize the algorithm, set up securities, and load the model."""
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        self.symbols = ["AAPL", "MSFT", "GOOGL"]  # Example stocks
        self.num_stocks = len(self.symbols)
        for symbol in self.symbols:
            self.AddEquity(symbol, Resolution.Daily)
        self.model = UMIModel(
            num_stocks=self.num_stocks,
            feature_dim=2,
            d_model=64,
            nhead=8,
            num_layers=2,
            market_factor_dim=32,
            hidden_dim=128,
            embedding_dim=16
        ).to(self.device)
        try:
            self.model.load_state_dict(torch.load("umi_model.pth", map_location=self.device))
            self.model.eval()
            self.Debug("Loaded pre-trained UMI model.")
        except FileNotFoundError:
            self.Debug("Model file not found. Please train and save the model first.")

    def OnData(self, data):
        """Handle incoming data and make trading decisions."""
        if not all(symbol in data and data[symbol].Close is not None for symbol in self.symbols):
            return
        history = self.History(self.symbols, self.lookback, Resolution.Daily)
        if history.empty or len(history) < self.lookback:
            return
        prices = np.array([[history.loc[symbol]["close"].iloc[i] for symbol in self.symbols]
                          for i in range(self.lookback)])
        prices = torch.tensor(prices, dtype=torch.float32).to(self.device)
        edge_index = self._compute_edge_index()
        market_factor = torch.zeros(32, device=self.device)  # Placeholder
        with torch.no_grad():
            predictions = self.model(prices, market_factor, edge_index)
        top_stock_idx = predictions.argmax().item()
        top_symbol = self.symbols[top_stock_idx]
        if predictions[top_stock_idx] > 0:  # Only trade if predicted return is positive
            self.SetHoldings(top_symbol, 0.3)
            self.Debug(f"Buying {top_symbol} with predicted return {predictions[top_stock_idx]:.4f}")
        else:
            self.Liquidate(top_symbol, "Negative predicted return")
            self.Debug(f"Liquidating {top_symbol} with predicted return {predictions[top_stock_idx]:.4f}")

    def _compute_edge_index(self):
        """Compute a simple edge index based on stock correlations."""
        history = self.History(self.symbols, self.lookback, Resolution.Daily)
        if history.empty:
            return torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        returns = np.array([[history.loc[symbol]["close"].pct_change().iloc[i]
                           for symbol in self.symbols]
                           for i in range(1, self.lookback)])
        corr_matrix = np.corrcoef(returns.T)
        edges = []
        for i in range(self.num_stocks):
            for j in range(i + 1, self.num_stocks):
                if corr_matrix[i, j] > 0.5:  # Threshold for edge
                    edges.append([i, j])
                    edges.append([j, i])
        if not edges:
            edges = [[0, 0]]  # Dummy edge
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        return edge_index

    def TrainModel(self, prices, true_returns, market_factor, edge_index):
        """Train the UMI model offline (for demonstration)."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Step 1: Factor learning
        for epoch in range(10):  # Example epochs
            optimizer.zero_grad()
            loss = self.model.factor_learning_step(prices)
            loss.backward()
            optimizer.step()
            self.Debug(f"Factor Learning Epoch {epoch+1}, Loss: {loss.item():.4f}")
        # Step 2: Forecasting training
        for epoch in range(10):
            optimizer.zero_grad()
            loss = self.model.forecasting_training_step(prices, market_factor, true_returns, edge_index)
            loss.backward()
            optimizer.step()
            self.Debug(f"Forecasting Epoch {epoch+1}, Loss: {loss.item():.4f}")
        torch.save(self.model.state_dict(), "umi_model.pth")
        self.model.eval()