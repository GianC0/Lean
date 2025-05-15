import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import os
from QuantConnect.Data import SubscriptionDataSource
from QuantConnect.Python import PythonData
from AlgorithmImports import *

# Stock-level Factor Learning Module
class StockLevelFactorLearning(nn.Module):
    def __init__(self, stk_total, embedding_dim=16, dropout=0.1):
        super().__init__()
        self.stock_embeddings = nn.Embedding(stk_total, embedding_dim)
        self.beta = nn.Parameter(torch.randn(stk_total, stk_total))
        self.rho = nn.Parameter(torch.zeros(stk_total))
        self.dropout = nn.Dropout(dropout)

    def forward(self, prices, stock_ids):
        """
        Compute virtual rational prices and irrationality factor u.
        - prices: Tensor of shape (B, I), batch of stock prices.
        - stock_ids: Tensor of shape (B, I), stock indices.
        """
        B, I = prices.shape
        # Get embeddings for selected stocks
        embeddings = self.stock_embeddings(stock_ids)  # (B, I, embedding_dim)
        
        # Compute attention weights (similar to stk_pred_small_2)
        w = torch.bmm(embeddings, embeddings.transpose(1, 2))  # (B, I, I)
        w.diagonal(dim1=1, dim2=2).fill_(-float('inf'))  # Mask self-attention
        exp_w = torch.exp(w)
        sum_exp_w = exp_w.sum(dim=2, keepdim=True) + 1e-8
        ATT = exp_w / sum_exp_w  # Attention weights: (B, I, I)
        ATT = self.dropout(ATT)
        
        # Apply cointegration attention
        beta = self.beta.clone()
        beta.diagonal().zero_()  # No self-contribution
        prices_expanded = prices.unsqueeze(2)  # (B, I, 1)
        beta_expanded = beta[stock_ids, :][:, :, stock_ids]  # (B, I, I)
        candidate_prices = beta_expanded * prices_expanded  # (B, I, I)
        virtual_rational_prices = (ATT * candidate_prices).sum(dim=2)  # (B, I)
        u = virtual_rational_prices - prices  # Irrationality factor: (B, I)
        return virtual_rational_prices, u

    def loss(self, prices, virtual_rational_prices, u, stock_ids, u_old_list):
        """
        Compute stock-level loss with stationary regularization.
        - u_old_list: List of previous u values for each batch.
        """
        B, I = prices.shape
        loss1 = F.mse_loss(prices, virtual_rational_prices)  # Regression loss
        
        # Stationary regularization on u (matches provided code)
        rho = torch.clamp(self.rho, -1, 1)  # Clamp rho between -1 and 1
        loss2 = 0
        for b in range(B):
            u_old = torch.index_select(u_old_list[b], 0, stock_ids[b])  # (I,)
            rho_now = torch.index_select(rho, 0, stock_ids[b])  # (I,)
            diff = u[b] - u_old * rho_now  # (I,)
            square_diff = torch.pow(diff, 2)
            loss2 = loss2 + square_diff.mean()
        loss2 = loss2 / B
        return loss1 + 0.5 * loss2  # Combine losses as per original

# Market-level Factor Learning Module
class MarketLevelFactorLearning(nn.Module):
    def __init__(self, input_size=1, dim_model2=32, stk_total=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_size, dim_model2 // 2)
        self.attn = nn.MultiheadAttention(dim_model2 // 2, num_heads=2, dropout=dropout)
        self.fc = nn.Linear(dim_model2 // 2, dim_model2 // 2)
        self.stock_embeddings = nn.Embedding(stk_total, dim_model2 // 2)
        self.norm = nn.LayerNorm(dim_model2 // 2)
        self.dropout = nn.Dropout(dropout)
        self.L = 5  # Window size
        
        # Market comparison
        self.market_comp = nn.Linear(dim_model2, 1)  # Similar to stk_classification_small_2
        # Market prediction
        self.market_pred = nn.Sequential(
            nn.Linear(dim_model2 // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3 classes for synchronism
        )

    def compute_market_factor(self, u, stock_ids):
        """
        Compute market factor embedding (akin to stk_classification_att1).
        - u: Tensor of shape (T, I), irrationality factors.
        - stock_ids: Tensor of shape (I,), stock indices.
        """
        T, I = u.shape
        u_embed = self.embed(u.unsqueeze(-1))  # (T, I, dim_model2 // 2)
        stock_emb = self.stock_embeddings(stock_ids)  # (I, dim_model2 // 2)
        
        # Attention over historical window
        u_seq = u_embed.permute(1, 0, 2)  # (I, T, dim_model2 // 2)
        attn_output, _ = self.attn(u_seq, u_seq, u_seq)  # (I, T, dim_model2 // 2)
        attn_output = attn_output.permute(1, 0, 2)  # (T, I, dim_model2 // 2)
        
        # Combine with stock embeddings
        total_embed = self.fc(attn_output[-1]) + stock_emb  # (I, dim_model2 // 2)
        total_embed = self.norm(total_embed)
        total_embed = self.dropout(total_embed)
        return total_embed.mean(dim=0)  # (dim_model2 // 2)

    def loss(self, u, returns, stock_ids):
        """
        Compute market-level self-supervised loss.
        """
        T, I = u.shape
        # Split stocks into two groups
        perm = torch.randperm(I)
        group1, group2 = perm[:I//2], perm[I//2:]
        
        embed1 = self.compute_market_factor(u, stock_ids[group1])  # (dim_model2 // 2)
        embed2 = self.compute_market_factor(u, stock_ids[group2])  # (dim_model2 // 2)
        concat = torch.cat([embed1, embed2]).unsqueeze(0)  # (1, dim_model2)
        score = self.market_comp(concat)  # (1, 1)
        label = torch.zeros(1, dtype=torch.long, device=u.device)
        loss1 = F.cross_entropy(score.unsqueeze(0), label)  # Comparative loss
        
        # Market synchronism prediction
        total_embed = self.compute_market_factor(u, stock_ids)
        signs = torch.sign(returns[1:])  # (T-1, I)
        S = signs.sum(dim=1)  # (T-1,)
        H_m = I * 0.5
        labels = torch.zeros(T-1, dtype=torch.long, device=u.device)
        labels[S > H_m] = 0  # Positive
        labels[S < -H_m] = 1  # Negative
        labels[(S >= -H_m) & (S <= H_m)] = 2  # Neutral
        pred = self.market_pred(total_embed.unsqueeze(0))  # (1, 3)
        loss2 = F.cross_entropy(pred, labels[-1].unsqueeze(0))  # Last time step
        return loss1 + loss2

# Forecasting Module
class UMIForecastingModel(nn.Module):
    def __init__(self, input_size=1, num_heads=8, dim_model=64, dim_ff=128, seq_len=30, num_layers=2, dropout=0.1, add_xdim=32):
        super().__init__()
        self.input_proj = nn.Linear(input_size, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(dim_model + add_xdim, 1)  # Incorporates market factor
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, dim_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, addi_x=None):
        """
        Forecast returns using transformer (matches Trans model).
        - x: Tensor of shape (T, I, input_size), price sequence.
        - addi_x: Tuple of (market_embed, _), market factor embedding.
        """
        T, I, _ = x.shape
        x = self.input_proj(x) + self.pos_encoding[:, :T, :]  # (T, I, dim_model)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # (I, T, dim_model)
        transformer_out = self.transformer(x)  # (I, T, dim_model)
        last_out = transformer_out[:, -1, :]  # (I, dim_model)
        
        if addi_x is not None:
            market_embed, _ = addi_x  # market_embed: (dim_model2 // 2)
            market_embed = market_embed[:I]  # Match batch size
            combined = torch.cat([last_out, market_embed], dim=1)  # (I, dim_model + add_xdim)
        else:
            combined = last_out
        return self.fc_out(combined).squeeze(1)  # (I,)

# Complete UMI Model
class UMIModel(nn.Module):
    def __init__(self, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.stock_level = StockLevelFactorLearning(stk_total=num_stocks)
        self.market_level = MarketLevelFactorLearning(stk_total=num_stocks)
        self.forecasting = UMIForecastingModel(add_xdim=16)  # dim_model2 // 2 = 16

    def factor_learning_step(self, prices, returns, stock_ids, u_old_list):
        virtual_prices, u = self.stock_level(prices, stock_ids)
        stock_loss = self.stock_level.loss(prices, virtual_prices, u, stock_ids, u_old_list)
        market_loss = self.market_level.loss(u, returns, stock_ids)
        return stock_loss + market_loss

    def forward(self, prices, stock_ids):
        _, u = self.stock_level(prices[-self.forecasting.seq_len:], stock_ids)
        market_embed = self.market_level.compute_market_factor(u, stock_ids)
        x = prices.unsqueeze(-1)  # (T, I, 1)
        addi_x = (market_embed.repeat(self.num_stocks, 1), None)
        return self.forecasting(x, addi_x)

# Lean Algorithm
class UMIModelAlgorithm(QCAlgorithm):
    def initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2023, 1, 1)
        self.SetCash(100000)
        self.symbols = [self.AddEquity(s, Resolution.Daily).Symbol for s in ["SPY", "AAPL", "MSFT"]]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UMIModel(len(self.symbols)).to(self.device)
        self.model_path = "umi_model.pth"
        self.lookback = 30
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.stock_ids = torch.arange(len(self.symbols), device=self.device)
        
        if not os.path.exists(self.model_path):
            self.TrainModel()
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def TrainModel(self):
        history = self.History(self.symbols, 1000, Resolution.Daily)
        if history.empty or len(history) < 1000:
            self.Debug("Insufficient data for training.")
            return
        prices = np.array([[history.loc[s]["close"].iloc[i] for s in self.symbols] 
                          for i in range(len(history.index.levels[1]))])
        prices = torch.tensor(prices[:-1], dtype=torch.float32).to(self.device)  # (T-1, I)
        returns = (prices[1:] - prices[:-1]) / prices[:-1]  # (T-2, I)
        prices = prices[:-1]  # (T-2, I)
        T = prices.shape[0]
        
        self.model.train()
        optimizer_factor = torch.optim.Adam(
            list(self.model.stock_level.parameters()) + list(self.model.market_level.parameters()), 
            lr=0.001
        )
        u_old_list = [torch.zeros(len(self.symbols), device=self.device) 
                     for _ in range(32)]  # Batch size 32
        
        # Factor learning
        for epoch in range(10):
            total_loss = 0
            for t in range(0, T - 32, 32):
                batch_prices = prices[t:t+32]
                batch_returns = returns[t:t+32] if t+32 <= T-1 else returns[t:]
                loss = self.model.factor_learning_step(batch_prices, batch_returns, 
                                                      self.stock_ids.unsqueeze(0).repeat(32, 1), 
                                                      u_old_list)
                optimizer_factor.zero_grad()
                loss.backward()
                optimizer_factor.step()
                total_loss += loss.item()
                
                # Update u_old_list
                with torch.no_grad():
                    _, u = self.model.stock_level(batch_prices, self.stock_ids.unsqueeze(0).repeat(32, 1))
                    for b in range(min(32, u.shape[0])):
                        u_old_list[b] = u[b].detach()
            self.Debug(f"Factor Learning Epoch {epoch+1}, Loss: {total_loss / (T // 32):.4f}")
        
        # Forecasting
        optimizer_forecast = torch.optim.Adam(self.model.forecasting.parameters(), lr=0.001)
        for epoch in range(10):
            total_loss = 0
            for t in range(self.lookback, T):
                x = prices[t-self.lookback:t]
                y = returns[t-1]
                pred = self.model(x, self.stock_ids)
                loss = F.mse_loss(pred, y) + 0.1 * (-torch.corrcoef(torch.stack([pred, y]))[0,1])
                optimizer_forecast.zero_grad()
                loss.backward()
                optimizer_forecast.step()
                total_loss += loss.item()
            self.Debug(f"Forecasting Epoch {epoch+1}, Loss: {total_loss / (T - self.lookback):.4f}")
        
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()

    def on_data(self, data):
        for symbol in self.symbols:
            if symbol in data:
                self.price_history[symbol].append(float(data[symbol].Close))
                if len(self.price_history[symbol]) > self.lookback:
                    self.price_history[symbol].pop(0)
        if all(len(self.price_history[s]) == self.lookback for s in self.symbols):
            prices = torch.tensor([[self.price_history[s][i] for s in self.symbols] 
                                 for i in range(self.lookback)], 
                                 dtype=torch.float32).to(self.device)
            with torch.no_grad():
                predictions = self.model(prices, self.stock_ids)
            for i, symbol in enumerate(self.symbols):
                if predictions[i] > 0 and not self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, 0.3)
                elif predictions[i] < 0 and self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)