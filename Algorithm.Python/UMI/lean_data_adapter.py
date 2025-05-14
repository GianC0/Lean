'''
Custom Data Adapter for UMI Model in QuantConnect Lean.

This adapter converts Lean's `Slice` data into a format suitable for the UMI model.
It maintains a rolling window of historical data for each symbol.

Key Considerations:
- Feature Engineering: This basic adapter provides sequences of raw OHLCV data.
  The original UMI model might expect more complex, qlib-generated features.
  If so, this adapter or the model itself will need further modification.
- Stock ID Mapping: UMI models sometimes use internal integer IDs for stocks.
  This adapter does not currently implement that mapping. If required by your
  chosen UMI model, this functionality will need to be added.
- Data Normalization: UMI models often pre-process data (e.g., normalization).
  This should be replicated here if necessary.
'''
import torch
import numpy as np
from collections import deque

# If UMI's stk_dic is needed for stock ID mapping, it would be imported from model_pretrain_lean
# from .model_pretrain_lean import stk_dic

class UMILeanDataAdapter:
    def __init__(self, history_window_size=60, features=None):
        self.history_window_size = history_window_size
        self.features = features if features is not None else ['open', 'high', 'low', 'close', 'volume']
        self.feature_map = {f: i for i, f in enumerate(self.features)}
        self.data_buffer = {}  # symbol_str -> deque of feature arrays

        # Placeholder for UMI stock dictionary if needed for ID mapping
        # self.stk_dic_instance = stk_dic() # Example: if stk_dic is a class in model_pretrain_lean
        # self.stk_id_map = {} # Lean Symbol string to UMI StkCode/ID

    def _initialize_symbol_data(self, symbol_str):
        if symbol_str not in self.data_buffer:
            self.data_buffer[symbol_str] = deque(maxlen=self.history_window_size)

    def update_data(self, slice_data):
        """
        Updates the internal buffer with the latest data from Lean's Slice.
        slice_data is expected to be a QuantConnect Slice object.
        """
        for symbol_obj, bar in slice_data.Bars.items(): # Use .items() for dictionary iteration
            symbol_str = str(symbol_obj) # Use the string representation of the Lean Symbol object
            self._initialize_symbol_data(symbol_str)

            current_features = np.zeros(len(self.features))
            # Ensure bar has the necessary attributes (Open, High, Low, Close, Volume)
            # For custom data, these might be different.
            if 'open' in self.feature_map: current_features[self.feature_map['open']] = float(bar.Open)
            if 'high' in self.feature_map: current_features[self.feature_map['high']] = float(bar.High)
            if 'low' in self.feature_map: current_features[self.feature_map['low']] = float(bar.Low)
            if 'close' in self.feature_map: current_features[self.feature_map['close']] = float(bar.Close)
            if 'volume' in self.feature_map: current_features[self.feature_map['volume']] = float(bar.Volume)
            # Add other features if defined
            
            self.data_buffer[symbol_str].append(current_features)

    def get_model_input_for_symbol(self, symbol_str):
        """
        Prepares the data for a single symbol in the format the UMI model expects.
        Returns a PyTorch tensor or None if not enough data.
        UMI's Transformer (`Trans` in `model_seq.py`) expects input `x` of shape 
        (batch_size, seq_len, input_size/num_features).
        For a single stock prediction, batch_size = 1.
        """
        if symbol_str not in self.data_buffer or len(self.data_buffer[symbol_str]) < self.history_window_size:
            return None  # Not enough data

        # Data in deque is [oldest, ..., newest]
        # UMI's sequence format: Fea_feature_CLOSE59 (oldest) ... Fea_feature_CLOSE0 (newest).
        # np.array(deque) preserves this order.
        historical_data = np.array(self.data_buffer[symbol_str])  # Shape: (history_window_size, num_features)
        
        # Reshape for the model: (1, seq_len, num_features)
        model_input_tensor = torch.tensor(historical_data, dtype=torch.float32).unsqueeze(0)
        
        # If stock IDs (`id_out`) or additional features (`addi_x`) are needed by the specific UMI model,
        # they would be prepared and returned here as well, potentially as a tuple or dictionary.
        # For now, returning only the primary feature tensor `x`.
        return model_input_tensor

    # --- Placeholder for UMI Stock ID Mapping Logic (if needed by the model) ---
    # def initialize_stk_dictionary_from_symbols(self, qc_symbols_list):
    #     """ Call this once with all symbols in your universe if ID mapping is needed. """
    #     # Assumes self.stk_dic_instance is an object similar to UMI's stk_dic
    #     # UMI often uses an offset, starting IDs from 2.
    #     # self.stk_dic_instance.stk_dic = {str(qc_symbol): i + 2 for i, qc_symbol in enumerate(qc_symbols_list)}
    #     # self.stk_id_map = {str(qc_symbol): self.stk_dic_instance.stk_code2id(str(qc_symbol)) for qc_symbol in qc_symbols_list}
    #     pass

    # def get_stk_id_tensor_for_symbol(self, symbol_str):
    #     """ Returns the mapped UMI stock ID as a tensor. """
    #     # if symbol_str in self.stk_id_map:
    #     #     stk_id = self.stk_id_map[symbol_str]
    #     #     return torch.tensor([stk_id], dtype=torch.long) # Shape (1) or (1,1) as needed
    #     # else:
    #     #     # Handle unknown symbol - UMI might map it to a default ID (e.g., 1)
    #     #     return torch.tensor([1], dtype=torch.long) # Default/unknown ID
    #     return None
