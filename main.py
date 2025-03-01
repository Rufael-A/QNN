# region imports
from AlgorithmImports import *
# endregion
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data.Market import TradeBar
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer
import numpy as np

class QuantumNeuralNetworkTrading(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Load trained parameters (θ0 to θ7)
        self.trained_params = np.array([3.3926, 5.3705, 2.8713, 4.3222, 5.1231, 3.8132, 3.4503, 1.7366])
        self.num_qubits = 2
        
        # Initialize quantum circuit
        self.circuit, self.params = self._create_circuit()
        self.simulator = Aer.get_backend('aer_simulator')
        self.transpiled_circuit = transpile(self.circuit, self.simulator)
        
        # Schedule trading logic
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), 
                        self.TimeRules.AfterMarketOpen(self.symbol, 30),
                        self.Trade)

    def _create_circuit(self):
        """Recreate the QNN circuit architecture"""
        circuit = QuantumCircuit(self.num_qubits)
        params = [Parameter(f'θ{i}') for i in range(4 * self.num_qubits)]
        
        # Input encoding layer
        for i in range(self.num_qubits):
            circuit.rx(params[i], i)
            circuit.ry(params[i + self.num_qubits], i)
        
        # Entanglement layer
        for i in range(self.num_qubits-1):
            circuit.cx(i, (i+1) % self.num_qubits)
        
        # Additional layers
        for i in range(self.num_qubits):
            circuit.rx(params[i + 2 * self.num_qubits], i)
            circuit.ry(params[i + 3 * self.num_qubits], i)
        
        circuit.measure_all()
        return circuit, params

    def Trade(self):
        """Execute trades based on QNN predictions"""
        # Get historical data
        history = self.History(self.symbol, 5, Resolution.Daily)
        if history.empty:
            return
        
        # Feature engineering (example: use closing prices as inputs)
        latest_prices = history["close"].values[-2:]  # Last 2 days' closing prices
        normalized_input = (latest_prices - np.min(latest_prices)) / (np.max(latest_prices) - np.min(latest_prices)) * np.pi
        
        # Run QNN inference
        prediction = self.predict(normalized_input)
        
        # Execute trades
        if prediction == 1:
            self.SetHoldings(self.symbol, 1.0)  # Buy
        else:
            self.Liquidate(self.symbol)  # Sell

    def predict(self, x):
        """Make predictions using the trained QNN"""
        param_values = np.concatenate([x, self.trained_params[self.num_qubits:]])
        param_dict = {p: param_values[i] for i, p in enumerate(self.params)}
        
        bound_qc = self.transpiled_circuit.assign_parameters(param_dict)
        result = self.simulator.run(bound_qc, shots=1000).result()
        counts = result.get_counts()
        
        prob_class0 = counts.get('00', 0) / 1000
        prob_class1 = counts.get('11', 0) / 1000
        return 0 if prob_class0 > prob_class1 else 1
        
