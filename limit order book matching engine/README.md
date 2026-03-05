# Limit Order Book Matching Engine

A high-performance limit order book (LOB) implementation for real-time order matching and execution. This project demonstrates efficient financial market mechanics with price-time priority execution.

## Overview

This implementation provides a robust and efficient limit order book system that handles:
- Real-time order processing and matching
- Price-time priority execution
- Order cancellation
- Mid-price calculation
- Book depth visualization

## Core Features

### 1. StreamProcessor Class
The main `StreamProcessor` class manages the complete order book state and provides:

- **Order Management**: Add, cancel, and match orders with sequence tracking
- **Price Level Management**: Efficient price-level aggregation with lazy deletion
- **FIFO Execution**: First-in-first-out order matching within price levels
- **Real-time Pricing**: Mid-price calculation based on best bid/ask

### 2. Key Data Structures
- **Order Book**: `seq -> (side, price, remaining_volume)` for active orders
- **Price Queues**: `side -> price -> deque[(seq, remaining_volume)]` for FIFO execution
- **Volume Aggregation**: `side -> price -> total_volume` for quick book depth queries
- **Sorted Price Lists**: Maintained bid/ask price levels for efficient best price lookup

### 3. Market Mechanics
- **BUY Orders**: Match against lowest SELL prices (asks) first
- **SELL Orders**: Match against highest BUY prices (bids) first
- **Price Improvement**: Always executes at most favorable available prices
- **Residual Volume**: Unfilled volume rests at specified price levels

## Files Structure

```
limit-order-book-matching-engine/
├── solution.py              # Main LOB implementation
├── solution.ipynb           # Jupyter notebook with implementation
├── test_public.ipynb        # Test cases and examples
└── README.md               # This documentation
```

## API Reference

### Core Methods

#### `add_message(message: dict) -> None`
Process incoming order messages:
```python
# Add order
sp.add_message({"seq": 1, "side": "BUY", "price": 100.0, "volume": 10})

# Cancel order
sp.add_message({"seq": 1, "side": "BUY", "price": 100.0, "volume": 10, "action": "CANCEL"})
```

#### `get_mid_price() -> float`
Returns the mid-price (average of best bid and best ask):
```python
mid_price = sp.get_mid_price()  # Returns 0.0 if book is incomplete
```

#### `get_book_depth() -> Dict[str, List[Tuple[float, int]]]`
Returns current order book depth:
```python
depth = sp.get_book_depth()
# Returns: {"BIDS": [(price, volume), ...], "ASKS": [(price, volume), ...]}
```

## Performance Characteristics

### Time Complexity
- **Order Addition**: O(log P) where P is number of price levels
- **Order Cancellation**: O(log P) 
- **Order Matching**: O(log P + M) where M is number of matched orders
- **Best Price Lookup**: O(1)
- **Book Depth Query**: O(P)

### Space Complexity
- **Overall**: O(N + P) where N is active orders, P is price levels
- **Lazy Deletion**: Efficient memory management with deferred cleanup

## Testing

The implementation includes comprehensive test cases covering:
- Price priority execution
- Mid-price calculation
- Order cancellation
- Book depth sorting
- Residual volume handling

Run tests with:
```bash
python -m pytest test_public.ipynb
```

## Requirements

- **Python**: 3.11+
- **Dependencies**: Standard library only (no external packages)

## Project Context

This implementation demonstrates core financial market concepts and provides a foundation for understanding:
- Real-time order matching systems
- Market microstructure mechanics
- High-performance data structure design
- Algorithmic trading principles

## Implementation Highlights

### 1. Efficient Price Management
```python
# Maintain sorted price levels using bisect
self.bid_prices: List[float] = []  # Ascending, best bid is last
self.ask_prices: List[float] = []  # Ascending, best ask is first
```

### 2. Lazy Deletion Strategy
```python
# Orders are marked for deletion and cleaned up when encountered
if seq not in self.orders:
    q.popleft()  # Lazy removal
    continue
```

### 3. FIFO Within Price Levels
```python
# Maintain order queues for price-time priority
self.level_queues[side][price] = deque()
self.level_queues[side][price].append((seq, volume))
```

## Usage Example

```python
from solution import StreamProcessor

# Initialize order book
sp = StreamProcessor()

# Add some liquidity
sp.add_message({"seq": 1, "side": "BUY", "price": 99.0, "volume": 10})
sp.add_message({"seq": 2, "side": "SELL", "price": 101.0, "volume": 10})

# Check mid-price
print(f"Mid-price: {sp.get_mid_price()}")  # 100.0

# Process market order
sp.add_message({"seq": 3, "side": "BUY", "price": 101.0, "volume": 5})

# Check book depth
depth = sp.get_book_depth()
print(f"BIDS: {depth['BIDS']}")
print(f"ASKS: {depth['ASKS']}")
```

## Contributing

This is an educational project demonstrating financial market mechanics. Feel free to submit issues, feature requests, or pull requests to improve the implementation.

---

**Note**: This implementation focuses on correctness and efficiency in order matching while maintaining fair market mechanics. The core logic is optimized for real-time trading scenarios while ensuring proper price-time priority execution.
