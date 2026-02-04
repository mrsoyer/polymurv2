# Smart Contract Event Monitoring

## Overview

Smart contract events provide real-time signals for prediction market trading. This guide covers implementation patterns for monitoring Polymarket's CTF Exchange events using web3.py and ethers.js.

## Event Fundamentals

### What Are Events?

Events (also called logs) are emitted by smart contracts when specific actions occur. They provide:
- **Real-time notifications** of on-chain activity
- **Historical record** queryable via blockchain explorers
- **Indexed parameters** for efficient filtering
- **Cost-effective** logging (cheaper than storage)

### Event Structure

```solidity
event OrderFilled(
    bytes32 indexed orderHash,
    address indexed maker,
    address indexed taker,
    uint256 makerAssetId,
    uint256 takerAssetId,
    uint256 makerAmountFilled,
    uint256 takerAmountFilled,
    uint256 fee
);
```

**Components**:
- **Event Signature**: Keccak-256 hash of event name and parameters (topic[0])
- **Indexed Parameters**: Up to 3 searchable parameters (topics[1-3])
- **Non-Indexed Data**: Remaining parameters in `data` field
- **Transaction Metadata**: Block number, transaction hash, log index

## Polymarket CTF Exchange Events

### Key Contracts

| Contract | Address | Purpose |
|----------|---------|---------|
| **CTF Exchange** | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Binary markets |
| **NegRisk Exchange** | (varies) | Multi-outcome markets |
| **Conditional Tokens** | `0x4d97dcd97ec945f40cf65f87097ace5ea0476045` | Token management |

### Primary Events

#### 1. OrderFilled
Emitted when a market order is executed.

```solidity
event OrderFilled(
    bytes32 indexed orderHash,
    address indexed maker,
    address indexed taker,
    uint256 makerAssetId,
    uint256 takerAssetId,
    uint256 makerAmountFilled,
    uint256 takerAmountFilled,
    uint256 fee
);
```

**Use Cases**:
- Track individual trades
- Monitor maker/taker activity
- Calculate effective prices
- Detect large orders

**Event Signature**: `0x...` (compute with `keccak256("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")`)

#### 2. OrdersMatched
Emitted when multiple orders are matched together.

```solidity
event OrdersMatched(
    bytes32 indexed takerOrderHash,
    bytes32[] makerOrderHashes,
    uint256 takerAssetFilledAmount,
    uint256 makerAssetFilledAmount
);
```

**Use Cases**:
- Batch trade detection
- Market maker activity monitoring
- Volume spike alerts

#### 3. PositionsSplit
Emitted when user creates new outcome positions.

```solidity
event PositionsSplit(
    address indexed stakeholder,
    address indexed collateralToken,
    bytes32 indexed parentCollectionId,
    bytes32 conditionId,
    uint256[] partition,
    uint256 amount
);
```

**Use Cases**:
- New position creation
- Market entry signals
- Liquidity addition tracking

#### 4. PositionsMerged
Emitted when user redeems positions.

```solidity
event PositionsMerged(
    address indexed stakeholder,
    address indexed collateralToken,
    bytes32 indexed parentCollectionId,
    bytes32 conditionId,
    uint256[] partition,
    uint256 amount
);
```

**Use Cases**:
- Position exits
- Profit-taking detection
- Market closure tracking

## Implementation: ethers.js

### Setup

```bash
npm install ethers dotenv
```

```javascript
import { ethers } from 'ethers';
import dotenv from 'dotenv';

dotenv.config();

// WebSocket provider for real-time events
const provider = new ethers.WebSocketProvider(
  `wss://polygon-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`
);

// Contract address
const CTF_EXCHANGE = '0x4bFb41d5B3570DeFd03C39a9a4d8de6bd8b8982e';
```

### Listening to Events

#### Basic Event Listener

```javascript
// Minimal ABI with just the events we want
const abi = [
  "event OrderFilled(bytes32 indexed orderHash, address indexed maker, address indexed taker, uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)"
];

const contract = new ethers.Contract(CTF_EXCHANGE, abi, provider);

// Listen to all OrderFilled events
contract.on('OrderFilled', (orderHash, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee, event) => {
  console.log('Order Filled!');
  console.log('Maker:', maker);
  console.log('Taker:', taker);
  console.log('Amount:', ethers.formatUnits(makerAmountFilled, 6)); // USDC has 6 decimals
  console.log('Fee:', ethers.formatUnits(fee, 6));
  console.log('Tx Hash:', event.log.transactionHash);
});
```

#### Filtered Event Listener

```javascript
// Listen only to trades where specific address is the maker
const myAddress = '0xYourAddressHere';

const filter = contract.filters.OrderFilled(null, myAddress, null);

contract.on(filter, async (orderHash, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee, event) => {
  console.log(`You made a trade! Filled ${ethers.formatUnits(makerAmountFilled, 6)} USDC`);

  // Fetch additional transaction details
  const tx = await event.log.getTransaction();
  const block = await event.log.getBlock();

  console.log('Block:', block.number);
  console.log('Timestamp:', new Date(block.timestamp * 1000).toISOString());
  console.log('Gas Used:', tx.gasLimit.toString());
});
```

#### Multi-Event Listener

```javascript
class PolymarketEventMonitor {
  constructor(provider, contractAddress) {
    this.provider = provider;
    this.contract = new ethers.Contract(contractAddress, abi, provider);
    this.handlers = {};
  }

  addHandler(eventName, callback) {
    this.handlers[eventName] = callback;
  }

  start() {
    // Listen to OrderFilled
    this.contract.on('OrderFilled', (...args) => {
      const event = args[args.length - 1];
      if (this.handlers['OrderFilled']) {
        this.handlers['OrderFilled'](...args);
      }
    });

    // Listen to OrdersMatched
    this.contract.on('OrdersMatched', (...args) => {
      if (this.handlers['OrdersMatched']) {
        this.handlers['OrdersMatched'](...args);
      }
    });

    console.log('Event monitoring started...');
  }

  stop() {
    this.contract.removeAllListeners();
    console.log('Event monitoring stopped.');
  }
}

// Usage
const monitor = new PolymarketEventMonitor(provider, CTF_EXCHANGE);

monitor.addHandler('OrderFilled', (orderHash, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee, event) => {
  console.log(`Trade: ${ethers.formatUnits(makerAmountFilled, 6)} USDC`);
});

monitor.start();
```

### Querying Historical Events

```javascript
async function getRecentTrades(hours = 24) {
  const currentBlock = await provider.getBlockNumber();
  const blocksPerHour = 1800; // Polygon: ~2 second blocks
  const fromBlock = currentBlock - (hours * blocksPerHour);

  const filter = contract.filters.OrderFilled();
  const events = await contract.queryFilter(filter, fromBlock, currentBlock);

  return events.map(event => ({
    orderHash: event.args.orderHash,
    maker: event.args.maker,
    taker: event.args.taker,
    amount: ethers.formatUnits(event.args.makerAmountFilled, 6),
    fee: ethers.formatUnits(event.args.fee, 6),
    blockNumber: event.blockNumber,
    txHash: event.transactionHash
  }));
}

// Get trades from last 24 hours
const trades = await getRecentTrades(24);
console.log(`Found ${trades.length} trades in last 24 hours`);
```

### Advanced: Decode Raw Logs

```javascript
// For custom processing or handling unknown events
provider.on({
  address: CTF_EXCHANGE,
  topics: [
    ethers.id("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
  ]
}, (log) => {
  // Manually decode
  const iface = new ethers.Interface(abi);
  const decoded = iface.parseLog({
    topics: log.topics,
    data: log.data
  });

  console.log('Decoded event:', decoded.name);
  console.log('Arguments:', decoded.args);
});
```

## Implementation: web3.py

### Setup

```bash
pip install web3 python-dotenv
```

```python
import os
import asyncio
from web3 import AsyncWeb3, WebSocketProvider
from web3.contract import AsyncContract
from dotenv import load_dotenv

load_dotenv()

# WebSocket connection
w3 = AsyncWeb3(WebSocketProvider(
    f"wss://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}"
))

CTF_EXCHANGE = '0x4bFb41d5B3570DeFd03C39a9a4d8de6Bd8B8982E'
```

### Event Listener with Subscription

```python
import json

# Load ABI
with open('ctf_exchange_abi.json') as f:
    abi = json.load(f)

async def listen_to_events():
    """Real-time event listener using WebSocket subscription."""
    contract = w3.eth.contract(address=CTF_EXCHANGE, abi=abi)

    # Subscribe to logs for this contract
    subscription_id = await w3.eth.subscribe('logs', {
        'address': CTF_EXCHANGE,
        'topics': [
            w3.keccak(text='OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)').hex()
        ]
    })

    print(f'Subscribed with ID: {subscription_id}')

    async for response in w3.socket.process_subscriptions():
        if response.get('result'):
            log = response['result']

            # Decode event
            decoded = contract.events.OrderFilled().process_log(log)

            print(f"Order Filled!")
            print(f"  Maker: {decoded['args']['maker']}")
            print(f"  Taker: {decoded['args']['taker']}")
            print(f"  Amount: {decoded['args']['makerAmountFilled'] / 1e6} USDC")
            print(f"  Tx: {decoded['transactionHash'].hex()}")
            print()

# Run
asyncio.run(listen_to_events())
```

### Polling Pattern (HTTP Provider)

```python
from web3 import Web3
import time

# HTTP provider (for polling)
w3 = Web3(Web3.HTTPProvider(
    f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}"
))

contract = w3.eth.contract(address=CTF_EXCHANGE, abi=abi)

def poll_events(poll_interval=5):
    """Poll for new events every N seconds."""
    # Start from current block
    from_block = w3.eth.block_number

    while True:
        try:
            to_block = w3.eth.block_number

            if to_block > from_block:
                # Get OrderFilled events in this range
                events = contract.events.OrderFilled().get_logs(
                    fromBlock=from_block,
                    toBlock=to_block
                )

                for event in events:
                    print(f"New trade: {event['args']['maker']} → {event['args']['taker']}")
                    print(f"  Amount: {event['args']['makerAmountFilled'] / 1e6} USDC")

                from_block = to_block + 1

            time.sleep(poll_interval)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(poll_interval)

poll_events()
```

### Batch Historical Query

```python
def get_trades_in_range(from_block, to_block):
    """Fetch all OrderFilled events in block range."""
    # For large ranges, chunk into smaller batches
    CHUNK_SIZE = 10000

    all_events = []

    for start in range(from_block, to_block, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, to_block)

        print(f"Querying blocks {start} to {end}...")

        events = contract.events.OrderFilled().get_logs(
            fromBlock=start,
            toBlock=end
        )

        all_events.extend(events)

    return all_events

# Get last 100,000 blocks (~55 hours on Polygon)
current_block = w3.eth.block_number
from_block = current_block - 100000

trades = get_trades_in_range(from_block, current_block)
print(f"Found {len(trades)} trades")
```

### Redundant Event Listener (Production)

```python
import logging
from typing import List

class RedundantEventListener:
    """Monitor events using multiple RPC endpoints for reliability."""

    def __init__(self, websocket_urls: List[str]):
        self.providers = [
            AsyncWeb3(WebSocketProvider(url))
            for url in websocket_urls
        ]
        self.seen_events = set()  # Track event hashes to avoid duplicates

    async def listen_all_providers(self):
        """Listen on all providers concurrently."""
        tasks = [
            self.listen_provider(provider, idx)
            for idx, provider in enumerate(self.providers)
        ]
        await asyncio.gather(*tasks)

    async def listen_provider(self, provider, provider_id):
        """Listen to events from single provider."""
        contract = provider.eth.contract(address=CTF_EXCHANGE, abi=abi)

        try:
            subscription_id = await provider.eth.subscribe('logs', {
                'address': CTF_EXCHANGE
            })

            logging.info(f"Provider {provider_id} subscribed: {subscription_id}")

            async for response in provider.socket.process_subscriptions():
                if response.get('result'):
                    log = response['result']

                    # Create unique event hash
                    event_hash = f"{log['transactionHash']}-{log['logIndex']}"

                    if event_hash not in self.seen_events:
                        self.seen_events.add(event_hash)

                        # Process event
                        await self.process_event(log, provider_id)

        except Exception as e:
            logging.error(f"Provider {provider_id} error: {e}")
            # Retry connection
            await asyncio.sleep(5)
            await self.listen_provider(provider, provider_id)

    async def process_event(self, log, provider_id):
        """Handle incoming event."""
        print(f"Event from provider {provider_id}: {log['transactionHash']}")
        # Your processing logic here

# Usage with multiple endpoints
listener = RedundantEventListener([
    'wss://polygon-mainnet.g.alchemy.com/v2/KEY1',
    'wss://polygon-mainnet.infura.io/ws/v3/KEY2',
    'wss://polygon-mainnet.g.quicknode.pro/KEY3'
])

asyncio.run(listener.listen_all_providers())
```

## Event Decoding Patterns

### Decode OrderFilled Event

```javascript
// ethers.js
function decodeOrderFilled(event) {
  const { args, blockNumber, transactionHash } = event;

  return {
    orderHash: args.orderHash,
    maker: args.maker,
    taker: args.taker,
    makerAssetId: args.makerAssetId.toString(),
    takerAssetId: args.takerAssetId.toString(),
    makerAmountFilled: parseFloat(ethers.formatUnits(args.makerAmountFilled, 6)),
    takerAmountFilled: parseFloat(ethers.formatUnits(args.takerAmountFilled, 6)),
    fee: parseFloat(ethers.formatUnits(args.fee, 6)),
    effectivePrice: calculatePrice(args.makerAmountFilled, args.takerAmountFilled),
    blockNumber,
    txHash: transactionHash
  };
}

function calculatePrice(makerAmount, takerAmount) {
  // Price = takerAmount / (makerAmount + takerAmount)
  const maker = parseFloat(ethers.formatUnits(makerAmount, 6));
  const taker = parseFloat(ethers.formatUnits(takerAmount, 6));
  return taker / (maker + taker);
}
```

### Aggregate Events

```python
from collections import defaultdict
import pandas as pd

def aggregate_trades_by_market(events):
    """Group trades by market and calculate statistics."""
    market_stats = defaultdict(lambda: {
        'total_volume': 0,
        'num_trades': 0,
        'unique_traders': set(),
        'avg_trade_size': 0
    })

    for event in events:
        market_id = event['args']['makerAssetId']
        amount = event['args']['makerAmountFilled'] / 1e6

        stats = market_stats[market_id]
        stats['total_volume'] += amount
        stats['num_trades'] += 1
        stats['unique_traders'].add(event['args']['maker'])
        stats['unique_traders'].add(event['args']['taker'])

    # Convert to DataFrame
    data = []
    for market_id, stats in market_stats.items():
        data.append({
            'market_id': market_id,
            'total_volume': stats['total_volume'],
            'num_trades': stats['num_trades'],
            'unique_traders': len(stats['unique_traders']),
            'avg_trade_size': stats['total_volume'] / stats['num_trades']
        })

    return pd.DataFrame(data).sort_values('total_volume', ascending=False)
```

## Performance Optimization

### Connection Pooling

```javascript
class ConnectionPool {
  constructor(providerUrls, maxConnections = 3) {
    this.providers = providerUrls.map(url =>
      new ethers.WebSocketProvider(url)
    );
    this.currentIndex = 0;
  }

  getProvider() {
    // Round-robin provider selection
    const provider = this.providers[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.providers.length;
    return provider;
  }
}
```

### Rate Limiting

```python
import time
from functools import wraps

def rate_limit(max_per_second):
    """Decorator to rate limit function calls."""
    min_interval = 1.0 / max_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:
                time.sleep(left_to_wait)

            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper
    return decorator

@rate_limit(max_per_second=5)
def query_events(from_block, to_block):
    return contract.events.OrderFilled().get_logs(
        fromBlock=from_block,
        toBlock=to_block
    )
```

### Caching

```javascript
import NodeCache from 'node-cache';

const cache = new NodeCache({ stdTTL: 60 }); // 60 second TTL

async function getCachedEvents(fromBlock, toBlock) {
  const cacheKey = `events:${fromBlock}:${toBlock}`;

  // Check cache first
  const cached = cache.get(cacheKey);
  if (cached) {
    console.log('Cache hit!');
    return cached;
  }

  // Query blockchain
  const events = await contract.queryFilter(
    contract.filters.OrderFilled(),
    fromBlock,
    toBlock
  );

  // Cache results
  cache.set(cacheKey, events);
  return events;
}
```

## Error Handling

### Reconnection Logic

```javascript
class RobustEventListener {
  constructor(providerUrl, contractAddress, abi) {
    this.providerUrl = providerUrl;
    this.contractAddress = contractAddress;
    this.abi = abi;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
  }

  async start() {
    try {
      this.provider = new ethers.WebSocketProvider(this.providerUrl);
      this.contract = new ethers.Contract(
        this.contractAddress,
        this.abi,
        this.provider
      );

      // Reset reconnect counter on successful connection
      this.reconnectAttempts = 0;

      this.contract.on('OrderFilled', (...args) => {
        this.handleEvent(...args);
      });

      // Handle WebSocket errors
      this.provider.websocket.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.reconnect();
      });

      this.provider.websocket.on('close', () => {
        console.log('WebSocket closed');
        this.reconnect();
      });

      console.log('Event listener started');

    } catch (error) {
      console.error('Start error:', error);
      this.reconnect();
    }
  }

  async reconnect() {
    this.reconnectAttempts++;

    if (this.reconnectAttempts > this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    await new Promise(resolve => setTimeout(resolve, delay));
    await this.start();
  }

  handleEvent(...args) {
    try {
      // Your event processing logic
      console.log('Event received:', args);
    } catch (error) {
      console.error('Event handling error:', error);
    }
  }
}

const listener = new RobustEventListener(
  'wss://polygon-mainnet.g.alchemy.com/v2/YOUR-KEY',
  CTF_EXCHANGE,
  abi
);

listener.start();
```

## References

### Technical Documentation
- [Consensys: Events and Logs Guide](https://consensys.io/blog/guide-to-events-and-logs-in-ethereum-smart-contracts)
- [Moralis: Ethers.js Event Listening](https://moralis.com/how-to-listen-to-smart-contract-events-using-ethers-js/)
- [Ethers.js Events Documentation](https://docs.ethers.org/v5/concepts/events/)

### Code Examples
- [Web3.js Events Documentation](https://web3js.readthedocs.io/en/v1.2.11/web3-eth-contract.html)
- [Web3.py Events and Logs](https://web3py.readthedocs.io/en/stable/filters.html)
- [Chainstack: Ethereum Event Listener (Python)](https://docs.chainstack.com/docs/ethereum-redundant-event-listener-python-version)

### Community Resources
- [DEV: Real-time Blockchain Updates with web3.py](https://dev.to/divine_igbinoba_fb6de7207/part-4-real-time-blockchain-updates-listening-for-smart-contract-events-with-web3py-32dl)
- [Medium: Building Web3 Event Listener](https://medium.com/@pearliboy1/building-a-web3-event-listener-a-comprehensive-guide-to-tracking-specific-blockchain-events-d562dc8a635b)

---

**Version**: 1.0
**Last Updated**: 2026-02-04
