# Complete On-Chain Analysis Implementation Guide

## Overview

This guide provides production-ready code for implementing on-chain analysis systems for Polymarket trading. Includes whale tracking, event monitoring, signal generation, and trading automation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  • Polygon RPC (WebSocket)      • The Graph (GraphQL)      │
│  • PolygonScan API              • Alchemy Webhooks          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVENT PROCESSING LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  • Event Decoding               • Data Normalization        │
│  • Duplicate Detection          • Error Handling            │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                     ANALYSIS LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  • Whale Detection              • Pattern Recognition       │
│  • Address Clustering           • Sentiment Analysis        │
│  • Volume Analysis              • ML Models                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   SIGNAL GENERATION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  • Trading Signals              • Risk Scoring              │
│  • Confidence Levels            • Alert Triggers            │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACTION LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  • Discord/Telegram Alerts      • Dashboard Updates         │
│  • Database Storage             • Automated Trading (opt)   │
└─────────────────────────────────────────────────────────────┘
```

## Complete Implementation: Python

### Project Structure

```
polymarket-onchain/
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration management
├── data/
│   ├── __init__.py
│   ├── rpc_client.py         # RPC connection handling
│   ├── graph_client.py       # The Graph queries
│   └── event_listener.py     # Event monitoring
├── analysis/
│   ├── __init__.py
│   ├── whale_tracker.py      # Whale detection
│   ├── clustering.py         # Address clustering
│   └── signals.py            # Signal generation
├── models/
│   ├── __init__.py
│   ├── events.py             # Event data models
│   └── whales.py             # Whale profile models
├── alerts/
│   ├── __init__.py
│   ├── discord.py            # Discord webhooks
│   └── telegram.py           # Telegram bot
├── storage/
│   ├── __init__.py
│   ├── database.py           # SQLite/PostgreSQL
│   └── cache.py              # Redis caching
├── main.py                   # Main application
├── requirements.txt
└── .env
```

### requirements.txt

```txt
web3>=6.15.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
networkx>=3.2
redis>=5.0.0
sqlalchemy>=2.0.0
discord-webhook>=1.3.0
python-telegram-bot>=20.7
```

### config/settings.py

```python
"""Configuration management."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RPCConfig:
    """RPC provider configuration."""
    alchemy_key: str = os.getenv('ALCHEMY_API_KEY')
    infura_key: str = os.getenv('INFURA_API_KEY')
    quicknode_url: str = os.getenv('QUICKNODE_URL')

    @property
    def alchemy_ws_url(self) -> str:
        return f"wss://polygon-mainnet.g.alchemy.com/v2/{self.alchemy_key}"

    @property
    def alchemy_http_url(self) -> str:
        return f"https://polygon-mainnet.g.alchemy.com/v2/{self.alchemy_key}"

@dataclass
class ContractConfig:
    """Smart contract addresses."""
    ctf_exchange: str = '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E'
    conditional_tokens: str = '0x4d97dcd97ec945f40cf65f87097ace5ea0476045'
    usdc: str = '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'

@dataclass
class AnalysisConfig:
    """Analysis thresholds and parameters."""
    whale_threshold_usd: float = 10000
    min_whale_trades: int = 5
    clustering_min_samples: int = 3
    signal_confidence_threshold: float = 0.7

@dataclass
class AlertConfig:
    """Alert configuration."""
    discord_webhook: str = os.getenv('DISCORD_WEBHOOK_URL')
    telegram_token: str = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID')

@dataclass
class Settings:
    """Global settings."""
    rpc: RPCConfig = RPCConfig()
    contracts: ContractConfig = ContractConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    alerts: AlertConfig = AlertConfig()
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'

settings = Settings()
```

### models/events.py

```python
"""Event data models."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class OrderFilledEvent:
    """OrderFilled event data."""
    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount_filled: float  # In USDC
    taker_amount_filled: float  # In USDC
    fee: float
    block_number: int
    transaction_hash: str
    timestamp: datetime
    log_index: int

    @property
    def total_value(self) -> float:
        """Total trade value in USDC."""
        return self.maker_amount_filled + self.taker_amount_filled

    @property
    def effective_price(self) -> float:
        """Effective price (probability) of outcome."""
        total = self.total_value
        return self.taker_amount_filled / total if total > 0 else 0

    @property
    def is_whale_trade(self) -> bool:
        """Check if trade size exceeds whale threshold."""
        from config.settings import settings
        return self.total_value >= settings.analysis.whale_threshold_usd

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'order_hash': self.order_hash,
            'maker': self.maker,
            'taker': self.taker,
            'maker_asset_id': self.maker_asset_id,
            'taker_asset_id': self.taker_asset_id,
            'maker_amount_filled': self.maker_amount_filled,
            'taker_amount_filled': self.taker_amount_filled,
            'fee': self.fee,
            'total_value': self.total_value,
            'effective_price': self.effective_price,
            'block_number': self.block_number,
            'transaction_hash': self.transaction_hash,
            'timestamp': self.timestamp.isoformat(),
            'log_index': self.log_index
        }
```

### data/event_listener.py

```python
"""Real-time event listener with reconnection."""
import asyncio
import logging
from typing import Callable, List
from web3 import AsyncWeb3, WebSocketProvider
from web3.exceptions import Web3Exception
from datetime import datetime

from config.settings import settings
from models.events import OrderFilledEvent

logger = logging.getLogger(__name__)

class EventListener:
    """Listen to Polymarket CTF Exchange events."""

    def __init__(self, handlers: List[Callable]):
        self.handlers = handlers
        self.w3: Optional[AsyncWeb3] = None
        self.subscription_id: Optional[str] = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.seen_events = set()

    async def start(self):
        """Start listening to events."""
        await self._connect()

    async def _connect(self):
        """Establish WebSocket connection."""
        try:
            self.w3 = AsyncWeb3(WebSocketProvider(settings.rpc.alchemy_ws_url))

            # Test connection
            block_number = await self.w3.eth.block_number
            logger.info(f"Connected to Polygon. Current block: {block_number}")

            # Reset reconnect counter
            self.reconnect_attempts = 0

            # Start listening
            await self._subscribe()

        except Exception as e:
            logger.error(f"Connection error: {e}")
            await self._reconnect()

    async def _subscribe(self):
        """Subscribe to OrderFilled events."""
        try:
            # Subscribe to logs from CTF Exchange
            self.subscription_id = await self.w3.eth.subscribe('logs', {
                'address': settings.contracts.ctf_exchange
            })

            logger.info(f"Subscribed to events: {self.subscription_id}")

            # Process events
            async for response in self.w3.socket.process_subscriptions():
                if response.get('result'):
                    await self._process_log(response['result'])

        except Exception as e:
            logger.error(f"Subscription error: {e}")
            await self._reconnect()

    async def _process_log(self, log: dict):
        """Process incoming log entry."""
        try:
            # Create unique event ID
            event_id = f"{log['transactionHash']}-{log['logIndex']}"

            # Skip duplicates
            if event_id in self.seen_events:
                return

            self.seen_events.add(event_id)

            # Keep only recent events in memory (last 10k)
            if len(self.seen_events) > 10000:
                self.seen_events = set(list(self.seen_events)[-10000:])

            # Decode event
            event = await self._decode_event(log)

            if event:
                # Call all handlers
                for handler in self.handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")

        except Exception as e:
            logger.error(f"Log processing error: {e}")

    async def _decode_event(self, log: dict) -> Optional[OrderFilledEvent]:
        """Decode OrderFilled event from log."""
        try:
            # Get event signature (topic[0])
            event_sig = log['topics'][0]

            # OrderFilled signature
            order_filled_sig = self.w3.keccak(
                text='OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)'
            )

            if event_sig.hex() == order_filled_sig.hex():
                # Decode data (simplified - in production, use contract ABI)
                # This is just an example, actual decoding requires the ABI

                # Get block timestamp
                block = await self.w3.eth.get_block(log['blockNumber'])

                return OrderFilledEvent(
                    order_hash=log['topics'][1].hex(),
                    maker=f"0x{log['topics'][2].hex()[-40:]}",
                    taker=f"0x{log['topics'][3].hex()[-40:]}",
                    maker_asset_id=0,  # Decode from data
                    taker_asset_id=0,  # Decode from data
                    maker_amount_filled=0,  # Decode from data
                    taker_amount_filled=0,  # Decode from data
                    fee=0,  # Decode from data
                    block_number=log['blockNumber'],
                    transaction_hash=log['transactionHash'].hex(),
                    timestamp=datetime.fromtimestamp(block['timestamp']),
                    log_index=log['logIndex']
                )

            return None

        except Exception as e:
            logger.error(f"Decode error: {e}")
            return None

    async def _reconnect(self):
        """Reconnect after error."""
        self.reconnect_attempts += 1

        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        # Exponential backoff
        delay = min(2 ** self.reconnect_attempts, 60)
        logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")

        await asyncio.sleep(delay)
        await self._connect()
```

### analysis/whale_tracker.py

```python
"""Whale tracking and analysis."""
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd

from models.events import OrderFilledEvent
from config.settings import settings

logger = logging.getLogger(__name__)

class WhaleProfile:
    """Profile for a whale wallet."""

    def __init__(self, address: str):
        self.address = address
        self.trades: List[OrderFilledEvent] = []
        self.total_volume = 0.0
        self.first_seen: Optional[datetime] = None
        self.last_seen: Optional[datetime] = None

    def add_trade(self, event: OrderFilledEvent):
        """Add trade to profile."""
        self.trades.append(event)
        self.total_volume += event.total_value

        if not self.first_seen or event.timestamp < self.first_seen:
            self.first_seen = event.timestamp

        if not self.last_seen or event.timestamp > self.last_seen:
            self.last_seen = event.timestamp

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def avg_trade_size(self) -> float:
        return self.total_volume / self.num_trades if self.num_trades > 0 else 0

    @property
    def is_active(self) -> bool:
        """Check if traded in last 24 hours."""
        if not self.last_seen:
            return False
        return (datetime.now() - self.last_seen) < timedelta(hours=24)

    def to_dict(self) -> dict:
        return {
            'address': self.address,
            'num_trades': self.num_trades,
            'total_volume': self.total_volume,
            'avg_trade_size': self.avg_trade_size,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'is_active': self.is_active
        }

class WhaleTracker:
    """Track and analyze whale activity."""

    def __init__(self):
        self.whales: Dict[str, WhaleProfile] = {}

    async def process_event(self, event: OrderFilledEvent):
        """Process trade event for whale analysis."""
        # Check if whale trade
        if not event.is_whale_trade:
            return None

        # Update profiles for both maker and taker
        for address in [event.maker, event.taker]:
            if address not in self.whales:
                self.whales[address] = WhaleProfile(address)

            self.whales[address].add_trade(event)

        # Check if this triggers any alerts
        return await self._check_alerts(event)

    async def _check_alerts(self, event: OrderFilledEvent) -> Optional[dict]:
        """Check if event triggers whale alert."""
        maker_profile = self.whales.get(event.maker)
        taker_profile = self.whales.get(event.taker)

        alerts = []

        # New whale detection
        if maker_profile and maker_profile.num_trades == settings.analysis.min_whale_trades:
            alerts.append({
                'type': 'NEW_WHALE',
                'address': event.maker,
                'profile': maker_profile.to_dict()
            })

        # Large single trade
        if event.total_value >= settings.analysis.whale_threshold_usd * 2:
            alerts.append({
                'type': 'LARGE_TRADE',
                'event': event.to_dict(),
                'maker_profile': maker_profile.to_dict() if maker_profile else None,
                'taker_profile': taker_profile.to_dict() if taker_profile else None
            })

        return alerts if alerts else None

    def get_top_whales(self, n: int = 10, sort_by: str = 'total_volume') -> List[WhaleProfile]:
        """Get top N whales by specified metric."""
        sorted_whales = sorted(
            self.whales.values(),
            key=lambda w: getattr(w, sort_by),
            reverse=True
        )
        return sorted_whales[:n]

    def get_active_whales(self) -> List[WhaleProfile]:
        """Get whales active in last 24 hours."""
        return [w for w in self.whales.values() if w.is_active]
```

### alerts/discord.py

```python
"""Discord webhook alerts."""
import aiohttp
import logging
from typing import Dict, List
from config.settings import settings

logger = logging.getLogger(__name__)

class DiscordAlert:
    """Send alerts to Discord via webhook."""

    def __init__(self):
        self.webhook_url = settings.alerts.discord_webhook

    async def send_whale_alert(self, alerts: List[dict]):
        """Send whale activity alert."""
        if not self.webhook_url:
            logger.warning("Discord webhook not configured")
            return

        for alert in alerts:
            embed = self._create_embed(alert)

            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(self.webhook_url, json={
                        'content': '🐋 **WHALE ALERT**',
                        'embeds': [embed]
                    })
                    logger.info(f"Sent Discord alert: {alert['type']}")

            except Exception as e:
                logger.error(f"Discord alert error: {e}")

    def _create_embed(self, alert: dict) -> dict:
        """Create Discord embed from alert data."""
        if alert['type'] == 'LARGE_TRADE':
            event = alert['event']
            return {
                'title': f"Large Trade: ${event['total_value']:,.0f} USDC",
                'description': f"Market Asset: {event['maker_asset_id']}",
                'color': 0x00ff00,  # Green
                'fields': [
                    {
                        'name': 'Maker',
                        'value': f"`{event['maker'][:8]}...{event['maker'][-6:]}`",
                        'inline': True
                    },
                    {
                        'name': 'Taker',
                        'value': f"`{event['taker'][:8]}...{event['taker'][-6:]}`",
                        'inline': True
                    },
                    {
                        'name': 'Price',
                        'value': f"{event['effective_price']:.2%}",
                        'inline': True
                    },
                    {
                        'name': 'Transaction',
                        'value': f"[View on PolygonScan](https://polygonscan.com/tx/{event['transaction_hash']})"
                    }
                ],
                'timestamp': event['timestamp']
            }

        elif alert['type'] == 'NEW_WHALE':
            profile = alert['profile']
            return {
                'title': '🆕 New Whale Detected',
                'description': f"Address: `{profile['address'][:10]}...{profile['address'][-8:]}`",
                'color': 0x0000ff,  # Blue
                'fields': [
                    {
                        'name': 'Total Volume',
                        'value': f"${profile['total_volume']:,.0f}",
                        'inline': True
                    },
                    {
                        'name': 'Trades',
                        'value': str(profile['num_trades']),
                        'inline': True
                    },
                    {
                        'name': 'Avg Trade Size',
                        'value': f"${profile['avg_trade_size']:,.0f}",
                        'inline': True
                    }
                ]
            }

        return {}
```

### main.py

```python
"""Main application entry point."""
import asyncio
import logging
from data.event_listener import EventListener
from analysis.whale_tracker import WhaleTracker
from alerts.discord import DiscordAlert
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Run the on-chain analysis system."""
    logger.info("Starting Polymarket on-chain analysis system")

    # Initialize components
    whale_tracker = WhaleTracker()
    discord_alert = DiscordAlert()

    # Event handler
    async def handle_event(event):
        """Process each event."""
        logger.info(f"Event: {event.transaction_hash} | Value: ${event.total_value:,.2f}")

        # Whale analysis
        alerts = await whale_tracker.process_event(event)

        if alerts:
            await discord_alert.send_whale_alert(alerts)

    # Start event listener
    listener = EventListener(handlers=[handle_event])

    try:
        await listener.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
```

## Complete Implementation: JavaScript/TypeScript

### package.json

```json
{
  "name": "polymarket-onchain",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "start": "node src/main.js",
    "dev": "nodemon src/main.js"
  },
  "dependencies": {
    "ethers": "^6.10.0",
    "axios": "^1.6.5",
    "dotenv": "^16.4.0",
    "discord.js": "^14.14.1",
    "node-cache": "^5.1.2",
    "better-sqlite3": "^9.3.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.3"
  }
}
```

### src/config.js

```javascript
import dotenv from 'dotenv';
dotenv.config();

export const config = {
  rpc: {
    alchemyKey: process.env.ALCHEMY_API_KEY,
    alchemyWs: `wss://polygon-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`,
    alchemyHttp: `https://polygon-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`
  },
  contracts: {
    ctfExchange: '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E',
    conditionalTokens: '0x4d97dcd97ec945f40cf65f87097ace5ea0476045',
    usdc: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'
  },
  analysis: {
    whaleThreshold: 10000,
    minWhaleTrades: 5
  },
  alerts: {
    discordWebhook: process.env.DISCORD_WEBHOOK_URL
  }
};
```

### src/models/Event.js

```javascript
export class OrderFilledEvent {
  constructor(data) {
    this.orderHash = data.orderHash;
    this.maker = data.maker;
    this.taker = data.taker;
    this.makerAssetId = data.makerAssetId;
    this.takerAssetId = data.takerAssetId;
    this.makerAmountFilled = data.makerAmountFilled;
    this.takerAmountFilled = data.takerAmountFilled;
    this.fee = data.fee;
    this.blockNumber = data.blockNumber;
    this.transactionHash = data.transactionHash;
    this.timestamp = data.timestamp;
  }

  get totalValue() {
    return this.makerAmountFilled + this.takerAmountFilled;
  }

  get effectivePrice() {
    const total = this.totalValue;
    return total > 0 ? this.takerAmountFilled / total : 0;
  }

  isWhaleTrade(threshold = 10000) {
    return this.totalValue >= threshold;
  }
}
```

### src/listeners/EventListener.js

```javascript
import { ethers } from 'ethers';
import { config } from '../config.js';
import { OrderFilledEvent } from '../models/Event.js';

export class EventListener {
  constructor(handlers = []) {
    this.handlers = handlers;
    this.provider = null;
    this.contract = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
  }

  async start() {
    await this.connect();
  }

  async connect() {
    try {
      this.provider = new ethers.WebSocketProvider(config.rpc.alchemyWs);

      const abi = [
        "event OrderFilled(bytes32 indexed orderHash, address indexed maker, address indexed taker, uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)"
      ];

      this.contract = new ethers.Contract(
        config.contracts.ctfExchange,
        abi,
        this.provider
      );

      // Test connection
      const blockNumber = await this.provider.getBlockNumber();
      console.log(`Connected to Polygon. Block: ${blockNumber}`);

      // Reset reconnect counter
      this.reconnectAttempts = 0;

      // Start listening
      this.subscribe();

    } catch (error) {
      console.error('Connection error:', error);
      this.reconnect();
    }
  }

  subscribe() {
    this.contract.on('OrderFilled', async (...args) => {
      try {
        const event = await this.parseEvent(...args);
        await this.handleEvent(event);
      } catch (error) {
        console.error('Event handling error:', error);
      }
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
  }

  async parseEvent(...args) {
    const event = args[args.length - 1];

    const [orderHash, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee] = args.slice(0, -1);

    const block = await event.getBlock();

    return new OrderFilledEvent({
      orderHash,
      maker,
      taker,
      makerAssetId: makerAssetId.toString(),
      takerAssetId: takerAssetId.toString(),
      makerAmountFilled: parseFloat(ethers.formatUnits(makerAmountFilled, 6)),
      takerAmountFilled: parseFloat(ethers.formatUnits(takerAmountFilled, 6)),
      fee: parseFloat(ethers.formatUnits(fee, 6)),
      blockNumber: event.log.blockNumber,
      transactionHash: event.log.transactionHash,
      timestamp: new Date(block.timestamp * 1000)
    });
  }

  async handleEvent(event) {
    for (const handler of this.handlers) {
      try {
        await handler(event);
      } catch (error) {
        console.error('Handler error:', error);
      }
    }
  }

  reconnect() {
    this.reconnectAttempts++;

    if (this.reconnectAttempts > this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }
}
```

### src/main.js

```javascript
import { EventListener } from './listeners/EventListener.js';
import { WhaleTracker } from './analysis/WhaleTracker.js';
import { DiscordAlert } from './alerts/DiscordAlert.js';

async function main() {
  console.log('Starting Polymarket on-chain analysis system');

  const whaleTracker = new WhaleTracker();
  const discordAlert = new DiscordAlert();

  const eventListener = new EventListener([
    async (event) => {
      console.log(`Event: ${event.transactionHash} | Value: $${event.totalValue.toFixed(2)}`);

      const alerts = await whaleTracker.processEvent(event);

      if (alerts && alerts.length > 0) {
        await discordAlert.sendWhaleAlert(alerts);
      }
    }
  ]);

  await eventListener.start();
}

main().catch(console.error);
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  onchain-analyzer:
    build: .
    environment:
      - ALCHEMY_API_KEY=${ALCHEMY_API_KEY}
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
    restart: unless-stopped
    volumes:
      - ./data:/app/data

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### Systemd Service

```ini
# /etc/systemd/system/polymarket-onchain.service
[Unit]
Description=Polymarket On-Chain Analysis
After=network.target

[Service]
Type=simple
User=polymarket
WorkingDirectory=/opt/polymarket-onchain
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/opt/polymarket-onchain/.env

[Install]
WantedBy=multi-user.target
```

## Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
events_processed = Counter('events_processed_total', 'Total events processed')
whale_trades = Counter('whale_trades_total', 'Total whale trades detected')
event_processing_time = Histogram('event_processing_seconds', 'Event processing time')
active_whales = Gauge('active_whales', 'Number of active whales')

# Start metrics server
start_http_server(9090)
```

### Health Check Endpoint

```javascript
import express from 'express';

const app = express();

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

app.listen(3000);
```

## References

- [The Graph: Polymarket Subgraph Tutorial](https://thegraph.com/docs/en/subgraphs/guides/polymarket/)
- [Polymarket CTF Exchange GitHub](https://github.com/Polymarket/ctf-exchange)
- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [Ethers.js Documentation](https://docs.ethers.org/)

---

**Version**: 1.0
**Last Updated**: 2026-02-04
