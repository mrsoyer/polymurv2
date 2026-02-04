# On-Chain Data Sources & APIs

## Overview

Multiple approaches exist for accessing Polymarket's blockchain data on Polygon. This guide compares options, provides implementation patterns, and recommends optimal strategies.

## Access Methods Comparison

| Method | Latency | Cost | Complexity | Best For |
|--------|---------|------|------------|----------|
| **Direct RPC** | <100ms | Free-$200/mo | Low | Real-time events |
| **The Graph** | 100-500ms | Free-$99/mo | Medium | Complex queries |
| **PolygonScan API** | 1-2s | Free | Low | Historical lookup |
| **Dune Analytics** | 5-30s | Free-$99/mo | Low | Analysis dashboards |
| **Bitquery** | 200-1000ms | Free-$399/mo | Medium | Advanced GraphQL |
| **Alchemy Webhooks** | <100ms | Free-$499/mo | Low | Custom alerts |

## 1. Direct Polygon RPC Access

### Overview
Query Polygon blockchain directly through RPC providers for real-time, low-latency access.

### Providers

#### Alchemy (Recommended)
- **Free Tier**: 300M compute units/month
- **Paid**: $49-$499/month
- **Features**: Enhanced APIs, webhooks, NFT APIs
- **Endpoint**: `https://polygon-mainnet.g.alchemy.com/v2/YOUR-API-KEY`
- **WebSocket**: `wss://polygon-mainnet.g.alchemy.com/v2/YOUR-API-KEY`

#### Infura
- **Free Tier**: 100k requests/day
- **Paid**: $50-$225/month
- **Endpoint**: `https://polygon-mainnet.infura.io/v3/YOUR-PROJECT-ID`

#### QuickNode
- **Free Trial**: 7 days
- **Paid**: $9-$299/month
- **Features**: 99.9% uptime, global coverage
- **Endpoint**: `https://YOUR-ENDPOINT.quiknode.pro/YOUR-TOKEN/`

### Use Cases
- Real-time event monitoring
- Transaction simulation
- Block-by-block processing
- Low-latency trading signals

### Implementation Example (ethers.js)

```javascript
import { ethers } from 'ethers';

// Connect to Polygon via Alchemy
const provider = new ethers.JsonRpcProvider(
  'https://polygon-mainnet.g.alchemy.com/v2/YOUR-API-KEY'
);

// Get current block
const blockNumber = await provider.getBlockNumber();
console.log(`Current block: ${blockNumber}`);

// Monitor new blocks
provider.on('block', (blockNumber) => {
  console.log(`New block: ${blockNumber}`);
});
```

### Rate Limits
- **Alchemy Free**: ~25 req/sec burst, ~5 req/sec sustained
- **Infura Free**: ~10 req/sec
- **QuickNode**: Varies by plan, typically 25-unlimited

## 2. The Graph Protocol

### Overview
Decentralized indexing protocol providing GraphQL APIs for blockchain data. Pre-indexes smart contract events for fast complex queries.

### Polymarket Subgraphs

#### Official Polymarket Subgraph
- **Endpoint**: `https://gateway.thegraph.com/api/{API-KEY}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp`
- **Coverage**: All Polymarket markets, trades, redemptions
- **Update Frequency**: Near real-time (< 1 minute lag)

#### Getting Started
1. Visit [thegraph.com/studio](https://thegraph.com/studio)
2. Connect wallet
3. Generate API key at [thegraph.com/studio/apikeys/](https://thegraph.com/studio/apikeys/)
4. Free tier: 100k queries/month

### Example Query

```graphql
query TopTraders {
  redemptions(
    orderBy: payout,
    orderDirection: desc,
    first: 10
  ) {
    id
    redeemer
    payout
    timestamp
    collateralAmount
  }
}
```

### Node.js Implementation

```javascript
import axios from 'axios';

const SUBGRAPH_URL = 'https://gateway.thegraph.com/api/YOUR-API-KEY/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp';

async function queryTopTraders() {
  const query = `
    {
      redemptions(orderBy: payout, orderDirection: desc, first: 10) {
        redeemer
        payout
        timestamp
      }
    }
  `;

  const response = await axios.post(SUBGRAPH_URL, { query });
  return response.data.data.redemptions;
}
```

### Pricing
- **Free**: 100k queries/month
- **Growth**: $100/month for 1M queries
- **Enterprise**: Custom pricing

## 3. Dune Analytics

### Overview
SQL-based blockchain analytics platform with pre-built dashboards and community queries.

### Key Polymarket Dashboards

1. **Polymarket Activity & Volume**: [dune.com/filarm/polymarket-activity](https://dune.com/filarm/polymarket-activity)
   - Monthly bets and payouts
   - Market creation trends
   - User growth metrics

2. **Polymarket on Polygon**: [dune.com/petertherock/polymarket-on-polygon](https://dune.com/petertherock/polymarket-on-polygon)
   - Transaction volume breakdown
   - Gas usage analysis
   - Active users

3. **Polymarket CLOB Stats**: [dune.com/lifewillbeokay/polymarket-clob-stats](https://dune.com/lifewillbeokay/polymarket-clob-stats)
   - Order book depth
   - Maker/taker ratios
   - Fee analysis

### SQL Query Example

```sql
SELECT
  DATE_TRUNC('day', block_time) AS date,
  COUNT(*) AS num_trades,
  SUM(outcome_tokens_bought) AS volume,
  COUNT(DISTINCT buyer) AS unique_traders
FROM polygon.ctf_exchange_trades
WHERE contract_address = 0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e
  AND block_time >= NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1 DESC;
```

### Access Methods
- **Web Interface**: Interactive query builder
- **API**: Export results as JSON/CSV
- **Embeds**: Embed charts in applications

### Pricing
- **Free**: Public dashboards and queries
- **Premium**: $39/month (priority execution, private queries)
- **Plus**: $99/month (API access, unlimited refreshes)

## 4. PolygonScan API

### Overview
Block explorer API for Polygon transactions, addresses, and contracts.

### Key Endpoints

#### Get Event Logs
```
GET https://api.polygonscan.com/api
?module=logs
&action=getLogs
&fromBlock=5000000
&toBlock=6000000
&address=0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e
&topic0=0x...  # Event signature hash
&apikey=YOUR_API_KEY
```

#### Get Transaction List by Address
```
GET https://api.polygonscan.com/api
?module=account
&action=txlist
&address=0x...
&startblock=0
&endblock=99999999
&sort=desc
&apikey=YOUR_API_KEY
```

### Python Implementation

```python
import requests

POLYGONSCAN_API = 'https://api.polygonscan.com/api'
API_KEY = 'YOUR_API_KEY'

def get_event_logs(contract_address, event_signature, from_block, to_block):
    params = {
        'module': 'logs',
        'action': 'getLogs',
        'address': contract_address,
        'topic0': event_signature,
        'fromBlock': from_block,
        'toBlock': to_block,
        'apikey': API_KEY
    }

    response = requests.get(POLYGONSCAN_API, params=params)
    return response.json()['result']

# Example: Get OrderFilled events from CTF Exchange
logs = get_event_logs(
    contract_address='0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',
    event_signature='0x...',  # Keccak256 of OrderFilled event
    from_block=50000000,
    to_block='latest'
)
```

### Rate Limits
- **Free**: 5 requests/second
- **Pro**: No rate limit, $99-$599/month

## 5. Bitquery

### Overview
Advanced GraphQL API for blockchain data with filtering, aggregation, and streaming capabilities.

### Features
- Real-time WebSocket subscriptions
- Complex filtering and aggregations
- Cross-chain queries
- Historical data back to genesis

### Example Query

```graphql
query PolymarketTrades {
  ethereum(network: matic) {
    smartContractEvents(
      smartContractAddress: {is: "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"}
      smartContractEvent: {is: "OrderFilled"}
      options: {limit: 100, desc: "block.timestamp.time"}
    ) {
      block {
        timestamp {
          time(format: "%Y-%m-%d %H:%M:%S")
        }
        height
      }
      transaction {
        hash
      }
      arguments {
        argument
        value
      }
    }
  }
}
```

### Pricing
- **Free**: 10k API points/month (~1k queries)
- **Developer**: $49/month (100k points)
- **Startup**: $149/month (500k points)
- **Business**: $399/month (2M points)

## 6. Alchemy Webhooks

### Overview
Custom webhook service for real-time notifications on specific blockchain events.

### Setup Process

1. **Create Webhook** at [dashboard.alchemy.com](https://dashboard.alchemy.com)
2. **Configure GraphQL Query**:
```graphql
{
  block {
    transactions(filter: {
      to: "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
    }) {
      hash
      from
      to
      value
      logs {
        topics
        data
      }
    }
  }
}
```
3. **Set Webhook URL**: Your server endpoint
4. **Test**: Use webhook.site for testing

### Webhook Payload

```json
{
  "webhookId": "wh_...",
  "id": "...",
  "createdAt": "2026-02-04T10:30:00.000Z",
  "type": "GRAPHQL",
  "event": {
    "data": {
      "block": {
        "transactions": [...]
      }
    }
  }
}
```

### Node.js Handler

```javascript
import express from 'express';
const app = express();

app.post('/webhook', express.json(), (req, res) => {
  const { event } = req.body;
  const transactions = event.data.block.transactions;

  // Process transactions
  transactions.forEach(tx => {
    console.log(`New trade: ${tx.hash}`);
    // Decode logs, extract trade data, send alerts
  });

  res.sendStatus(200);
});

app.listen(3000);
```

## Recommendations by Use Case

### Real-Time Trading Signals
**Best**: Alchemy Webhooks + Direct RPC
- **Why**: <100ms latency, custom filtering
- **Cost**: $49-99/month

### Historical Analysis
**Best**: Dune Analytics + The Graph
- **Why**: Pre-indexed data, SQL/GraphQL flexibility
- **Cost**: Free for public data

### Budget-Constrained
**Best**: PolygonScan API + Public RPC
- **Why**: Free tier sufficient for moderate usage
- **Cost**: $0

### Enterprise/High-Volume
**Best**: Bitquery + Alchemy
- **Why**: Unlimited scaling, advanced features
- **Cost**: $500-1000/month

## Performance Optimization

### Caching Strategy
```javascript
import NodeCache from 'node-cache';
const cache = new NodeCache({ stdTTL: 60 }); // 60 second cache

async function getCachedMarketData(marketId) {
  const cached = cache.get(marketId);
  if (cached) return cached;

  const data = await fetchFromGraph(marketId);
  cache.set(marketId, data);
  return data;
}
```

### Rate Limit Handling
```javascript
import pThrottle from 'p-throttle';

// Limit to 5 requests per second
const throttle = pThrottle({
  limit: 5,
  interval: 1000
});

const throttledRequest = throttle(async (url) => {
  return await fetch(url);
});
```

## References

### Official Documentation
- [PolygonScan API Docs](https://docs.polygonscan.com/api-endpoints/logs)
- [The Graph: Polymarket Guide](https://thegraph.com/docs/en/subgraphs/guides/polymarket/)
- [Alchemy Custom Webhooks](https://www.alchemy.com/docs/how-to-use-custom-webhooks-for-web3-data-ingestion)
- [Bitquery Polymarket API](https://docs.bitquery.io/docs/examples/polymarket-api/)

### Community Resources
- [Dune: Polymarket Activity Dashboard](https://dune.com/filarm/polymarket-activity)
- [Dune: Prediction Markets Overview](https://dune.com/datadashboards/prediction-markets)

---

**Version**: 1.0
**Last Updated**: 2026-02-04
