# Polygon RPC Direct - Fallback pour Données On-Chain

> Accès direct blockchain Polygon comme fallback si Bitquery down ou budget dépassé

---

## Overview

Accès direct aux smart contracts Polymarket sur Polygon via RPC nodes. Solution de **fallback** si Bitquery indisponible ou trop coûteux.

**Réseau**: Polygon (chain_id: 137)
**Contract CTF Exchange**: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`

---

## RPC Providers

### 1. Alchemy (Recommandé)

**Endpoint**: `https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY`

**Plans**:
| Plan | Requests/sec | Requests/mois | Prix |
|------|--------------|---------------|------|
| Free | 5 | 3M compute units | Gratuit |
| Growth | 25 | 15M compute units | 49 USD/mois |
| Scale | 100 | 100M compute units | 399 USD/mois |

**Inscription**: [alchemy.com](https://www.alchemy.com/)

---

### 2. Infura

**Endpoint**: `https://polygon-mainnet.infura.io/v3/YOUR_PROJECT_ID`

**Plans**:
| Plan | Requests/jour | Prix |
|------|---------------|------|
| Free | 100,000 | Gratuit |
| Developer | 500,000 | 50 USD/mois |
| Team | 2,000,000 | 225 USD/mois |

**Inscription**: [infura.io](https://infura.io/)

---

### 3. QuickNode

**Endpoint**: Custom endpoint fourni

**Plans**: À partir de 49 USD/mois

---

## Configuration Web3.py

```python
from web3 import Web3

# Connect to Polygon via Alchemy
ALCHEMY_URL = "https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY"
w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))

# Verify connection
print(f"Connected: {w3.is_connected()}")
print(f"Latest block: {w3.eth.block_number}")

# Polymarket CTF Exchange contract
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
```

---

## Contract ABI Minimal

```python
# ABI minimal pour les events OrderFilled et PositionSplit
CTF_EXCHANGE_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "orderHash", "type": "bytes32"},
            {"indexed": True, "name": "maker", "type": "address"},
            {"indexed": False, "name": "taker", "type": "address"},
            {"indexed": False, "name": "makerAssetId", "type": "uint256"},
            {"indexed": False, "name": "takerAssetId", "type": "uint256"},
            {"indexed": False, "name": "makerAmountFilled", "type": "uint256"},
            {"indexed": False, "name": "takerAmountFilled", "type": "uint256"},
            {"indexed": False, "name": "fee", "type": "uint256"}
        ],
        "name": "OrderFilled",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "stakeholder", "type": "address"},
            {"indexed": True, "name": "collateralToken", "type": "address"},
            {"indexed": True, "name": "conditionId", "type": "bytes32"},
            {"indexed": False, "name": "partition", "type": "uint256[]"},
            {"indexed": False, "name": "amount", "type": "uint256"}
        ],
        "name": "PositionSplit",
        "type": "event"
    }
]

contract = w3.eth.contract(
    address=Web3.to_checksum_address(CTF_EXCHANGE_ADDRESS),
    abi=CTF_EXCHANGE_ABI
)
```

---

## Queries Principales

### 1. Obtenir Tous les Holders d'un Token

⚠️ **Complexe**: Nécessite parsing de tous les events Transfer du token.

```python
def get_all_holders_via_rpc(token_address, from_block=0):
    """
    Parse Transfer events pour reconstruire liste holders
    WARNING: Très lent pour gros historiques
    """
    token_address = Web3.to_checksum_address(token_address)

    # ABI ERC20 Transfer event
    transfer_event_signature = w3.keccak(text="Transfer(address,address,uint256)")

    # Get logs
    latest_block = w3.eth.block_number
    holders = {}

    # Scan par batches de 10k blocks (limite RPC)
    batch_size = 10000

    for start_block in range(from_block, latest_block, batch_size):
        end_block = min(start_block + batch_size, latest_block)

        logs = w3.eth.get_logs({
            "fromBlock": start_block,
            "toBlock": end_block,
            "address": token_address,
            "topics": [transfer_event_signature.hex()]
        })

        for log in logs:
            # Decode Transfer event
            from_address = "0x" + log["topics"][1].hex()[-40:]
            to_address = "0x" + log["topics"][2].hex()[-40:]
            amount = int(log["data"].hex(), 16)

            # Update balances
            if from_address in holders:
                holders[from_address] -= amount
            if to_address in holders:
                holders[to_address] += amount
            else:
                holders[to_address] = amount

    # Filter holders with balance > 0
    active_holders = {
        addr: balance
        for addr, balance in holders.items()
        if balance > 0
    }

    return active_holders

# Usage (WARNING: très lent)
yes_token = "0x..."
holders = get_all_holders_via_rpc(yes_token, from_block=50000000)
print(f"Total holders: {len(holders)}")
```

**⚠️ PROBLÈME**: Scan complet blockchain = **très lent** (minutes/heures pour gros tokens).

**Solution**: Utiliser Bitquery ou cache holders périodiquement.

---

### 2. Monitor OrderFilled Events (Temps Réel)

```python
def monitor_order_filled_events(callback):
    """
    Écoute en temps réel les événements OrderFilled
    """
    event_filter = contract.events.OrderFilled.create_filter(fromBlock='latest')

    while True:
        for event in event_filter.get_new_entries():
            # Parse event
            order_data = {
                "maker": event["args"]["maker"],
                "taker": event["args"]["taker"],
                "makerAmountFilled": event["args"]["makerAmountFilled"],
                "takerAmountFilled": event["args"]["takerAmountFilled"],
                "fee": event["args"]["fee"],
                "block": event["blockNumber"],
                "tx_hash": event["transactionHash"].hex()
            }

            # Call user callback
            callback(order_data)

        time.sleep(2)  # Poll every 2 seconds

# Usage
def on_order_filled(order):
    print(f"Order filled by {order['maker']}")
    # Vérifier si maker est un top trader
    if order['maker'] in top_traders:
        alert_top_trader_trade(order)

monitor_order_filled_events(on_order_filled)
```

---

### 3. Récupérer Historique Trades d'un Wallet

```python
def get_wallet_trade_history(wallet_address, from_block=0):
    """
    Récupère tous les trades d'un wallet via OrderFilled events
    """
    wallet_address = Web3.to_checksum_address(wallet_address)
    latest_block = w3.eth.block_number

    trades = []
    batch_size = 10000

    for start_block in range(from_block, latest_block, batch_size):
        end_block = min(start_block + batch_size, latest_block)

        # Get events où wallet est maker OU taker
        logs_maker = w3.eth.get_logs({
            "fromBlock": start_block,
            "toBlock": end_block,
            "address": CTF_EXCHANGE_ADDRESS,
            "topics": [
                w3.keccak(text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)").hex(),
                None,  # orderHash (any)
                w3.to_bytes(hexstr=wallet_address).rjust(32, b'\0').hex()  # maker = wallet
            ]
        })

        # Parse logs
        for log in logs_maker:
            trades.append({
                "type": "maker",
                "block": log["blockNumber"],
                "tx_hash": log["transactionHash"].hex()
                # ... decode full event data
            })

    return trades
```

---

## Cas d'Usage pour l'Algorithme

### Fallback si Bitquery Down

```python
class HolderDataProvider:
    def __init__(self):
        self.bitquery_available = True

    def get_all_holders(self, token_address):
        if self.bitquery_available:
            try:
                return get_all_holders_bitquery(token_address)
            except Exception as e:
                print(f"Bitquery error: {e}, falling back to RPC")
                self.bitquery_available = False

        # Fallback to RPC
        return get_all_holders_via_rpc(token_address, from_block=recent_block())

def recent_block():
    """
    Au lieu de scanner depuis block 0, scanner seulement derniers 7 jours
    """
    blocks_per_day = 43200  # ~2 sec/block sur Polygon
    return w3.eth.block_number - (7 * blocks_per_day)
```

### Monitor Top Traders Temps Réel

```python
def monitor_top_traders(top_trader_addresses):
    """
    Écoute OrderFilled pour détecter quand top traders tradent
    """
    event_filter = contract.events.OrderFilled.create_filter(fromBlock='latest')

    while True:
        for event in event_filter.get_new_entries():
            maker = event["args"]["maker"]
            taker = event["args"]["taker"]

            # Check si un top trader
            if maker in top_trader_addresses or taker in top_trader_addresses:
                trader = maker if maker in top_trader_addresses else taker

                # Fetch position details
                position = get_position_from_event(event)

                # Trigger signal
                on_top_trader_position(trader, position)

        time.sleep(2)
```

---

## Rate Limits

### Alchemy Free Tier
- **5 requests/sec**
- **3M compute units/mois** (~300k requests)

**Compute Units**:
| Call | Units |
|------|-------|
| eth_blockNumber | 10 |
| eth_getLogs | 75 |
| eth_call | 26 |

**Estimation**: 1 scan holders = 100 batches × 75 units = **7,500 units**

**Budget Free**: 3M units / 7500 = **400 scans holders/mois** ⚠️ Très limité

---

## Best Practices

1. **Cache agressif**: Ne scan que nouvelles positions depuis dernier check
2. **Recent blocks only**: Scanner derniers 7-30 jours, pas tout l'historique
3. **Batch requests**: Grouper multiple getLogs en une requête
4. **Fallback uniquement**: Utiliser RPC comme backup, pas primary
5. **Monitor quotas**: Logger compute units consommés

---

## Limitations Connues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Scan très lent** | Minutes pour un token | Scanner seulement blocks récents |
| **10k blocks/query max** | Multiples requêtes | Batching automatique |
| **Rate limits stricts** | 5 req/sec sur free tier | Queue avec throttling |
| **Compute units** | Budget vite épuisé | Cache + Bitquery primary |

---

## Comparaison: RPC vs Bitquery

| Aspect | Polygon RPC Direct | Bitquery GraphQL |
|--------|-------------------|------------------|
| **Coût** | Gratuit (free tier) | 149 USD/mois |
| **Latence holders** | Minutes/heures | 1-3 secondes |
| **Complexité** | Élevée (parsing events) | Faible (GraphQL) |
| **Fiabilité** | Dépend RPC provider | Service dédié |
| **Use case** | Fallback uniquement | Primary source |

**Recommandation**: Utiliser Bitquery comme source principale, RPC comme fallback d'urgence uniquement.

---

## Ressources

- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [Alchemy Docs](https://docs.alchemy.com/)
- [Polygon RPC Endpoints](https://wiki.polygon.technology/docs/develop/network-details/network/)
- [Ethers.js (alternative)](https://docs.ethers.org/)

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - sym-web-research
**Projet**: Polymarket Trading Algorithm
