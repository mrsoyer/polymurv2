# Bitquery GraphQL API - Polymarket On-Chain Data

> Solution pour obtenir TOUS les holders (pas limité à 20) via données on-chain Polygon

---

## Overview

Bitquery fournit un accès GraphQL aux données on-chain de Polymarket sur Polygon. C'est la **seule solution** pour obtenir la liste complète des holders d'un market (pas limité à 20).

**Endpoint**: `https://graphql.bitquery.io/`

**Blockchain**: Polygon (chain_id: 137)

**Contract**: Polymarket CTF Exchange (`0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`)

---

## Authentication

### API Key

Inscription requise sur [Bitquery](https://bitquery.io/).

```bash
# Headers requis
X-API-KEY: your_bitquery_api_key
Content-Type: application/json
```

### Plans Tarifaires

| Plan | Requêtes/jour | Prix/mois | Use Case |
|------|---------------|-----------|----------|
| Free | 1,000 | Gratuit | Testing |
| Developer | 50,000 | 49 USD | Prototyping |
| Startup | 250,000 | 149 USD | Production (recommandé) |
| Professional | 1,000,000 | 399 USD | High volume |

**Budget 200-500 USD/mois**: Plan **Startup** (149 USD) + marge confortable.

---

## Queries GraphQL Principales

### 1. Obtenir TOUS les Holders d'un Token

⚠️ **C'est la query la plus critique pour l'algorithme**

```graphql
query GetAllHolders($token: String!, $limit: Int!, $offset: Int!) {
  ethereum(network: matic) {
    transfers(
      options: {limit: $limit, offset: $offset}
      currency: {is: $token}
      amount: {gt: 0}
    ) {
      receiver {
        address
      }
      amount
      currency {
        address
        symbol
      }
      block {
        timestamp {
          unixtime
        }
      }
    }
  }
}
```

**Variables**:
```json
{
  "token": "0x...",  // Conditional token address (YES ou NO)
  "limit": 100,
  "offset": 0
}
```

**Réponse**:
```json
{
  "data": {
    "ethereum": {
      "transfers": [
        {
          "receiver": {
            "address": "0x1234..."
          },
          "amount": 150.5,
          "currency": {
            "address": "0x...",
            "symbol": "YES"
          },
          "block": {
            "timestamp": {
              "unixtime": 1706832000
            }
          }
        }
      ]
    }
  }
}
```

**Pagination**: Utiliser `offset` pour récupérer tous les holders (pas de limite à 20).

---

### 2. Historique Complet des Trades d'un Wallet

```graphql
query GetWalletTrades($wallet: String!, $startDate: ISO8601DateTime!) {
  ethereum(network: matic) {
    dexTrades(
      options: {limit: 1000, desc: "block.timestamp.unixtime"}
      exchangeAddress: {is: "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"}
      txSender: {is: $wallet}
      date: {since: $startDate}
    ) {
      transaction {
        hash
      }
      block {
        timestamp {
          unixtime
        }
      }
      buyCurrency {
        address
        symbol
      }
      buyAmount
      sellCurrency {
        address
        symbol
      }
      sellAmount
      tradeAmount(in: USD)
    }
  }
}
```

**Variables**:
```json
{
  "wallet": "0x1234...",
  "startDate": "2025-10-01T00:00:00Z"
}
```

**Usage**: Calculer le ROI historique d'un trader.

---

### 3. Events OrderFilled (Détection Nouveaux Trades)

```graphql
query GetRecentOrderFilled($since: ISO8601DateTime!) {
  ethereum(network: matic) {
    arguments(
      smartContractAddress: {is: "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"}
      smartContractEvent: {is: "OrderFilled"}
      date: {since: $since}
      options: {limit: 100, desc: "block.timestamp.unixtime"}
    ) {
      block {
        timestamp {
          unixtime
        }
      }
      transaction {
        hash
      }
      argument {
        name
        value
      }
    }
  }
}
```

**Usage**: Détecter en temps réel quand un top trader place une position.

---

### 4. PositionSplit Events (Mint/Burn Tokens)

```graphql
query GetPositionSplits($conditionId: String!, $since: ISO8601DateTime!) {
  ethereum(network: matic) {
    arguments(
      smartContractAddress: {is: "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"}
      smartContractEvent: {is: "PositionSplit"}
      argument: {is: {name: "conditionId", value: $conditionId}}
      date: {since: $since}
      options: {limit: 100}
    ) {
      block {
        timestamp {
          unixtime
        }
      }
      transaction {
        hash
        txFrom {
          address
        }
      }
      argument {
        name
        value
      }
    }
  }
}
```

**Usage**: Tracker qui mint/burn des tokens conditionnels (positions ouvertes/fermées).

---

## Cas d'Usage pour l'Algorithme

### Phase 3: Obtenir TOUS les Holders (Pas 20)

```python
import requests

BITQUERY_ENDPOINT = "https://graphql.bitquery.io/"
BITQUERY_API_KEY = "your_api_key"

def get_all_holders(token_address):
    """
    Récupère TOUS les holders d'un token (pas limité à 20)
    """
    holders = []
    offset = 0
    batch_size = 100

    while True:
        query = """
        query($token: String!, $limit: Int!, $offset: Int!) {
          ethereum(network: matic) {
            transfers(
              options: {limit: $limit, offset: $offset}
              currency: {is: $token}
              amount: {gt: 0}
            ) {
              receiver { address }
              amount
            }
          }
        }
        """

        variables = {
            "token": token_address,
            "limit": batch_size,
            "offset": offset
        }

        response = requests.post(
            BITQUERY_ENDPOINT,
            json={"query": query, "variables": variables},
            headers={"X-API-KEY": BITQUERY_API_KEY}
        )

        data = response.json()
        transfers = data["data"]["ethereum"]["transfers"]

        if not transfers:
            break  # Plus de holders

        # Agréger par holder (un holder peut avoir multiple transfers)
        for transfer in transfers:
            address = transfer["receiver"]["address"]
            amount = float(transfer["amount"])

            # Ajouter ou mettre à jour
            existing = next((h for h in holders if h["address"] == address), None)
            if existing:
                existing["total"] += amount
            else:
                holders.append({"address": address, "total": amount})

        offset += batch_size

    return holders

# Obtenir TOUS les holders YES d'un market
yes_token = "0x..."
all_yes_holders = get_all_holders(yes_token)

print(f"Total holders YES: {len(all_yes_holders)}")  # Ex: 1247 holders
```

### Phase 3: Calculer ROI d'un Wallet

```python
def calculate_wallet_roi(wallet_address, days=90):
    """
    Calcule le ROI historique d'un wallet sur N jours
    """
    start_date = (datetime.now() - timedelta(days=days)).isoformat()

    query = """
    query($wallet: String!, $startDate: ISO8601DateTime!) {
      ethereum(network: matic) {
        dexTrades(
          exchangeAddress: {is: "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"}
          txSender: {is: $wallet}
          date: {since: $startDate}
        ) {
          buyCurrency { symbol }
          buyAmount
          sellCurrency { symbol }
          sellAmount
          tradeAmount(in: USD)
        }
      }
    }
    """

    variables = {
        "wallet": wallet_address,
        "startDate": start_date
    }

    response = requests.post(
        BITQUERY_ENDPOINT,
        json={"query": query, "variables": variables},
        headers={"X-API-KEY": BITQUERY_API_KEY}
    )

    trades = response.json()["data"]["ethereum"]["dexTrades"]

    # Calculer ROI
    total_invested = 0
    total_returns = 0

    for trade in trades:
        if trade["sellCurrency"]["symbol"] == "USDC":
            # Achat (vend USDC, achète outcome)
            total_invested += float(trade["sellAmount"])
        else:
            # Vente (vend outcome, reçoit USDC)
            total_returns += float(trade["buyAmount"])

    if total_invested == 0:
        return 0

    roi = ((total_returns - total_invested) / total_invested) * 100

    return {
        "roi": roi,
        "trades_count": len(trades),
        "total_invested": total_invested,
        "total_returns": total_returns
    }
```

---

## Optimisation Coûts

### Stratégie de Caching

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def get_all_holders_cached(token_address, ttl=300):
    """
    Cache les holders pendant 5 minutes (ttl=300)
    """
    cache_key = f"holders:{token_address}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fetch from Bitquery
    holders = get_all_holders(token_address)

    # Store in cache
    redis_client.setex(cache_key, ttl, json.dumps(holders))

    return holders
```

**Économie**: Cache 5 min = 288 requêtes/jour → 8,640 requêtes/mois par token
**Sans cache**: 1 requête/seconde = 2,592,000 requêtes/mois ❌ (hors budget)

---

## Rate Limits

| Plan | Requêtes/min | Requêtes/jour |
|------|--------------|---------------|
| Free | 10 | 1,000 |
| Developer | 60 | 50,000 |
| Startup | 120 | 250,000 |
| Professional | 300 | 1,000,000 |

**Budget 200-500 USD/mois**:
- Plan Startup (149 USD) = **120 req/min**, **250k req/jour**
- Avec cache 5 min = **28.8k holders queries/jour** (suffisant)

---

## Limitations Connues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Latence 1-3 sec** | Pas temps réel | Utiliser pour enrichissement batch, pas signaux immédiats |
| **Pagination manuelle** | Code complexe | Wrapper avec fonction récursive |
| **Coût élevé** | Budget | Cache agressif (5-15 min TTL) |
| **Historical data** | Pas d'instantané exact | Reconstruire positions via transfers |

---

## Comparaison: Bitquery vs Polymarket API

| Aspect | Bitquery GraphQL | Polymarket Data API |
|--------|------------------|---------------------|
| **Holders complets** | ✅ Tous les holders | ❌ Max 20 holders |
| **Latence** | 1-3 secondes | 200-500ms |
| **Coût** | 149+ USD/mois | Gratuit/50 USD |
| **Historique** | ✅ Complet on-chain | Limité |
| **Complexité** | GraphQL (moyen) | REST (simple) |

**Conclusion**: Bitquery est **ESSENTIEL** pour Phase 3 (enrichissement complet) et Phase 5 (monitoring post-achat tous holders).

---

## Alternatives

### Polygon RPC Direct

Si budget Bitquery dépassé, fallback sur Polygon RPC direct (voir [polygon-rpc.md](polygon-rpc.md)).

**Avantages**: Gratuit (avec Alchemy/Infura free tier)
**Inconvénients**: Parsing events manuel, plus complexe

---

## Ressources

- [Bitquery Docs](https://docs.bitquery.io/)
- [Polymarket CTF API Examples](https://docs.bitquery.io/docs/examples/polymarket-api/)
- [GraphQL Playground](https://graphql.bitquery.io/)

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - sym-web-research
**Projet**: Polymarket Trading Algorithm
