# Polymarket Data API

> Documentation de l'API Data pour récupérer positions, trades et activité utilisateur

---

## Overview

L'API Data de Polymarket fournit l'accès aux données historiques et en temps réel des utilisateurs, positions et transactions.

**Base URL**: `https://data-api.polymarket.com`

---

## Authentication

### API Keys
```bash
# Headers requis
X-API-KEY: your_api_key
Content-Type: application/json
```

### Rate Limits
- **Non-authentifié**: 100 requests/minute
- **Authentifié**: 1000 requests/minute

---

## Endpoints Principaux

### 1. GET /positions

Récupère les positions d'un utilisateur.

**Paramètres**:
```json
{
  "user": "0x...",         // Adresse Ethereum (optionnel)
  "market": "string",      // Market ID (optionnel)
  "status": "active|closed" // Filtrer par statut
}
```

**Réponse**:
```json
{
  "positions": [
    {
      "id": "pos_123",
      "market_id": "market_456",
      "user_address": "0x...",
      "side": "YES|NO",
      "size": 100.5,
      "avg_price": 0.65,
      "current_price": 0.70,
      "pnl_realized": 5.25,
      "pnl_unrealized": 5.00,
      "opened_at": "2026-01-15T10:30:00Z",
      "closed_at": null
    }
  ],
  "total": 25,
  "page": 1
}
```

---

### 2. GET /trades

Historique des trades par utilisateur ou market.

**Paramètres**:
```json
{
  "user": "0x...",
  "market": "string",
  "limit": 100,
  "offset": 0,
  "start_date": "2026-01-01",
  "end_date": "2026-01-31"
}
```

**Réponse**:
```json
{
  "trades": [
    {
      "id": "trade_789",
      "market_id": "market_456",
      "user_address": "0x...",
      "side": "BUY|SELL",
      "outcome": "YES|NO",
      "price": 0.65,
      "size": 50.0,
      "fee": 0.25,
      "timestamp": "2026-01-15T10:30:00Z",
      "tx_hash": "0x..."
    }
  ],
  "total": 150,
  "page": 1
}
```

---

### 3. GET /activity

Activité récente d'un utilisateur (trades, positions ouvertes/fermées).

**Paramètres**:
```json
{
  "user": "0x...",
  "limit": 50,
  "types": ["trade", "position_opened", "position_closed"]
}
```

**Réponse**:
```json
{
  "activities": [
    {
      "type": "trade",
      "timestamp": "2026-01-15T10:30:00Z",
      "details": {
        "market_id": "market_456",
        "side": "BUY",
        "outcome": "YES",
        "price": 0.65,
        "size": 50.0
      }
    }
  ]
}
```

---

### 4. GET /holders (LIMITE: 20)

⚠️ **LIMITATION CRITIQUE**: Cet endpoint retourne UNIQUEMENT les 20 plus gros holders par market.

**Paramètres**:
```json
{
  "market": "market_456",
  "outcome": "YES|NO",
  "limit": 20  // Maximum: 20
}
```

**Réponse**:
```json
{
  "holders": [
    {
      "address": "0x...",
      "outcome": "YES",
      "size": 1500.0,
      "avg_price": 0.60,
      "current_value": 1050.0,
      "pnl": 50.0
    }
  ],
  "total_holders": 1247,  // Total réel (mais seuls 20 retournés)
  "returned": 20
}
```

**⚠️ PROBLÈME**: Pour obtenir TOUS les holders (pas seulement 20), voir [bitquery-graphql.mdc](bitquery-graphql.mdc).

---

### 5. GET /leaderboard

Classement des top traders par ROI, volume ou win rate.

**Paramètres**:
```json
{
  "metric": "roi|volume|win_rate",
  "category": "crypto|politics|sports|all",
  "period": "24h|7d|30d|all",
  "limit": 100
}
```

**Réponse**:
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "address": "0x...",
      "display_name": "TraderPro",
      "roi": 45.8,
      "volume": 125000.0,
      "win_rate": 0.68,
      "trades_count": 342,
      "category_affinity": "crypto"
    }
  ]
}
```

---

## Cas d'Usage pour l'Algorithme

### Phase 1: Seeding Top Traders
```python
import requests

def fetch_top_traders(category="all", limit=50):
    url = "https://data-api.polymarket.com/leaderboard"
    params = {
        "metric": "roi",
        "category": category,
        "period": "30d",
        "limit": limit
    }
    response = requests.get(url, params=params)
    return response.json()["leaderboard"]

# Seeder la base de données
top_traders = fetch_top_traders("crypto", 50)
```

### Phase 2: Surveillance Positions
```python
def monitor_trader_positions(trader_address):
    url = "https://data-api.polymarket.com/positions"
    params = {
        "user": trader_address,
        "status": "active"
    }
    response = requests.get(url, params=params)
    return response.json()["positions"]
```

### Phase 3: Historique ROI
```python
def calculate_trader_roi(trader_address, days=90):
    url = "https://data-api.polymarket.com/trades"
    params = {
        "user": trader_address,
        "start_date": (datetime.now() - timedelta(days=days)).isoformat(),
        "end_date": datetime.now().isoformat()
    }
    response = requests.get(url, params=params)
    trades = response.json()["trades"]

    # Calculer ROI global
    total_invested = sum(t["size"] * t["price"] for t in trades if t["side"] == "BUY")
    total_returns = sum(t["size"] * t["price"] for t in trades if t["side"] == "SELL")
    roi = (total_returns - total_invested) / total_invested * 100

    return roi
```

---

## Rate Limits & Quotas

| Plan | Requests/minute | Requests/jour | Prix |
|------|-----------------|---------------|------|
| Free | 100 | 10,000 | Gratuit |
| Basic | 1,000 | 100,000 | 50 USD/mois |
| Pro | 5,000 | 500,000 | 200 USD/mois |

**Budget 200-500 USD/mois**: Plan Pro recommandé (5000 req/min suffit pour polling fréquent).

---

## Best Practices

1. **Cache agressif**: Holder lists ne changent pas toutes les secondes, cache 1-5 minutes
2. **Batching**: Grouper requêtes pour plusieurs traders en parallèle
3. **Incremental updates**: Ne fetch que les nouvelles positions depuis last_sync
4. **Pagination**: Utiliser offset/limit pour gros datasets
5. **Monitoring**: Logger les 429 (rate limit) et backoff exponentiel

---

## Limitations Connues

| Limitation | Impact | Solution |
|------------|--------|----------|
| **20 holders max** | Données incomplètes | Utiliser Bitquery GraphQL (voir [bitquery-graphql.mdc](bitquery-graphql.mdc)) |
| **Rate limiting** | Latence sur pics | Cache + queue avec retry |
| **Pas de WebSocket** | Polling requis | Utiliser WebSocket CLOB pour prix temps réel (voir [websocket-api.mdc](websocket-api.mdc)) |
| **Historique limité** | Cold start difficile | Combiner avec on-chain data (Bitquery) |

---

## Ressources

- [Documentation officielle](https://docs.polymarket.com/quickstart/reference/endpoints)
- [Python client](https://github.com/Polymarket/py-clob-client)
- [Exemples d'intégration](https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf)

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - sym-web-research
**Projet**: Polymarket Trading Algorithm
