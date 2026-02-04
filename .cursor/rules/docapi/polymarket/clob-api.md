# Polymarket CLOB API

> Central Limit Order Book API pour le trading automatisé

---

## Overview

L'API CLOB (Central Limit Order Book) de Polymarket permet de placer des ordres, gérer des positions et exécuter des trades programmatiquement.

**Base URL**: `https://clob.polymarket.com`

**Architecture**: Hybride décentralisé
- Order book off-chain (operator centralisé)
- Settlement on-chain (smart contracts Polygon)

---

## Authentication

### Types de Signature

Polymarket supporte 3 types de signatures pour l'authentification:

#### Type 0: EOA (Externally Owned Account)
```python
from py_clob_client.client import ClobClient

client = ClobClient(
    "https://clob.polymarket.com",
    key="0x...",  # Private key
    chain_id=137,  # Polygon
    signature_type=0
)
```

**Usage**: MetaMask, hardware wallets, wallets dont vous contrôlez directement la clé privée.

#### Type 1: Email/Magic Wallet
```python
client = ClobClient(
    "https://clob.polymarket.com",
    key="0x...",
    chain_id=137,
    signature_type=1,
    funder="0x..."  # Adresse qui détient les fonds
)
```

**Usage**: Email wallets, Magic wallets avec gestion déléguée.

#### Type 2: Browser Proxy
```python
client = ClobClient(
    "https://clob.polymarket.com",
    key="0x...",
    chain_id=137,
    signature_type=2,
    funder="0x..."
)
```

**Usage**: Smart contract wallets, proxy contracts.

### API Credentials

```python
# Générer ou dériver les credentials API
client.set_api_creds(client.create_or_derive_api_creds())

# Les credentials dérivent de la clé privée
# Pas besoin de les fournir manuellement
```

---

## Endpoints Principaux

### 1. GET /price

Récupère le prix actuel d'un token conditionnel.

**Paramètres**:
```json
{
  "token_id": "0x..."  // Conditional token address
}
```

**Réponse**:
```json
{
  "token_id": "0x...",
  "price": 0.65,
  "bid": 0.64,
  "ask": 0.66,
  "last_trade": 0.65,
  "volume_24h": 125000.0,
  "timestamp": "2026-02-04T12:00:00Z"
}
```

---

### 2. GET /book

Récupère l'order book complet pour un token.

**Paramètres**:
```json
{
  "token_id": "0x...",
  "depth": 10  // Nombre de niveaux de prix (optionnel)
}
```

**Réponse**:
```json
{
  "token_id": "0x...",
  "bids": [
    {"price": 0.64, "size": 500.0},
    {"price": 0.63, "size": 300.0}
  ],
  "asks": [
    {"price": 0.66, "size": 400.0},
    {"price": 0.67, "size": 250.0}
  ],
  "timestamp": "2026-02-04T12:00:00Z"
}
```

---

### 3. GET /midpoint

Récupère le prix midpoint (moyenne bid/ask).

**Paramètres**:
```json
{
  "token_id": "0x..."
}
```

**Réponse**:
```json
{
  "token_id": "0x...",
  "midpoint": 0.65,
  "spread": 0.02,
  "bid": 0.64,
  "ask": 0.66
}
```

---

### 4. POST /order (Authentifié)

Place un ordre limit ou market.

#### Ordre Limit (Share-Based)
```python
from py_clob_client.order_builder.constants import OrderType

# Ordre limit GTC (Good-Till-Cancelled)
order_args = {
    "token_id": "0x...",
    "price": 0.65,
    "size": 100.0,
    "side": "BUY",  # ou "SELL"
    "order_type": OrderType.GTC
}

order = client.create_order(order_args)
signed_order = client.post_order(order)
```

**Réponse**:
```json
{
  "order_id": "ord_123",
  "status": "LIVE",
  "token_id": "0x...",
  "price": 0.65,
  "size": 100.0,
  "filled": 0.0,
  "remaining": 100.0,
  "created_at": "2026-02-04T12:00:00Z"
}
```

#### Ordre Market (Dollar-Based)
```python
# Ordre market FOK (Fill-or-Kill)
market_order_args = {
    "token_id": "0x...",
    "amount": 100.0,  # En USD
    "side": "BUY",
    "order_type": OrderType.FOK
}

order = client.create_market_order(market_order_args)
result = client.post_order(order)
```

---

### 5. POST /orders (Batch Orders)

Place plusieurs ordres simultanément.

**⚠️ Limite**: 15 ordres par batch (augmenté de 5 à 15 en 2025)

```python
orders = [
    {"token_id": "0x...", "price": 0.65, "size": 50.0, "side": "BUY"},
    {"token_id": "0x...", "price": 0.70, "size": 50.0, "side": "SELL"},
    # ... jusqu'à 15 ordres
]

batch_result = client.post_batch_orders(orders)
```

**Réponse**:
```json
{
  "orders": [
    {"order_id": "ord_123", "status": "LIVE"},
    {"order_id": "ord_124", "status": "LIVE"}
  ],
  "success": 15,
  "failed": 0
}
```

---

### 6. DELETE /order (Authentifié)

Annule un ordre par ID.

```python
cancel_result = client.cancel_order(order_id="ord_123")
```

**Réponse**:
```json
{
  "order_id": "ord_123",
  "status": "CANCELLED",
  "cancelled_at": "2026-02-04T12:05:00Z"
}
```

### 7. DELETE /orders (Cancel All)

Annule tous les ordres ouverts.

```python
client.cancel_all()
```

---

### 8. GET /orders (Open Orders)

Récupère les ordres ouverts d'un utilisateur.

```python
from py_clob_client.client import OpenOrderParams

open_orders = client.get_orders(OpenOrderParams())
```

**Réponse**:
```json
{
  "orders": [
    {
      "order_id": "ord_123",
      "token_id": "0x...",
      "price": 0.65,
      "size": 100.0,
      "filled": 25.0,
      "remaining": 75.0,
      "status": "LIVE",
      "created_at": "2026-02-04T12:00:00Z"
    }
  ]
}
```

---

## Prérequis: Approbations USDC

⚠️ **IMPORTANT**: Avant de trader, vous devez approuver 3 smart contracts pour dépenser vos tokens.

```python
# Approuver USDC pour trading
client.approve_usdc_spending()

# Approuver conditional tokens YES
client.approve_conditional_token(token_id="0x...", side="YES")

# Approuver conditional tokens NO
client.approve_conditional_token(token_id="0x...", side="NO")
```

Ces approbations permettent aux exchange contracts de déplacer vos fonds lors de l'exécution d'ordres.

---

## HeartBeat API

Endpoint de monitoring pour vérifier la connexion (utile pour bots long-running).

```bash
GET https://clob.polymarket.com/heartbeat
```

**Réponse**:
```json
{
  "status": "ok",
  "timestamp": "2026-02-04T12:00:00Z",
  "version": "2.1.0"
}
```

**Usage**: Ping toutes les 30 secondes pour détecter les disconnections.

---

## Cas d'Usage pour l'Algorithme

### Phase 4: Exécution Signal d'Achat

```python
def execute_buy_signal(token_id, roi_expected, confidence_score):
    # Calculer taille position basée sur confiance
    position_size_usd = 100 * confidence_score  # Max 100 USD

    # Placer ordre market
    order = client.create_market_order({
        "token_id": token_id,
        "amount": position_size_usd,
        "side": "BUY",
        "order_type": OrderType.FOK
    })

    result = client.post_order(order)

    # Enregistrer en DB
    save_our_position({
        "token_id": token_id,
        "entry_price": result["price"],
        "size": result["filled"],
        "roi_expected": roi_expected,
        "confidence": confidence_score
    })

    return result
```

### Phase 6: Exécution Signal de Vente

```python
def execute_sell_signal(our_position):
    # Récupérer position actuelle
    token_id = our_position["token_id"]
    size = our_position["size"]

    # Placer ordre market de vente
    order = client.create_market_order({
        "token_id": token_id,
        "amount": size,  # Vendre tout
        "side": "SELL",
        "order_type": OrderType.FOK
    })

    result = client.post_order(order)

    # Calculer PnL réalisé
    entry_price = our_position["entry_price"]
    exit_price = result["price"]
    realized_pnl = (exit_price - entry_price) * size

    # Mettre à jour DB
    update_position_closed(our_position["id"], realized_pnl)

    return result
```

### Monitoring Latence

```python
import time

def monitor_execution_latency():
    start = time.time()

    # Placer ordre test
    order = client.create_market_order({
        "token_id": "0x...",
        "amount": 10.0,
        "side": "BUY",
        "order_type": OrderType.FOK
    })
    result = client.post_order(order)

    latency = time.time() - start

    # Logger latence
    print(f"Ordre executé en {latency * 1000:.2f}ms")

    return latency
```

---

## Rate Limits & Coûts

### Rate Limits
| Endpoint | Limite | Fenêtre |
|----------|--------|---------|
| GET /price, /book, /midpoint | 100/sec | Par token |
| POST /order | 10/sec | Par utilisateur |
| POST /orders (batch) | 5/sec | Par utilisateur |
| DELETE /order | 20/sec | Par utilisateur |

### Frais de Trading
- **Maker fee**: 0.1% (ordres limit ajoutant liquidité)
- **Taker fee**: 0.2% (ordres market retirant liquidité)
- **Gas fees**: Payés sur Polygon lors du settlement (~0.001 USD/trade)

**Budget Impact (200-500 USD/mois)**:
- 100 trades/jour × 30 jours = 3000 trades/mois
- Volume moyen: 50 USD/trade
- Frais: 3000 × 50 × 0.002 = **300 USD/mois** en frais de trading
- **⚠️ IMPORTANT**: Les frais de trading sont le coût principal, pas les appels API

---

## Best Practices

1. **Fill-or-Kill pour signaux**: Utiliser FOK pour s'assurer que les ordres s'exécutent immédiatement ou pas du tout
2. **Stop-loss automatique**: Placer des ordres de vente stop-loss après chaque achat
3. **Cooldown après échec**: Si FOK échoue (pas de liquidité), attendre 30s avant retry
4. **Batch quand possible**: Grouper plusieurs ordres en batch pour réduire latence
5. **Monitoring HeartBeat**: Ping /heartbeat toutes les 30s pour détecter outages

---

## Limitations Connues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **15 ordres/batch max** | Pas de gros batch | Splitter en plusieurs batches |
| **Liquidité variable** | FOK peut échouer | Retry avec timeout exponentiel |
| **Settlement on-chain** | Latence 2-5 sec | Accepter latence, pas de HFT possible |
| **Gas fees Polygon** | Coût additionnel | Inclus automatiquement, négligeable |

---

## Ressources

- [GitHub: py-clob-client](https://github.com/Polymarket/py-clob-client)
- [Documentation officielle](https://docs.polymarket.com/)
- [NautilusTrader integration](https://nautilustrader.io/docs/latest/integrations/polymarket/)

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - sym-web-research
**Projet**: Polymarket Trading Algorithm
