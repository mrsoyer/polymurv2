# Polymarket WebSocket API

> API temps réel pour orderbook, prix et notifications utilisateur

---

## Overview

Polymarket fournit deux services WebSocket pour les données temps réel :

1. **CLOB WebSocket**: Orderbook, prix, statut ordres
2. **RTDS (Real-Time Data Service)**: Feeds crypto, commentaires

---

## 1. CLOB WebSocket

### Connection

**Endpoint**: `wss://ws-subscriptions-clob.polymarket.com/ws/`

```javascript
const ws = new WebSocket('wss://ws-subscriptions-clob.polymarket.com/ws/');

ws.onopen = () => {
  console.log('Connected to CLOB WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleMessage(data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Connection closed, reconnecting...');
  setTimeout(() => connect(), 5000); // Reconnect après 5s
};
```

### Channels Disponibles

#### Market Channel (Public)

Données publiques de prix et orderbook pour un token.

**Subscribe**:
```json
{
  "type": "subscribe",
  "channel": "market",
  "markets": ["0x..."]  // Liste de token_ids
}
```

**Messages Reçus**:

**Price Update**:
```json
{
  "channel": "market",
  "type": "price_update",
  "data": {
    "token_id": "0x...",
    "price": 0.65,
    "bid": 0.64,
    "ask": 0.66,
    "volume_24h": 125000.0,
    "timestamp": "2026-02-04T12:00:00Z"
  }
}
```

**Orderbook Update**:
```json
{
  "channel": "market",
  "type": "orderbook_update",
  "data": {
    "token_id": "0x...",
    "bids": [
      {"price": 0.64, "size": 500.0}
    ],
    "asks": [
      {"price": 0.66, "size": 400.0}
    ],
    "timestamp": "2026-02-04T12:00:00Z"
  }
}
```

**Trade**:
```json
{
  "channel": "market",
  "type": "trade",
  "data": {
    "token_id": "0x...",
    "price": 0.65,
    "size": 50.0,
    "side": "BUY",
    "timestamp": "2026-02-04T12:00:00Z"
  }
}
```

---

#### User Channel (Authentifié)

Notifications de statut d'ordres utilisateur.

**Subscribe** (Authentifié):
```json
{
  "type": "subscribe",
  "channel": "user",
  "auth_token": "your_auth_token"
}
```

**Messages Reçus**:

**Order Status Update**:
```json
{
  "channel": "user",
  "type": "order_status",
  "data": {
    "order_id": "ord_123",
    "status": "FILLED",  // LIVE, PARTIAL, FILLED, CANCELLED
    "filled": 100.0,
    "remaining": 0.0,
    "avg_fill_price": 0.65,
    "timestamp": "2026-02-04T12:00:00Z"
  }
}
```

**Position Update**:
```json
{
  "channel": "user",
  "type": "position_update",
  "data": {
    "token_id": "0x...",
    "side": "YES",
    "size": 100.0,
    "avg_price": 0.65,
    "pnl_unrealized": 5.00,
    "timestamp": "2026-02-04T12:00:00Z"
  }
}
```

---

### Unsubscribe

```json
{
  "type": "unsubscribe",
  "channel": "market",
  "markets": ["0x..."]
}
```

---

## 2. RTDS WebSocket

### Connection

**Endpoint**: `wss://ws-live-data.polymarket.com`

```javascript
const rtds = new WebSocket('wss://ws-live-data.polymarket.com');
```

### Channels Disponibles

#### Crypto Price Feeds
```json
{
  "type": "subscribe",
  "channel": "crypto_prices",
  "symbols": ["BTC", "ETH", "SOL"]
}
```

**Messages**:
```json
{
  "channel": "crypto_prices",
  "symbol": "BTC",
  "price": 102345.67,
  "change_24h": 2.5,
  "timestamp": "2026-02-04T12:00:00Z"
}
```

#### Comment Streams
```json
{
  "type": "subscribe",
  "channel": "comments",
  "market_id": "market_456"
}
```

---

## Cas d'Usage pour l'Algorithme

### Détection Temps Réel des Mouvements de Prix

```python
import asyncio
import websockets
import json

async def monitor_market_prices(token_ids):
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/"

    async with websockets.connect(uri) as ws:
        # Subscribe aux tokens à surveiller
        subscribe_msg = {
            "type": "subscribe",
            "channel": "market",
            "markets": token_ids
        }
        await ws.send(json.dumps(subscribe_msg))

        # Loop écoute
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "price_update":
                token_id = data["data"]["token_id"]
                price = data["data"]["price"]

                # Vérifier si mouvement significatif
                check_price_movement(token_id, price)

            elif data["type"] == "trade":
                # Détecter gros trades (whales)
                if data["data"]["size"] > 1000:
                    alert_whale_trade(data["data"])

# Lancer monitoring
asyncio.run(monitor_market_prices(["0x...", "0x..."]))
```

### Notification Ordres Remplis

```python
async def monitor_user_orders(auth_token):
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/"

    async with websockets.connect(uri) as ws:
        # Subscribe user channel
        subscribe_msg = {
            "type": "subscribe",
            "channel": "user",
            "auth_token": auth_token
        }
        await ws.send(json.dumps(subscribe_msg))

        async for message in ws:
            data = json.loads(message)

            if data["type"] == "order_status":
                order_id = data["data"]["order_id"]
                status = data["data"]["status"]

                if status == "FILLED":
                    # Ordre rempli, commencer surveillance holders
                    start_holder_monitoring(order_id)
                elif status == "CANCELLED":
                    # Ordre annulé, retry si nécessaire
                    handle_cancel(order_id)
```

---

## Reconnection Handling

```python
class PolymarketWebSocket:
    def __init__(self, uri, on_message):
        self.uri = uri
        self.on_message = on_message
        self.ws = None
        self.running = False

    async def connect(self):
        while self.running:
            try:
                async with websockets.connect(self.uri) as ws:
                    self.ws = ws
                    print("Connected")

                    async for message in ws:
                        await self.on_message(json.loads(message))

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed, reconnecting in 5s...")
                await asyncio.sleep(5)

            except Exception as e:
                print(f"Error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def start(self):
        self.running = True
        asyncio.create_task(self.connect())

    def stop(self):
        self.running = False
```

---

## Latence

| Événement | Latence Typique |
|-----------|-----------------|
| Price update | 50-200ms |
| Trade execution | 100-300ms |
| Order status | 200-500ms |
| Orderbook update | 50-150ms |

**⚠️ Latence critique pour l'algorithme**:
- Temps entre trade d'un top trader et détection : **100-500ms**
- Temps entre détection et placement ordre : **200-800ms**
- **Total latency** : 300-1300ms

**Conséquence**: Le prix peut avoir déjà bougé de 1-3% avant que notre ordre s'exécute. C'est l'**edge lost** du copy trading.

---

## Rate Limits

| Type | Limite |
|------|--------|
| Connections simultanées | 5 par compte |
| Subscriptions par connection | 50 channels |
| Messages envoyés | 10/seconde |

---

## Best Practices

1. **Reconnection automatique**: Toujours implémenter reconnection avec backoff exponentiel
2. **Heartbeat**: Envoyer ping toutes les 30s pour garder connexion vivante
3. **Buffer messages**: Si traitement lent, buffer les messages pour ne pas bloquer
4. **Multiple connections**: Utiliser plusieurs connexions si > 50 tokens à surveiller
5. **Prioritize user channel**: Si budget connexions limité, prioriser user channel pour nos ordres

---

## Limitations Connues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **5 connections max** | Limite tokens suivis | Grouper tokens par priorité, rotate subscriptions |
| **50 channels/connection** | Max 250 tokens simultanés | Utiliser Data API polling pour le reste |
| **Latence 100-500ms** | Prix déjà mové | Accepter edge loss, optimiser vitesse traitement |
| **Pas de replay** | Données perdues si disconnected | Logger côté client, backfill via Data API |

---

## Comparaison: WebSocket vs Polling

| Aspect | WebSocket | Polling Data API |
|--------|-----------|------------------|
| **Latence** | 50-200ms | 1-5 secondes |
| **Bandwidth** | Faible (push) | Élevé (pull répété) |
| **Complexité** | Reconnection handling | Simple HTTP |
| **Cost** | Gratuit | Rate limits API |
| **Use case** | Prix temps réel | Historique, holders |

**Recommandation pour l'algorithme**:
- **WebSocket**: Prix temps réel pour détection signaux
- **Data API**: Holders, historique ROI, leaderboard

---

## Ressources

- [Documentation CLOB](https://docs.polymarket.com/)
- [WebSocket RFC](https://datatracker.ietf.org/doc/html/rfc6455)
- [Python websockets library](https://websockets.readthedocs.io/)

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - sym-web-research
**Projet**: Polymarket Trading Algorithm
