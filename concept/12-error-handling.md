# Gestion des Erreurs et Résilience

> Retry, fallback, circuit breaker pour un système robuste

---

## Objectif

Documenter la stratégie de gestion d'erreurs pour garantir la résilience du système face aux défaillances APIs, rate limits et erreurs de trading.

---

## Catégories d'Erreurs

### 1. Erreurs APIs Externes

| Type | Cause | Impact | Mitigation |
|------|-------|--------|------------|
| **Rate Limiting** | Trop de requêtes | Données manquantes | Backoff exponentiel |
| **Timeout** | API lente/down | Latence élevée | Retry avec timeout progressif |
| **500 Server Error** | Problème serveur | Indisponibilité temporaire | Retry + fallback |
| **401 Unauthorized** | Clé API invalide | Service bloqué | Rotation clés + alert |
| **Network Error** | Connexion perdue | Pas de données | Retry + cache |

---

### 2. Erreurs Trading

| Type | Cause | Impact | Action |
|------|-------|--------|--------|
| **FOK Failed** | Pas de liquidité | Ordre non exécuté | Retry 3× puis skip |
| **Insufficient Funds** | Balance trop basse | Trade impossible | Alert + pause trading |
| **Invalid Token** | Token introuvable | Ordre rejeté | Skip market |
| **Gas Fee Spike** | Polygon congestion | Frais élevés | Wait + retry |

---

### 3. Erreurs Data Quality

| Type | Cause | Impact | Action |
|------|-------|--------|--------|
| **Missing Holders** | Bitquery incomplete | Scores faussés | Fallback RPC ou skip |
| **Stale Data** | Cache expiré | Décisions sur vieilles données | Force refresh |
| **Inconsistent ROI** | Calcul erroné | Faux signaux | Validation + alert |

---

## Stratégies de Retry

### Backoff Exponentiel

```python
import time

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """
    Retry avec délai exponentiel
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Dernier essai, propager erreur

            delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
            print(f"Retry {attempt + 1}/{max_retries} après {delay}s...")
            time.sleep(delay)

# Usage
holders = retry_with_backoff(
    lambda: bitquery.getAllHolders(token_id),
    max_retries=3,
    base_delay=2
)
```

---

### Retry avec Jitter

```python
import random

def retry_with_jitter(func, max_retries=3):
    """
    Retry avec randomisation (évite thundering herd)
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            # Delay: 1-3s, puis 2-6s, puis 4-12s
            base = 2 ** attempt
            delay = random.uniform(base, base * 3)

            print(f"Retry après {delay:.1f}s...")
            time.sleep(delay)
```

---

## Circuit Breaker

### Concept

Arrêter temporairement un service si trop d'échecs consécutifs.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # 5 min
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

    def call(self, func):
        # Si circuit ouvert, ne pas essayer
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"  # Retry après timeout
            else:
                raise Exception("Circuit breaker OPEN, service unavailable")

        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            alert_admin("Circuit breaker OPEN for Bitquery")

# Usage
bitquery_breaker = CircuitBreaker(failure_threshold=5, timeout=300)

def fetch_holders_safe(token_id):
    return bitquery_breaker.call(
        lambda: bitquery.getAllHolders(token_id)
    )
```

**Comportement** :
- 5 échecs consécutifs → Circuit OPEN (5 min)
- Pendant 5 min → Pas d'appels Bitquery
- Après 5 min → Retry (HALF_OPEN)
- Si succès → Circuit CLOSED

---

## Fallback Strategy

### Hiérarchie Sources Données

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIMARY: Bitquery GraphQL                                      │
│  ├─ Latence: 1-3 sec                                            │
│  ├─ Coverage: TOUS les holders                                  │
│  └─ Si échec → SECONDARY                                        │
├─────────────────────────────────────────────────────────────────┤
│  SECONDARY: Polymarket Data API                                 │
│  ├─ Latence: 200-500ms                                          │
│  ├─ Coverage: Top 20 holders                                    │
│  └─ Si échec → TERTIARY                                         │
├─────────────────────────────────────────────────────────────────┤
│  TERTIARY: Polygon RPC                                          │
│  ├─ Latence: Minutes (scan blocks)                              │
│  ├─ Coverage: TOUS (lent)                                       │
│  └─ Dernier recours                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
def get_all_holders_resilient(token_id):
    """
    Fetch holders avec fallback automatique
    """

    # Try PRIMARY
    try:
        return bitquery_breaker.call(
            lambda: bitquery.getAllHolders(token_id)
        )
    except Exception as e:
        log_error("Bitquery failed", e)

    # Try SECONDARY
    try:
        holders_top20 = polymarket_api.getTopHolders(token_id, limit=20)
        log_warning("Using top 20 holders only (Bitquery down)")
        return holders_top20
    except Exception as e:
        log_error("Polymarket API failed", e)

    # Try TERTIARY (dernier recours)
    try:
        log_warning("Falling back to Polygon RPC (slow)")
        return polygon_rpc.getAllHoldersViaScan(token_id, recent_blocks_only=True)
    except Exception as e:
        log_critical("All sources failed", e)
        raise Exception("Cannot fetch holders, all sources down")
```

---

## Rate Limiting Protection

### Token Bucket Algorithm

```python
class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def can_proceed(self):
        now = time.time()

        # Remove requêtes hors fenêtre
        self.requests = [r for r in self.requests if now - r < self.window_seconds]

        return len(self.requests) < self.max_requests

    def record_request(self):
        self.requests.append(time.time())

    def wait_if_needed(self):
        while not self.can_proceed():
            time.sleep(0.1)  # Wait 100ms

# Usage
bitquery_limiter = RateLimiter(max_requests=120, window_seconds=60)  # 120 req/min

def fetch_with_rate_limit(token_id):
    bitquery_limiter.wait_if_needed()
    result = bitquery.getAllHolders(token_id)
    bitquery_limiter.record_request()
    return result
```

---

## Gestion Erreurs Trading

### FOK Order Failed

```python
def execute_buy_with_retry(signal, event):
    """
    Tenter placement ordre avec retry
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            order = clob_client.create_market_order({
                "token_id": event.token,
                "amount": signal.size,
                "side": "BUY",
                "order_type": OrderType.FOK
            })

            result = clob_client.post_order(order)

            if result.status == "FILLED":
                return result  # Succès

            # FOK failed (pas de liquidité)
            if attempt < max_retries - 1:
                print(f"FOK failed, retry {attempt + 1}/{max_retries}")
                time.sleep(5)  # Wait 5s
            else:
                log_warning(f"FOK failed après {max_retries} tentatives, skip trade")
                return None

        except Exception as e:
            log_error(f"Trading error: {e}")
            if attempt == max_retries - 1:
                alert_admin(f"Trading failed: {e}")
                return None
            time.sleep(2)

    return None
```

---

### Insufficient Funds

```python
def check_balance_before_trade():
    """
    Vérifier balance avant de trader
    """

    balance = clob_client.get_balance()

    if balance < MIN_BALANCE_THRESHOLD:
        alert_admin(f"Low balance: {balance} USDC")
        pause_trading()
        return False

    return True
```

---

## Logging et Alerting

### Niveaux de Log

| Level | Usage | Action |
|-------|-------|--------|
| **DEBUG** | Trace détaillée | Log file only |
| **INFO** | Opérations normales | Log file |
| **WARNING** | Problèmes non-bloquants | Log + metrics |
| **ERROR** | Erreurs récupérables | Log + metrics + retry |
| **CRITICAL** | Erreurs bloquantes | Log + alert admin + pause |

### Alerting Rules

```python
alerts = {
    "bitquery_down": {
        "condition": "Circuit breaker OPEN",
        "action": "Email + SMS admin",
        "urgency": "HIGH"
    },

    "trading_fees_exceeded": {
        "condition": "Fees > 200 USD/mois",
        "action": "Email admin + pause trading",
        "urgency": "MEDIUM"
    },

    "win_rate_drop": {
        "condition": "Win rate < 50% sur 20 derniers trades",
        "action": "Email + suggest strategy review",
        "urgency": "MEDIUM"
    },

    "max_drawdown_exceeded": {
        "condition": "Drawdown > 20%",
        "action": "Email + SMS + pause trading",
        "urgency": "CRITICAL"
    }
}
```

---

## Monitoring Health

### Health Check Endpoint

```python
def health_check():
    """
    Vérifier santé de tous les composants
    """

    health = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "components": {}
    }

    # Check Bitquery
    try:
        bitquery.health_check()
        health["components"]["bitquery"] = "healthy"
    except:
        health["components"]["bitquery"] = "degraded"
        health["overall_status"] = "degraded"

    # Check Polymarket API
    try:
        response = requests.get("https://data-api.polymarket.com/health")
        health["components"]["polymarket"] = "healthy"
    except:
        health["components"]["polymarket"] = "down"
        health["overall_status"] = "critical"

    # Check Database
    try:
        db.query("SELECT 1")
        health["components"]["database"] = "healthy"
    except:
        health["components"]["database"] = "down"
        health["overall_status"] = "critical"

    # Check Workers
    health["components"]["workers"] = check_workers_status()

    return health
```

---

## Résumé

```
╔═══════════════════════════════════════════════════════════════════╗
║  STRATÉGIE RÉSILIENCE                                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  APIs :                                                           ║
║  ├─ Retry avec backoff exponentiel (3×)                          ║
║  ├─ Circuit breaker (5 échecs → pause 5 min)                     ║
║  ├─ Fallback Bitquery → Polymarket → RPC                         ║
║  └─ Rate limiting protection (token bucket)                      ║
║                                                                   ║
║  Trading :                                                        ║
║  ├─ Retry FOK failed (3× avec pause 5s)                          ║
║  ├─ Check balance avant trade                                    ║
║  ├─ Cap trading fees (max 50 trades/jour)                        ║
║  └─ Alert si balance faible                                      ║
║                                                                   ║
║  Monitoring :                                                     ║
║  ├─ Health check toutes les 5 min                                ║
║  ├─ Alerting multi-niveau (DEBUG → CRITICAL)                     ║
║  ├─ Pause automatique si drawdown > 20%                          ║
║  └─ Admin notification SMS + Email                               ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - Corrections sym-opus
**Status**: ✅ Error handling complet
