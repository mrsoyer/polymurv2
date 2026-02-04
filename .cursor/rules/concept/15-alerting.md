# Alerting & Notifications

> Notification utilisateur et admin des événements critiques

---

## Objectif

Documenter le système d'alerting pour monitoring et intervention.

---

## Canaux de Notification

| Canal | Usage | Latence |
|-------|-------|---------|
| **Email** | Rapports quotidiens | 1-5 min |
| **Telegram** | Alertes temps réel | < 30 sec |
| **SMS** | Alertes critiques | < 1 min |
| **Webhook** | Integration custom | < 5 sec |

---

## Alertes Critiques

### Trading

```yaml
alerts_trading:
  - trigger: "Balance < 100 USD"
    channels: [email, sms]
    action: "Pause trading automatiquement"

  - trigger: "Drawdown > 20%"
    channels: [sms, telegram]
    action: "Pause + review strategy"

  - trigger: "Trade executed"
    channels: [telegram]
    action: "Notification info"

  - trigger: "Position closed"
    channels: [telegram]
    action: "Notification avec PnL"
```

### Infrastructure

```yaml
alerts_infrastructure:
  - trigger: "Bitquery down"
    channels: [email, telegram]
    action: "Fallback RPC activé"

  - trigger: "Worker failed"
    channels: [email]
    action: "Retry automatique"

  - trigger: "Rate limit hit"
    channels: [telegram]
    action: "Throttling activé"
```

---

## Rapports Automatiques

### Rapport Quotidien (Email)

```
Subject: Trading Report - 2026-02-04

Positions ouvertes: 5
PnL journalier: +12.50 USD (+1.25%)
PnL cumulé: +127.80 USD (+12.78%)

Trades aujourd'hui: 3
├─ WIN: 2 (avg +8.5%)
└─ LOSS: 1 (-5.2%)

Win rate global: 68% (45/66 trades)
```

### Rapport Hebdomadaire

- Performance vs benchmarks
- Top 5 best trades
- Analyse secteurs
- Recommendations ajustements

---

**Status**: ✅ Alerting documenté
