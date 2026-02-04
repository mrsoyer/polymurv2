# Phase 7: Resolution - Gestion Clôture Markets

> Que faire quand un market se résout et les positions sont automatiquement payées

---

## Objectif

Gérer la résolution des markets et le payout automatique des positions ouvertes.

**Input**: Event résolu + nos positions ouvertes
**Output**: Positions closes, PnL réalisé, mise à jour DB
**Trigger**: Market status = "resolved"
**Coût**: Gratuit (payout automatique Polymarket)

---

## Détection Résolution

### Monitoring Status Events

```python
def monitor_event_resolutions():
    """
    Vérifier statut des events de notre watchlist
    """

    # Fetch events avec positions ouvertes
    our_events = db.query("""
        SELECT DISTINCT e.id, e.polymarket_id, e.resolution_date
        FROM events e
        JOIN our_positions p ON p.event_id = e.id
        WHERE p.status = 'open'
        AND e.status = 'active'
    """)

    for event in our_events:
        # Fetch status actuel
        response = requests.get(
            f"https://data-api.polymarket.com/markets/{event.polymarket_id}"
        )
        market = response.json()

        # Vérifier résolution
        if market.status == "resolved":
            handle_resolution(event, market)
```

**Fréquence** : Batch 1×/heure (résolutions annoncées à l'avance)

---

### Worker Cron

```sql
SELECT cron.schedule(
  'check-event-resolutions',
  '0 * * * *',  -- Every hour
  $$
    SELECT net.http_post(
      url := 'https://[project].supabase.co/functions/v1/check-resolutions',
      headers := '{"Authorization": "Bearer [key]"}',
      body := '{}'
    );
  $$
);
```

---

## Gestion Payout

### Cas 1 : Position Gagnante

```python
def handle_resolution(event, market):
    """
    Traiter résolution d'un market
    """

    # Récupérer nos positions sur cet event
    our_positions = db.query("""
        SELECT * FROM our_positions
        WHERE event_id = %s AND status = 'open'
    """, event.id)

    for position in our_positions:
        # Déterminer outcome gagnant
        winning_outcome = market.resolved_outcome  # "YES" ou "NO"

        # Calculer PnL
        if position.side == winning_outcome:
            # GAGNANT : payout = taille position × 1 USD
            payout = position.size
            pnl = payout - (position.entry_price * position.size)
            result = "WIN"
        else:
            # PERDANT : payout = 0
            payout = 0
            pnl = -(position.entry_price * position.size)
            result = "LOSS"

        # Clore position
        db.execute("""
            UPDATE our_positions
            SET
              status = 'closed',
              closed_at = NOW(),
              exit_price = %s,
              pnl_realized = %s,
              close_reason = 'RESOLUTION'
            WHERE id = %s
        """, (1.0 if result == "WIN" else 0.0, pnl, position.id))

        # Logger
        log_resolution({
            "event_id": event.id,
            "position_id": position.id,
            "result": result,
            "pnl": pnl,
            "payout": payout
        })
```

---

### Cas 2 : Résolution Invalide/Annulée

```python
if market.status == "invalid":
    # Market annulé → remboursement complet
    for position in our_positions:
        refund = position.entry_price * position.size

        db.execute("""
            UPDATE our_positions
            SET
              status = 'refunded',
              closed_at = NOW(),
              pnl_realized = 0,  # Pas de gain/perte
              close_reason = 'MARKET_INVALID'
            WHERE id = %s
        """, position.id)
```

---

## Payout Automatique Polymarket

**Important** : Polymarket gère le payout automatiquement via smart contracts.

```
Market résolu
    ↓
Smart contract distribue payouts
    ↓
Winning tokens → 1 USDC each
Losing tokens → 0 USDC
    ↓
Fonds apparaissent dans wallet automatiquement
```

**Action requise** : Aucune ! Juste mettre à jour notre DB pour refléter le résultat.

---

## Alerting Résolution

### Notification Utilisateur

```python
def notify_resolution(position, result, pnl):
    """
    Notifier utilisateur de la résolution
    """

    message = f"""
    Market résolu: {position.event_title}

    Résultat: {result} {'✅' if result == 'WIN' else '❌'}
    Votre position: {position.side}
    Entry price: {position.entry_price:.2f} USDC
    Size: {position.size:.2f}
    PnL: {pnl:+.2f} USD ({(pnl / (position.entry_price * position.size) * 100):+.1f}%)

    Payout: {position.size if result == 'WIN' else 0:.2f} USDC
    """

    # Envoyer notification
    send_email(user_email, "Market Résolu", message)
    # Ou Telegram, webhook, etc.
```

---

## Métriques Post-Résolution

### Mise à Jour Stats Globales

```sql
-- Après chaque résolution, mettre à jour stats
UPDATE global_stats
SET
  total_trades = total_trades + 1,
  total_wins = total_wins + (CASE WHEN result = 'WIN' THEN 1 ELSE 0 END),
  total_pnl = total_pnl + pnl_realized,
  win_rate = total_wins::FLOAT / total_trades,
  roi_cumulative = total_pnl / total_invested
WHERE id = 1;
```

### Dashboard Résolutions

```sql
SELECT
  DATE_TRUNC('day', closed_at) as date,
  COUNT(*) as resolutions_count,
  SUM(CASE WHEN pnl_realized > 0 THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN pnl_realized < 0 THEN 1 ELSE 0 END) as losses,
  SUM(pnl_realized) as daily_pnl
FROM our_positions
WHERE
  status = 'closed'
  AND close_reason = 'RESOLUTION'
  AND closed_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', closed_at)
ORDER BY date DESC;
```

---

## Edge Cases

### 1. Résolution Différée

**Problème** : Market devait se résoudre le 15/02, mais résolution reportée au 20/02.

**Solution** :
- Continuer surveillance (Phase 5)
- Pas d'action spéciale
- Attendre résolution effective

---

### 2. Position Ouverte Proche Résolution

**Problème** : On achète à J-1 de la résolution (trop tard).

**Protection** :
```yaml
filters:
  min_days_until_resolution: 2  # Ne pas entrer si < 2 jours
```

Si déjà en position et résolution imminente :
- Phase 6 génère signal sell préventif
- Exit avant résolution si incertitude

---

### 3. Market Invalide Post-Achat

**Problème** : Market annulé après qu'on ait acheté.

**Gestion** :
- Refund automatique Polymarket
- PnL = 0 (capital récupéré)
- Marquer position "refunded"

---

## Interaction avec Phase 6 (Sell Signal)

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIORITÉ EXIT                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 6 : Sell Signal (si détecté)                             │
│  ├─ Haute confiance holders sortent → SELL                      │
│  ├─ Profit target atteint → SELL                                │
│  └─ Stop-loss déclenché → SELL                                  │
│                                                                  │
│  Phase 7 : Resolution (si pas vendu avant)                      │
│  └─ Market résolu → Payout automatique                          │
│                                                                  │
│  Préférence : Sortir via Phase 6 (contrôle) > Attendre Phase 7  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Stratégie** : Essayer de sortir avant résolution (Phase 6) pour maximiser contrôle. Phase 7 = fallback si pas vendu.

---

## Prochaines Étapes

→ **Phase 8: Cold Start** ([08-cold-start.md](08-cold-start.md))

Démarrage progressif sur 30 jours.

---

## Ressources

- [Data API](../docapi/polymarket/data-api.md) - Endpoint status market
- [CLOB API](../docapi/polymarket/clob-api.md) - Pas d'action requise (payout auto)
- [Workers](../backend/workers/worker-monitor.md) - Cron hourly check

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - Corrections sym-opus
**Status**: ✅ Phase résolution documentée
