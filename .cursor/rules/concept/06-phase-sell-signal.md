# Phase 6: Sell Signal - Génération Signaux de Vente

> Vendre automatiquement quand holders fiables sortent

---

## Objectif

Détecter vagues de vente (Phase 5) et exécuter vente automatique.

**Input**: Alertes monitoring
**Output**: Ordres de vente CLOB API
**Latence**: < 500ms

---

## Logic Sell Signal

```python
def generate_sell_signal(alert, our_position):
    # Vérifier confiance holders sortis
    if alert.exited_high_confidence / alert.total_exited > 0.3:
        return {
            "action": "SELL",
            "reason": "HIGH_CONFIDENCE_EXIT",
            "urgency": "HIGH"
        }
    
    # Vérifier profit target
    current_price = get_current_price(our_position.token)
    pnl_pct = (current_price - our_position.entry_price) / our_position.entry_price
    
    if pnl_pct > 0.20:  # +20% profit
        return {
            "action": "SELL",
            "reason": "PROFIT_TARGET",
            "urgency": "NORMAL"
        }
    
    # Stop-loss
    if pnl_pct < -0.10:  # -10% loss
        return {
            "action": "SELL",
            "reason": "STOP_LOSS",
            "urgency": "CRITICAL"
        }
    
    return None
```

---

## Exécution Vente

```python
def execute_sell_signal(signal, our_position):
    # Placer ordre market
    order = clob_client.create_market_order({
        "token_id": our_position.token,
        "amount": our_position.size,
        "side": "SELL",
        "order_type": OrderType.FOK
    })
    
    result = clob_client.post_order(order)
    
    # Calculer PnL réalisé
    realized_pnl = (result.price - our_position.entry_price) * our_position.size
    
    # Close position
    close_position(our_position.id, realized_pnl)
    
    return result
```
