# Phase 5: Monitoring - Surveillance Post-Achat

> Surveiller TOUS les holders d'une position ouverte

---

## Objectif

Dès qu'on achète, surveiller TOUS les holders (pas juste 20) pour détecter vagues de vente.

**Input**: Nos positions ouvertes
**Output**: Alertes vagues de vente
**Fréquence**: Batch 5 min (cache plus fréquent)
**Coût**: Même budget Bitquery (149 USD/mois)

---

## Logic Détection Vague Vente

```python
def detect_sell_wave(our_position):
    event = our_position.event
    
    # Fetch snapshot actuel ALL holders
    current_holders = bitquery.getAllHolders(event.token, side=our_position.side)
    
    # Comparer avec snapshot précédent (5 min ago)
    previous_holders = get_previous_snapshot(our_position.id)
    
    # Détecter sorties
    exited_holders = []
    for prev in previous_holders:
        current = find_holder(current_holders, prev.address)
        if not current or current.size < prev.size * 0.5:
            exited_holders.append(prev)
    
    # Analyser confiance holders sortis
    high_confidence_exits = [
        h for h in exited_holders
        if get_trader_confidence(h.address) > 0.7
    ]
    
    # Vague détectée si > 30% holders fiables sortent
    if len(high_confidence_exits) / len(exited_holders) > 0.3:
        return {
            "alert": "SELL_WAVE_DETECTED",
            "exited_high_confidence": len(high_confidence_exits),
            "total_exited": len(exited_holders)
        }
    
    return None
```

---

→ **Phase 6: Sell Signal** ([06-phase-sell-signal.md](06-phase-sell-signal.md))
