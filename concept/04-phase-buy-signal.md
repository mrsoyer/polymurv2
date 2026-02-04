# Phase 4: Buy Signal - Génération Signaux d'Achat

> Décider automatiquement si acheter basé sur scores agrégés

---

## Objectif

Analyser les scores event et placer ordres d'achat si seuils dépassés.

**Input**: Scores event (Phase 3) + seuils configurables
**Output**: Ordres d'achat CLOB API
**Latence**: < 500ms
**Coût**: Trading fees (0.1-0.2%)

---

## Logic Signal d'Achat

```python
def generate_buy_signal(event):
    scores = get_event_scores(event.id)
    config = get_config()  # conservateur/équilibré/agressif
    
    # Comparer YES vs NO
    roi_diff = scores.roi_avg_yes - scores.roi_avg_no
    confidence_diff = scores.confidence_avg_yes - scores.confidence_avg_no
    
    # Signal YES
    if (roi_diff > config.min_roi_diff and
        confidence_diff > config.min_confidence_diff and
        scores.roi_avg_yes > config.min_roi_absolute):
        return {
            "action": "BUY",
            "side": "YES",
            "roi_expected": scores.roi_avg_yes,
            "confidence": scores.confidence_avg_yes
        }
    
    # Signal NO
    elif (roi_diff < -config.min_roi_diff and
          confidence_diff < -config.min_confidence_diff and
          scores.roi_avg_no > config.min_roi_absolute):
        return {
            "action": "BUY",
            "side": "NO",
            "roi_expected": scores.roi_avg_no,
            "confidence": scores.confidence_avg_no
        }
    
    return None  # Pas de signal
```

---

## Seuils par Profil

| Profil | min_roi_diff | min_confidence_diff | min_roi_absolute |
|--------|--------------|---------------------|------------------|
| Conservateur | 10% | 0.2 | 15% |
| Équilibré | 7% | 0.15 | 10% |
| Agressif | 5% | 0.1 | 5% |

---

## Exécution Ordre

```python
def execute_buy_signal(signal, event):
    # Calculer taille position
    position_size = calculate_position_size(signal.confidence)
    
    # Placer ordre market FOK
    order = clob_client.create_market_order({
        "token_id": event.yes_token if signal.side == "YES" else event.no_token,
        "amount": position_size,
        "side": "BUY",
        "order_type": OrderType.FOK
    })
    
    result = clob_client.post_order(order)
    
    # Enregistrer notre position
    save_our_position({
        "event_id": event.id,
        "side": signal.side,
        "entry_price": result.price,
        "size": result.filled,
        "roi_expected": signal.roi_expected,
        "confidence": signal.confidence
    })
    
    return result
```

---

→ **Phase 5: Monitoring** ([05-phase-monitoring.md](05-phase-monitoring.md))
