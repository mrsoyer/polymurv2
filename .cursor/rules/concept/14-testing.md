# Testing & Validation Strategy

> Tests pour garantir la fiabilité de l'algorithme

---

## Objectif

Documenter la stratégie de tests avant déploiement production.

---

## Tests Unitaires

### Formules de Calcul

```python
def test_confidence_calculation():
    # Test pondération volume
    trader = {"roi": 45, "win_rate": 0.9, "trades_count": 10}
    confidence = calculate_confidence_score(trader)
    assert confidence == 0.081  # Pénalisé pour faible volume

    trader2 = {"roi": 38, "win_rate": 0.65, "trades_count": 500}
    confidence2 = calculate_confidence_score(trader2)
    assert confidence2 == 0.247  # Préféré
```

---

## Tests Integration

### Workflow Complet

```python
def test_full_workflow():
    # Phase 1: Seed traders
    traders = seed_top_traders("crypto", limit=10)
    assert len(traders) >= 5  # Au moins 5 valides

    # Phase 2: Discover events
    events = discover_events(traders)
    assert len(events) > 0

    # Phase 3: Enrich
    scores = enrich_events(events)
    assert all(s.roi_avg_yes is not None for s in scores)

    # Phase 4: Generate signals
    signals = generate_buy_signals(scores)
    # Au moins un signal ou aucun (selon market)
```

---

## Smoke Tests Production

```bash
# Avant chaque déploiement
python tests/smoke_test.py

# Vérifications:
# ✅ Bitquery accessible
# ✅ Polymarket API responsive
# ✅ Database connection OK
# ✅ Workers déployés
# ✅ Balance wallet > minimum
```

---

**Status**: ✅ Tests documentés
