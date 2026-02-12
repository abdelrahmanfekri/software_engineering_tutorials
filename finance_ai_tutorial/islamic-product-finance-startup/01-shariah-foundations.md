# Module 01: Shariah Foundations — Riba, Gharar, Maysir, Maqasid

## 1.1 Introduction

Islamic finance rests on prohibitions (Riba, Gharar, Maysir) and objectives (Maqasid al-Shariah). Understanding these is essential for designing product-based financing that is both Shariah-compliant and commercially viable. This module covers classical positions, contemporary interpretations, and implications for AI systems.

**References**: AAOIFI Shariah Standards 1–59; Islamic Fiqh Academy; Chapra (2000) *The Islamic Vision of Development*; El-Gamal (2006) *Islamic Finance*.

---

## 1.2 Riba (Interest / Usury)

### 1.2.1 Definition and Scope

**Riba** (literally: increase, growth) denotes an unlawful excess in exchange of certain commodities or a predetermined return on a loan. Scholarly consensus treats it as categorically prohibited (haram) based on Quran (2:275–280, 3:130, 4:161) and Hadith.

**Two classical types:**

1. **Riba al-Nasi’ah** (Riba of delay): Excess demanded in exchange for deferral of payment. A loan of 100 repaid as 110 at maturity is riba.
2. **Riba al-Fadl** (Riba of excess): Unequal exchange of homogenous fungibles (e.g. gold for gold, wheat for wheat) without immediate delivery.

Modern finance primarily implicates Riba al-Nasi’ah: any predetermined return on a debt obligation.

### 1.2.2 What Constitutes Riba in Practice

| Structure | Shariah view | Rationale |
|-----------|--------------|-----------|
| Interest-bearing loan | Prohibited | Predetermined return on debt |
| Murabaha (cost + margin) | Permitted | Sale of asset at disclosed profit; not a loan |
| Profit-sharing (Musharakah) | Permitted | Return tied to actual profit; no guarantee |
| Ijarah (lease) | Permitted | Rent for use of asset; not interest on money |
| Tawarruq (commodity monetization) | Controversial | Sale of commodity for deferred payment; some scholars disallow institutional tawarruq |
| Late payment penalty | Restricted | Cannot be stipulated as extra profit; may be charity or compensation with conditions |

### 1.2.3 Implications for Product-Based Financing

- Pricing must be **asset- or transaction-based**, not interest on a loan.
- Murabaha margin, Musharakah profit share, Ijarah rent must be **disclosed and fixed at contract inception** for the sale/lease element; they are not interest.
- Late payment penalties: AAOIFI allows stipulating a penalty, but it must go to charity or be a genuine compensation mechanism, not additional profit for the financier.
- AI systems must avoid any logic, labels, or outputs that frame returns as "interest" or "interest rate."

---

## 1.3 Gharar (Excessive Uncertainty)

### 1.3.1 Definition and Scope

**Gharar** refers to excessive uncertainty (gharar fahish) that vitiates consent or makes the object of the contract indeterminate. It is prohibited to prevent unjust enrichment and exploitation.

**Elements that may constitute gharar:**

- Sale of non-existent or undeliverable object
- Sale of object whose existence, quantity, or quality is unknown
- Excessive ambiguity in price, delivery, or subject matter
- Gambling-like outcomes (overlap with Maysir)

### 1.3.2 Permissible vs Prohibited Uncertainty

| Scenario | Typical view | Rationale |
|----------|--------------|-----------|
| Sale of future commodity (Salam) | Permitted (with conditions) | Price and quantity specified; object defined |
| Sale of known asset for deferred payment | Permitted | Asset exists; terms clear |
| Sale of fish in the sea | Prohibited | Object not in seller's possession; excessive uncertainty |
| Insurance (conventional) | Prohibited | Aleatory; resembles gambling |
| Takaful | Permitted | Cooperative risk-sharing; Shariah-compliant structure |
| Derivatives (options, futures) | Controversial | Some permit hedging; many prohibit speculative use |
| ML model uncertainty | Must be managed | Gharar minimization in AI outputs; explainability |

### 1.3.3 Implications for AI Systems

- **Gharar minimization**: Model outputs (e.g. risk scores, recommendations) should be interpretable and grounded. Excessive opacity or non-explainable decisions can introduce gharar-like uncertainty.
- **Adverse action**: Reasons must reference concrete factors (payment capacity, asset quality, Shariah screening)—not vague or unexplained logic.
- **Confidence**: When confidence is low, systems should refer to human review rather than auto-approve/decline.
- **Gharar score**: Quantifying prediction uncertainty (e.g. entropy, variance) can support Shariah governance and SSB audit.

---

## 1.4 Maysir (Gambling / Speculation)

### 1.4.1 Definition and Scope

**Maysir** denotes gambling and pure speculation where one party gains at another's loss without productive activity. It is prohibited (Quran 2:219, 5:90–91).

**Characteristics:**

- Zero-sum or near-zero-sum outcome
- No underlying productive activity
- Reliance on chance rather than effort or asset
- Transfer of wealth without fair exchange

### 1.4.2 Permissible vs Prohibited

| Activity | View | Rationale |
|----------|------|-----------|
| Pure gambling | Prohibited | Maysir |
| Short-selling (naked) | Generally prohibited | Sale of non-owned asset; speculative |
| Hedging genuine risk | Often permitted | Protection of real exposure |
| Day trading / high-frequency speculation | Controversial | May resemble maysir if no economic purpose |
| Profit-sharing (Musharakah) | Permitted | Sharing of real business outcome |
| Trade finance (Murabaha) | Permitted | Sale of real asset; economic purpose |

### 1.4.3 Implications for Product-Based Financing

- Financing must be **tied to real assets or transactions** (e.g. trade, equipment, inventory).
- Speculative or purely financial transactions without underlying asset/activity are suspect.
- Product taxonomy should distinguish **productive use** (e.g. import of goods for resale) from **speculative use** (e.g. purely financial arbitrage).

---

## 1.5 Maqasid al-Shariah (Objectives of Shariah)

### 1.5.1 Framework

**Maqasid al-Shariah** are the higher objectives of Islamic law. Classical scholars (e.g. Al-Ghazali, Al-Shatibi) identify:

1. **Hifz al-Din** (preservation of religion)
2. **Hifz al-Nafs** (preservation of life)
3. **Hifz al-Aql** (preservation of intellect)
4. **Hifz al-Nasl** (preservation of progeny)
5. **Hifz al-Mal** (preservation of wealth)

Contemporary Islamic finance emphasizes **Hifz al-Mal** (protection of wealth) and broader social welfare: fairness, transparency, avoidance of exploitation, and contribution to economic development.

### 1.5.2 Alignment with Product-Based Financing

| Maqasid principle | Application |
|-------------------|-------------|
| Preservation of wealth | Asset-backed financing reduces systemic risk; no interest-based debt spiral |
| Fairness | Product-based risk pricing aligns margin with actual risk; no hidden charges |
| Transparency | Disclosed cost, margin, profit share; explainable AI outputs |
| Economic development | Financing real trade and productive assets |
| Avoidance of exploitation | No riba; no gharar; no maysir |

---

## 1.6 Synthesis for AI Product Design

### Design Principles

1. **No riba**: Never frame returns as interest; use sale/lease/profit-sharing language.
2. **Minimize gharar**: Explainable decisions; confidence thresholds; human referral when uncertain.
3. **Avoid maysir**: Tie financing to real assets and productive activity.
4. **Maqasid alignment**: Transparency, fairness, asset-backing, economic purpose.

### Operational Checklist for AI Outputs

- [ ] No use of terms: interest, APR, interest rate, riba
- [ ] Use: margin, profit share, rent, installment (for sale/lease)
- [ ] Adverse action reasons: payment capacity, asset quality, Shariah screening—never interest
- [ ] Confidence/referral logic for low-confidence cases
- [ ] Data provenance: halal sources; no haram-sector inputs for Shariah scoring
