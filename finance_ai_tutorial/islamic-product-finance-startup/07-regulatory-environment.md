# Module 07: Regulatory Environment

## 7.1 Introduction

Islamic finance operates under dual regulatory frameworks: conventional prudential regulation and Shariah governance. This module covers key regulatory bodies, licensing considerations, and cross-border issues relevant to Islamic product-based finance AI.

**References**: IFSB standards; central bank regulations (BNM, SAMA, CBUAE, CBB); AAOIFI.

---

## 7.2 Regulatory Architecture

### 7.2.1 Dual Pillars

| Pillar | Purpose | Bodies |
|--------|---------|--------|
| Prudential | Capital, liquidity, risk management | Central banks, IFSB |
| Shariah | Compliance with Shariah | SSB, national Shariah council, AAOIFI |

### 7.2.2 Typical Structure

- **Central bank**: Licensing, prudential supervision
- **Shariah Advisory Council (SAC)** or equivalent: National-level Shariah rulings; some jurisdictions (e.g. Malaysia) have binding SAC
- **Institution SSB**: Bank-level Shariah oversight; reports to board

---

## 7.3 Key Jurisdictions

### 7.3.1 Malaysia

- **Regulator**: Bank Negara Malaysia (BNM)
- **Shariah**: Shariah Advisory Council (SAC) — resolutions binding on courts
- **Framework**: Shariah Governance Framework (SGF); Islamic Financial Services Act (IFSA)
- **Licensing**: Islamic bank license; Islamic window (conventional bank); digital bank (including Islamic)
- **AI**: BNM has issued guidance on fintech and digital; Shariah compliance required for Islamic products

### 7.3.2 Saudi Arabia

- **Regulator**: Saudi Central Bank (SAMA)
- **Shariah**: SAMA Shariah Committee; bank-level SSBs
- **Standards**: AAOIFI widely adopted
- **Licensing**: Full bank; financing company
- **Vision 2030**: Push for fintech, digital; Islamic finance central

### 7.3.3 UAE

- **Regulators**: Central Bank of UAE (CBUAE); Dubai Financial Services Authority (DFSA) for DIFC
- **Shariah**: Central Bank Shariah Authority; bank SSBs; DFSA has own Shariah framework for DIFC
- **Licensing**: Full bank; finance company; fintech licenses (e.g. regulatory sandbox)
- **Abu Dhabi Global Market (ADGM)**: Separate regulator; fintech-friendly

### 7.3.4 Bahrain

- **Regulator**: Central Bank of Bahrain (CBB)
- **Shariah**: CBB Shariah Board; AAOIFI mandatory for Shariah standards
- **Licensing**: Conventional and Islamic bank; financing; crowdfunding; sandbox
- **Fintech**: CBB has active fintech sandbox

### 7.3.5 Indonesia

- **Regulator**: Otoritas Jasa Keuangan (OJK)
- **Shariah**: National Shariah Council (DSN-MUI); fatwas guide industry
- **Licensing**: Islamic bank; Islamic business unit; fintech (P2P, etc.)
- **Market**: Large Muslim population; Islamic finance growing

### 7.3.6 UK

- **Regulator**: FCA, PRA
- **Shariah**: No national Shariah authority; bank SSBs; UK Islamic Finance Council (UKIFC) advisory
- **Licensing**: Full bank (e.g. Gatehouse Bank); Shariah-compliant products under standard FCA regime

---

## 7.4 Licensing Considerations for Fintech

### 7.4.1 Model-Dependent

| Model | Typical license |
|-------|-----------------|
| B2B SaaS (no balance sheet) | May not need banking license; check local rules |
| API provider | Usually no license if not taking deposits or lending |
| White-label underwriting | Depends on who bears credit risk; often bank is regulated |
| Lending (balance sheet) | Requires lending/finance license in most jurisdictions |

### 7.4.2 Sandbox Options

- **Bahrain**: CBB sandbox
- **UAE**: ADGM, DFSA sandboxes
- **Malaysia**: BNM sandbox
- **Saudi**: SAMA sandbox
- **UK**: FCA sandbox

Sandbox can allow testing of AI underwriting, screening, or pricing with relaxed rules for limited period.

---

## 7.5 Cross-Border Considerations

### 7.5.1 Shariah Heterogeneity

- **Malaysia vs GCC**: Some structural differences (e.g. Bay al-Inah, Tawarruq); fatwas may differ
- **Solution**: Build adaptable screening and structure logic; allow jurisdiction-specific SSB overrides

### 7.5.2 Data and Privacy

- **GDPR**: EU customers; data processing, right to explanation
- **Local laws**: UAE, Saudi, etc. have data localization and privacy rules
- **Implication**: AI systems must support data residency, consent, explainability

### 7.5.3 Outsourcing and Cloud

- Regulators (BNM, SAMA, CBB, etc.) have outsourcing guidelines
- Cloud: Some require in-country or approved cloud; data residency
- AI vendor as outsourcer: Contract, audit rights, SSB access to logic

---

## 7.6 AI-Specific Regulatory Trends

### 7.6.1 Emerging Guidance

- **Shariah Governance Standard on Gen AI (2025)**: SSB oversight, human-in-loop, no fatwa by AI
- **EU AI Act**: Risk-based; credit scoring is high-risk; explainability required
- **Local**: BNM, SAMA, others may issue AI/fintech guidance

### 7.6.2 Implications

- **Explainability**: Required for credit; align with Shariah auditability
- **Human oversight**: Align with SSB human-in-loop
- **Documentation**: Model cards, data provenance, audit trail

---

## 7.7 Synthesis for Startup

1. **Choose anchor jurisdiction**: Malaysia, UAE, or Bahrain for Islamic fintech focus.
2. **Engage regulator early**: Sandbox or informal consultation.
3. **SSB and Shariah**: Ensure SSB approval and alignment with national Shariah framework.
4. **Licensing**: Clarify if B2B SaaS/API requires license; if lending, plan for finance license or bank partnership.
5. **Cross-border**: Design for jurisdiction-specific Shariah and data rules from outset.
