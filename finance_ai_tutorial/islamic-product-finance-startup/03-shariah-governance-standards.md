# Module 03: Shariah Governance Standards — AAOIFI, IFSB, SSB

## 3.1 Introduction

Shariah governance is the framework by which Islamic financial institutions ensure compliance with Shariah principles. Key bodies are AAOIFI (standards), IFSB (prudential guidance), and institutional Shariah Supervisory Boards (SSBs). This module covers standards, SSB role, fatwa process, and Shariah audit.

**References**: AAOIFI Governance Standard for IFIs 1 (GSIFI 1); IFSB-10 Guiding Principles on Shariah Governance; Bank Negara Malaysia Shariah Governance Framework.

---

## 3.2 AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions)

### 3.2.1 Role

AAOIFI develops Shariah standards, accounting standards (FAS), and auditing standards for Islamic financial institutions. Its Shariah Standards are widely adopted (mandatory in Bahrain, Saudi, UAE in various forms; reference in Malaysia, Indonesia).

### 3.2.2 Key Shariah Standards

| Standard | Subject |
|----------|---------|
| SS 1 | Shariah rules for the issuance of contemporary Islamic financial instruments |
| SS 2 | Concepts and principles (riba, gharar, maysir) |
| SS 8 | Murabaha and Murabaha to the purchase orderer |
| SS 9 | Ijarah and Ijarah Muntahia Bittamleek |
| SS 12 | Musharakah |
| SS 13 | Mudarabah |
| SS 17 | Investment Sukuk |
| SS 59 | Shariah standards for tawarruq |

### 3.2.3 Implications for Product Design

- Product features (Murabaha, Musharakah, Ijarah) must align with AAOIFI structure.
- Disclosure, possession, and documentation requirements must be met.
- Late payment treatment per AAOIFI (charity or compensation; not profit).

---

## 3.3 IFSB (Islamic Financial Services Board)

### 3.3.1 Role

IFSB issues prudential and supervisory standards for the Islamic financial services industry. It complements AAOIFI with capital adequacy, risk management, and governance guidance.

### 3.3.2 Key Standards

| Standard | Subject |
|----------|---------|
| IFSB-1 | Capital adequacy |
| IFSB-2 | Risk management |
| IFSB-3 | Corporate governance |
| IFSB-10 | Shariah governance |
| IFSB-12 | Liquidity risk |
| IFSB-15 | Revised capital adequacy |

### 3.3.3 IFSB-10: Shariah Governance Principles

- **Independence**: SSB independent from management
- **Competence**: SSB members with Shariah and finance expertise
- **Transparency**: Disclosure of Shariah governance framework and fatwas
- **Consistency**: Consistent application across products and jurisdictions
- **Audit**: Internal and external Shariah compliance review

---

## 3.4 Shariah Supervisory Board (SSB)

### 3.4.1 Role

The SSB (or Shariah Committee) issues fatwas on products, structures, and operations; reviews compliance; and advises the board and management.

### 3.4.2 Typical Composition

- 3–5 members (varies by jurisdiction)
- Scholars with Shariah expertise and familiarity with finance
- Some jurisdictions require mix of Shariah and finance/legal

### 3.4.3 Key Functions

1. **Product approval**: Fatwa on new products and structures
2. **Ongoing review**: Annual or periodic compliance review
3. **Dispute resolution**: Shariah-related complaints and interpretations
4. **Training**: Awareness for staff and management

### 3.4.4 Fatwa Process

1. Management submits product/structure proposal
2. SSB reviews against Shariah principles and AAOIFI
3. SSB issues fatwa (permitted, prohibited, or permitted with conditions)
4. Fatwa documented and disclosed
5. Implementation monitored

**For AI systems**: SSB must approve use of AI in underwriting, screening, or pricing. Shariah Governance Standard on Generative AI (2025) requires SSB oversight of Gen AI applications. AI outputs must be auditable and explainable.

---

## 3.5 Shariah Screening Methodology

### 3.5.1 Business Activity Screening

**Prohibited sectors** (typically):

- Alcohol, pork, gambling, conventional insurance
- Interest-based financial services
- Tobacco (controversial; some permit)
- Weapons (controversial; some permit defense)
- Adult entertainment

### 3.5.2 Financial Ratio Screening (for equity/investment)

Common thresholds (AAOIFI, S&P, MSCI, FTSE):

| Ratio | Threshold |
|-------|-----------|
| Debt / Total assets | ≤ 33% |
| Interest income / Total revenue | ≤ 5% |
| Non-permissible income / Total revenue | ≤ 5% |
| Liquid assets / Total assets | ≤ 70% (avoid cash-like firms) |
| Interest-bearing debt / Total assets | ≤ 33% |

### 3.5.3 Screening for Product-Based Financing

For trade finance and Murabaha:

- **Commodity/asset**: Must be halal (no alcohol, pork, haram goods)
- **Counterparty activity**: Business must not be predominantly haram
- **Use of proceeds**: Must not finance haram activity

AI screening modules must exclude haram categories and apply thresholds per SSB-approved methodology.

---

## 3.6 Shariah Audit

### 3.6.1 Internal Shariah Audit

- Periodic review of transactions, products, and processes
- Verification of compliance with fatwas and standards
- Reporting to SSB and audit committee

### 3.6.2 External Shariah Audit

- Independent review by external Shariah auditor (in some jurisdictions)
- Opinion on Shariah compliance for financial statements
- Required in Bahrain, Malaysia, others

### 3.6.3 AI System Auditability

For AI-driven underwriting and screening:

- **Traceability**: Logs of inputs, model version, outputs
- **Explainability**: Adverse action reasons; feature contributions
- **Data provenance**: Sources used; halal-only for Shariah scoring
- **SSB review**: Periodic review of AI logic and outputs

---

## 3.7 Synthesis for Startup

1. **SSB from day one**: Engage SSB or scholar early for product and AI approval.
2. **AAOIFI alignment**: Structure products per AAOIFI (Murabaha, Musharakah, Ijarah).
3. **Documentation**: Document fatwas, screening methodology, AI logic.
4. **Audit trail**: Build traceability and explainability into AI pipeline.
5. **Jurisdiction-specific**: Adapt to local SSB and regulator (BNM, SAMA, CBUAE, etc.).
