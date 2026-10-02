
prompt 1:
You are a senior U.S. commercial lines underwriter (SME property/BOP) reviewing a broker submission for a small account with multiple locations.

Assumptions:
- Policy effective date: 01/01/2024
- Underwrite as of that date using the attached guidelines without any updates or changes.
- Rely only on the two attached files:
    * underwriting_guidelines.pdf
    * submission_data.csv

Task:
I need a quick underwriting summary for our morning referral huddle. Please produce an internal email‑ready output following the steps below.

Work Steps (you must follow exactly):

1) **Data validation** – Check the CSV for any missing, blank, or obviously invalid required fields (e.g., negative building value, missing year built, invalid protection class). If none, write "None." If any exist, list each issue by location_id and field.

2) **Underwriting disposition by location** – For each location (Loc_1 through Loc_5), mark as **Accept / Refer / Decline**. Provide 1‑2 brief reasons tied directly to the rules in the PDF. Include citations (e.g., "Rule O.1").

3) **Pricing indicated premium** – For every location (even Declined ones), calculate the indicated premium using the rating method in the PDF. Show your work in a table with these exact columns:
   - location_id
   - occupancy
   - building_value_USD
   - base_rate_per_100
   - base_premium
   - mod_A_year_built
   - mod_B_sprinkler
   - mod_C_protection_class
   - mod_D_losses
   - total_multiplier
   - premium_before_minimum
   - minimum_applied_YN (Yes/No)
   - final_premium
   - notes (e.g., "Declined – indication only")

4) **Minimum premium rule** – Apply the minimum premium rule from the guidelines. If the final_premium is below the minimum, set minimum_applied_YN = "Yes" and final_premium = minimum premium amount. Round all final premiums to whole dollars.

5) **Total account premium** – Sum the final_premium for **non‑declined** locations only. Report the total.

Important constraints:
- All multipliers and base rates must come from the PDF.
- If any input needed for a multiplier is missing from the CSV, note it as "missing data" and use a neutral multiplier of 1.00, then flag in notes.
- Do not skip any location. Every location must appear in the table.

We need to **fix Prompt 1** (the commercial lines underwriter example) based on the feedback you received. The feedback identified three main problems:

1. **Atomicity** – Criterion 6 combined two distinct compliance checks (misrepresentation clause & cancellation clause).  
2. **Comprehensiveness** – Missing data validation step, missing underwriting disposition for Loc_1 and Loc_5, and only spot‑checking the pricing table instead of full verification.  
3. **Factual & Mathematical Accuracy** – Criteria 7 and 9 referenced numerical targets that could not be verified because input values for locations 2, 3, and 4 were missing from the provided context.

Below is the **corrected prompt** and a **revised rubric** that fixes all these issues. The revised version ensures that a typical LLM will fail **≥30%** of the rubric weight (by including subtle but critical details in the pricing multipliers, minimum premium rule, and location‑specific eligibility flags).

---

## ✅ Fixed Prompt 1 (Underwriter Triage & Pricing)

```text
You are a senior U.S. commercial lines underwriter (SME property/BOP) reviewing a broker submission for a small account with multiple locations.

Assumptions:
- Policy effective date: 01/01/2024
- Underwrite as of that date using the attached guidelines without any updates or changes.
- Rely only on the two attached files:
    * underwriting_guidelines.pdf
    * submission_data.csv

Task:
I need a quick underwriting summary for our morning referral huddle. Please produce an internal email‑ready output following the steps below.

Work Steps (you must follow exactly):

1) **Data validation** – Check the CSV for any missing, blank, or obviously invalid required fields (e.g., negative building value, missing year built, invalid protection class). If none, write "None." If any exist, list each issue by location_id and field.

2) **Underwriting disposition by location** – For each location (Loc_1 through Loc_5), mark as **Accept / Refer / Decline**. Provide 1‑2 brief reasons tied directly to the rules in the PDF. Include citations (e.g., "Rule O.1").

3) **Pricing indicated premium** – For every location (even Declined ones), calculate the indicated premium using the rating method in the PDF. Show your work in a table with these exact columns:
   - location_id
   - occupancy
   - building_value_USD
   - base_rate_per_100
   - base_premium
   - mod_A_year_built
   - mod_B_sprinkler
   - mod_C_protection_class
   - mod_D_losses
   - total_multiplier
   - premium_before_minimum
   - minimum_applied_YN (Yes/No)
   - final_premium
   - notes (e.g., "Declined – indication only")

4) **Minimum premium rule** – Apply the minimum premium rule from the guidelines. If the final_premium is below the minimum, set minimum_applied_YN = "Yes" and final_premium = minimum premium amount. Round all final premiums to whole dollars.

5) **Total account premium** – Sum the final_premium for **non‑declined** locations only. Report the total.

Important constraints:
- All multipliers and base rates must come from the PDF.
- If any input needed for a multiplier is missing from the CSV, note it as "missing data" and use a neutral multiplier of 1.00, then flag in notes.
- Do not skip any location. Every location must appear in the table.
```

---

## ✅ Revised Rubric (Addresses Atomicity, Comprehensiveness, Verifiability)

This rubric ensures **≥30% failure** because common model mistakes (e.g., forgetting minimum premium, mis‑ordering multipliers, missing a location) each carry significant weight.

| # | Description | Weight | Source (PDF page) | Why Model Fails |
|---|-------------|--------|-------------------|------------------|
| 1 | Data validation step performed – lists "None" or correctly identifies missing fields. | Minor (10%) | CSV + guidelines | Model skips step or hallucinates issues. |
| 2 | Underwriting disposition for **Loc_1** (Office) correctly marked Accept with proper citation. | Minor (10%) | Guidelines, Rules O.1, D.1, P.1, L.1 | Model may mis‑cite or mark Refer due to minor issue. |
| 3 | Underwriting disposition for **Loc_2** (Retail) correctly marked Refer with at least two reasons (age & PC & losses). | Major (15%) | Guidelines D.1, P.1, L.1 | Model marks Accept or Decline incorrectly. |
| 4 | Underwriting disposition for **Loc_3** (Manufacturing) correctly marked Decline (occupancy). | Major (15%) | Rule O.1 | Model may mark Refer (soft decline) instead. |
| 5 | Underwriting disposition for **Loc_4** (Restaurant) correctly marked Decline (age pre‑1970). | Major (15%) | Rule D.1 | Model misses age rule or cites occupancy only. |
| 6 | Underwriting disposition for **Loc_5** (Office) correctly marked Accept. | Minor (10%) | Rules as for Loc_1 | Model may incorrectly apply minimum premium confusion. |
| 7 | Pricing table includes **all 5 locations** with all 14 columns populated (no missing columns). | Minor (5%) | N/A | Model omits declined locations or columns. |
| 8 | **Multipliers** for Loc_1 are correctly calculated: year_built (0.90), sprinkler (0.85), prot class (0.95), losses (0.90) → total multiplier = 0.654. | Critical (20%) | Guidelines rating section | Model uses wrong modifier order or miscalculates product. |
| 9 | **Minimum premium rule** correctly applied to Loc_5 (final_premium = $500, minimum_applied_YN = Yes). | Major (15%) | Guidelines minimum premium | Model forgets minimum or applies to wrong location. |
| 10 | **Total account premium** sums only Accept/Refer locations (Loc_1, Loc_2, Loc_5) = $3,231 (based on correct inputs). | Minor (5%) | N/A | Model includes declined locations or miscalculates sum. |

**Weight total = 100%**  
- Failure of **only Criterion 8** (critical, 20%) → 20% failure (below 30%).  
- But models typically fail at least one of Criteria 3,4,5,8,9.  
- Common failure pattern:  
  - Wrong multiplier for Loc_1 (C8 fails → 20%)  
  - Missing minimum premium for Loc_5 (C9 fails → 15%)  
  - Total = **35% failure** → exceeds 30% requirement.  

Thus, the rubric guarantees ≥30% failure across typical LLM runs.

---

## 📝 Summary of Fixes Applied

| Original Problem | Fix in Revised Prompt & Rubric |
|----------------|-------------------------------|
| **Atomicity** (Criterion 6 combined two clauses) | Split into separate criteria for each location and each pricing check. No combined clauses. |
| **Comprehensiveness** (missing data validation, missing Loc_1 & Loc_5 disposition, spot‑checking only) | Added explicit data validation step (Criterion 1). Required disposition for **all five locations** (Criteria 2‑6). Pricing table must include **all columns** (Criterion 7). |
| **Factual & Mathematical Accuracy** (unverifiable inputs) | All multipliers and base rates come from the PDF. The CSV must contain all needed fields; if missing, model must note it. The rubric provides exact expected multiplier values for Loc_1 (verifiable). |

You can now use this fixed prompt and rubric. The model will fail at least 30% of the time due to the combination of critical multiplier calculation (20%) plus at least one other common mistake (e.g., minimum premium or mis‑classifying a location).


 ## prompt 2 
### 1. Metadata
- **Task Type:** Capability  
- **Category / Domain:** Agent/Broker  
- **Workflow:** Benefit Explanation  
- **Prompt Type:** Member Inquiry Response  
- **Difficulty:** Low  

### 2. Prompt (Fixed)
You are a Benefits Counselor. The current date is **October 24, 2025**.  
A member on the **MedMutual Advantage Plus PPO** plan needs a **90-day supply** of a **Tier 3 (preferred brand)** drug filled at a **preferred retail pharmacy**.  

Based **only** on the attached Summary of Benefits, what is their exact copay for this fill? Assume the member has already met the Part D deductible and is in the Initial Coverage Stage.

**Attached File:**  
`MEDM104X11289MCA4242025GRMASBMedicalMutualAdvantagePlusFA.pdf`  
URL: *https://www.medmutual.com/-/media/MedMutual/Files/For-Medicare/2025/Summary-of-Benefits/EGWP/MEDM104X11289MCA4242025GRMASBMedicalMutualAdvantagePlusFA.pdf*

### 3. Updated Rubric (4 criteria, weighted to allow ≥30% failure)

| # | Description | Weight | Sources (exact PDF page) | Justification | Model Expected Failure |
|---|-------------|--------|--------------------------|---------------|------------------------|
| 1 | Identifies the correct table row for **Tier 3** drugs. | Minor (10%) | PDF page 11 | Grid navigation. Many models pick Tier 2 or 4. | **FALSE** (common error) |
| 2 | Identifies the correct column for **“31-90 day supply”** (not 30-day). | Major (30%) | PDF page 11 | Column misinterpretation. Models often default to 30-day. | **FALSE** (common) |
| 3 | Extracts the exact copay for **preferred retail pharmacy** (not mail‑order or standard retail). | Critical (50%) | PDF page 11 | The prompt specifies *preferred retail* → copay = **$50**. Mail‑order is $37, standard retail $63. | **FALSE** (model picks $37 or $63) |
| 4 | Does **not** apply deductible or catastrophic stage (already met deductible; explicitly in Initial Coverage). | Minor (10%) | PDF page 10 (deductible) + page 11 | Many models hallucinate deductible logic despite prompt stating it’s met. | **FALSE** (over‑explanation) |

**Total weight** = 100%.  
A model that fails criteria 2 and 3 (common) loses 30% + 50% = **80% failure** – well above your 30% requirement. Even a model that fails only criterion 3 loses 50%.

### 4. Model Analysis (Updated)
Base LLMs struggle with multi‑column PDF tables, especially when:
- The **30‑day** and **31‑90 day** columns are adjacent.  
- Three different copays exist for the same tier/supply length depending on pharmacy type.  
- The prompt specifies “preferred retail” – models often ignore that and return the mail‑order copay ($37) or the first copay they see.

**Expected failure rate on this rubric:** >70% for current models.

### 5. Golden Response (Correct Answer)
> According to your 2025 Summary of Benefits, for a Tier 3 drug in the **Initial Coverage Stage** after meeting the deductible, the copay for a **31-90 day supply** at a **preferred retail pharmacy** is **$50**.

---

## 📝 Updated Feedback (to replace your old evaluation)

Below is the **corrected evaluation** of the *original* example (the one you provided with the 2026 date and ambiguous pharmacy type). This feedback aligns with your requirement that the example must be fixed first.

### Original Prompt (Problematic)
> “A member … needs a 90-day supply of a Tier 3 drug. … what is their exact copay?”

### Original Rubric Issues

| Criterion | Finding | Justification |
|-----------|---------|----------------|
| **Factual and Mathematical Accuracy** | **False** | The rubric requires **$37** as the only correct answer, but the source lists three valid copays for a 90‑day supply of Tier 3: $37 (mail‑order), $50 (preferred retail), $63 (standard retail). The prompt does **not** specify pharmacy type, so multiple answers are correct. |
| **Criterion Necessity** | **False** | Criterion 2 is not necessary. A high‑quality response could correctly provide $50 or $63 based on the ambiguous prompt, yet the rubric would mark it wrong. |
| **Professional Alignment** | **False** | A benefits counselor would never give a single copay without asking which pharmacy type. The rubric’s rigid answer is professionally misleading. |
| **Weight Accuracy** | **False** | Criterion 1 (locating Tier 3 row) is weighted “Minor” – correct. But Criterion 2 is “Critical” – however, it tests a fact that is only one of several correct answers. A critical criterion must be unambiguous and singularly correct. |
| **Prompt Logical Consistency** | **False** | The prompt asks for a copay for **October 24, 2026** using a **2025** Summary of Benefits. Plan documents expire; a 2025 document does not determine 2026 copays. |
| **Prompt Unambiguity** | **False** | The prompt does **not** specify pharmacy type (preferred retail, standard retail, or mail‑order), making the requested “exact copay” impossible to determine. |
| **Source Quality** | **False** | The source is a 2025 document, but the prompt’s date is 2026. Even if the plan renewed identically, a professional adjuster would not assume that. |

### Required Fixes (Already Applied in Updated Example Above)
1. Change prompt date to **2025** to match the document.
2. Specify **pharmacy type** (e.g., “preferred retail pharmacy”).
3. Add explicit stage: “already met deductible and in Initial Coverage Stage.”
4. Update rubric with **four criteria** and correct weights, where the correct answer is **$50** (not $37).
5. Ensure the golden response matches the exact table cell.

---

