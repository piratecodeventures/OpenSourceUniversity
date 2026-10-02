# Advanced Evaluation Frameworks for Artificial Intelligence in Complex Insurance and Reinsurance Workflows

### 

---

Example 1: Commercial Property Adjudication and Coinsurance Penalty Arithmetic

#### 1. Metadata

- Sub-domain: Claims Handling (Commercial Property)
    
- Difficulty: High
    
- Themes: Coinsurance Penalties, Debris Removal Limitations, Sub-limits, Order of Operations
    
- Task Type: Workflow
    
- Workflow: Quantum Calculation
    
- Prompt Type: Formal Claims Adjudication Memo
    

#### 2. Prompt

You are an Executive General Adjuster handling a complex commercial property claim. The current date is October 24, 2026.

On October 15, 2026, a severe fire caused substantial direct physical damage to a commercial property located in San Francisco. You must calculate the final total payable amount to the insured. Present your final deliverable as a formal claims adjudication memo, showing all step-by-step mathematical calculations. Round all final dollar amounts to the nearest whole number.

Loss Details:

- Direct physical damage from fire: $2,000,000
    
- Debris removal costs incurred: $80,000
    
- Expense to extract pollutants from the land caused by the fire: $65,000
    
- Fire department service charge: $5,000
    
- The building's actual Replacement Cost Value (RCV) at the time of loss: $10,000,000
    
- The Limit of Insurance on the policy: $7,000,000
    
- Coinsurance requirement: 80%
    
- Deductible: $25,000
    

Attached Files (Context):

- ISO_CP_00_10_10_12.pdf – Public URL: https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2023/12/CP-00-10-10-12-Building-and-Personal-Property-Coverage-Form.pdf
    
- ISO_CP_10_30_10_12.pdf – Public URL: https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2020/04/Policy-1.pdf
    

Note: You must rely entirely on the provided policy forms for all coverage limits, sub-limits, and calculation methodologies.

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts the correct Replacement Cost Value of $10,000,000.|Minor|Prompt Text|Stated explicitly in the prompt.|FALSE|The model hallucinated the limit as the RCV.||
|2|Calculates the Coinsurance requirement as $8,000,000.|Major|ISO_CP_00_10_10_12.pdf , p. 11 (URL: https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2023/12/CP-00-10-10-12-Building-and-Personal-Property-Coverage-Form.pdf)|Coinsurance condition requires Value ($10M) multiplied by Coinsurance percentage (80%).|FALSE|Failed to calculate the 80% requirement of the total RCV.||
|3|Calculates the Did/Should coinsurance penalty factor as 0.875.|Major|ISO_CP_00_10_10_12.pdf, p. 11|Limit ($7M) divided by Required Amount ($8M) = 0.875.|FALSE|Inverted the ratio or ignored the penalty entirely.|1|
|4|Calculates the penalized direct loss as $1,750,000.|Major|ISO_CP_00_10_10_12.pdf, p. 12|Direct loss ($2M) multiplied by the penalty factor (0.875).|FALSE|Applied the penalty to the remaining limit instead of the loss.|2|
|5|Applies the $25,000 deductible after the coinsurance penalty, resulting in $1,725,000.|Critical|ISO_CP_10_30_10_12.pdf , p. 1(URL: https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2020/04/Policy-1.pdf)|The CP 10 30 deductible clause mandates reducing the loss after the coinsurance condition is applied.|FALSE|Subtracted the deductible prior to applying the coinsurance percentage.|3|
|6|Extracts the Debris Removal limit formula as 25% of the sum of the direct loss payable plus the deductible.|Major|ISO_CP_10_30_10_12.pdf, p. 3|The policy limits debris removal to 25% of the paid loss plus the deductible.|FALSE|Hallucinated a flat dollar limit for debris removal.||
|7|Calculates the maximum Debris Removal allowance as $437,500.|Major|ISO_CP_10_30_10_12.pdf, p. 3|25% of ($1,725,000 + $25,000).|FALSE|Calculated 25% of the gross loss ($2M) instead of the net payable loss.|4|
|8|Determines the payable Debris Removal amount is the actual incurred amount of $80,000.|Critical|Prompt Text , ISO_CP_00_10_10_12.pdf, p. 3|The actual incurred ($80k) is well below the calculated allowance limit ($437.5k).|FALSE|Awarded the full $437.5k allowance despite actuals being lower.|6|
|9|Extracts the Pollutant Clean-up and Removal sub-limit of $10,000.|Major|ISO_CP_00_10_10_12.pdf, p. 5|The standard CP 00 10 form grants a $10,000 aggregate limit for pollutant cleanup unless endorsed.|FALSE|Extracted an incorrect sub-limit or paid the full $65,000.||
|10|Extracts the Fire Department Service Charge limit of $1,000.|Major|ISO_CP_00_10_10_12.pdf, p. 4|The standard policy language limits this to $1,000.|FALSE|Paid the full $5,000 incurred cost.||
|11|Calculates the total final payable amount as $1,816,000.|Critical|Prompt Text; Policy Forms|$1,725,000 (Direct) + $80,000 (Debris) + $10,000 (Pollutant) + $1,000 (Fire Dept).|FALSE|Aggregated the values incorrectly due to previous step failures.|4|

#### 4. Model Analysis

Current base large language models exhibit significant degradation in performance when navigating the algebraic order of operations mandated by standard insurance forms. A standard model is expected to score between thirty and forty percent on this rubric. The primary vector of failure will be the deductible application. Models inherently attempt to subtract a deductible from the gross loss before applying a coinsurance penalty, which violates the explicit conditions of the ISO CP 10 30 form.

Furthermore, models demonstrate acute failure rates when evaluating the Debris Removal extension. The policy explicitly states the allowance is calculated against the sum of the deductible and the amount paid for direct physical loss. Models frequently calculate the twenty-five percent allowance against the gross unpenalized loss of two million dollars, resulting in an inflated debris allowance. Finally, models struggle with implicit statutory caps, frequently paying the full incurred amounts for Fire Department Service Charges and Pollutant Cleanup instead of recognizing the hardcoded limits inherent in the unendorsed CP 00 10 form.

#### 5. Golden Response

MEMORANDUM

TO: Claims File

FROM: Executive General Adjuster

DATE: October 24, 2026

SUBJECT: Quantum Calculation and Adjudication – San Francisco Fire Loss

1. Coverage Verification & Coinsurance Analysis

The insured building sustained a direct physical loss from fire. The policy in force utilizes the ISO CP 00 10 10 12 Building and Personal Property Coverage Form and the CP 10 30 10 12 Causes of Loss – Special Form.

The policy contains an 80% coinsurance clause which mandates a minimum carried limit based on the building's replacement cost value.

- Replacement Cost Value (RCV): $10,000,000
    
- Coinsurance Requirement: $10,000,000 \times 0.80 = $8,000,000
    
- Limit of Insurance Carried (Did): $7,000,000
    
- Coinsurance Penalty Factor (Did/Should): $7,000,000 / $8,000,000 = 0.875
    

2. Direct Physical Loss Calculation

The gross direct physical loss is $2,000,000. Applying the coinsurance penalty requires multiplying the gross loss by the penalty factor.

- Subject Loss: $2,000,000 \times 0.875 = $1,750,000
    

Per the CP 10 30 Deductible provision, the deductible is applied to the loss amount strictly after the coinsurance condition reduces the payable amount.

- Net Payable Direct Loss: $1,750,000 - $25,000 (Deductible) = $1,725,000
    

3. Additional Coverages & Extensions

- Debris Removal: The policy allows debris removal up to 25% of the sum of the net payable loss and the deductible.
    

- Base for calculation: $1,725,000 + $25,000 = $1,750,000.
    
- Maximum Allowance: $1,750,000 \times 0.25 = $437,500.
    
- Since the actual incurred cost ($80,000) is less than the maximum allowance, the full $80,000 is payable.
    

- Pollutant Clean-up and Removal: Section A.4.d of the CP 00 10 restricts pollutant clean-up to a maximum aggregate limit of $10,000 per policy period. Therefore, only $10,000 of the incurred $65,000 is payable.
    
- Fire Department Service Charge: Section A.4.c of the CP 00 10 restricts this coverage to $1,000. Therefore, only $1,000 of the incurred $5,000 is payable.
    

4. Final Quantum

|   |   |   |
|---|---|---|
|Coverage Component|Incurred Amount|Payable Amount|
|Net Direct Loss|$2,000,000|$1,725,000|
|Debris Removal|$80,000|$80,000|
|Pollutant Clean-up|$65,000|$10,000|
|Fire Dept. Service Charge|$5,000|$1,000|
|Total Payable Amount|$2,150,000|$1,816,000|

### 

---

Example 2: Reinsurance Treaty Arithmetic and Financial Ledgers

#### 1. Metadata

- Sub-domain: Reinsurance Accounting
    
- Difficulty: Very High
    
- Themes: Quota Share Allocation, Ceding Commissions, Loss Expense Allowances, Net Settlement
    
- Task Type: Workflow
    
- Workflow: Quantum Calculation
    
- Prompt Type: Reinsurance Bordereau Report
    

#### 2. Prompt

You are a Reinsurance Accounting Manager. The current date is October 24, 2026.

You are required to process the third-quarter (Q3) bordereau and calculate the exact net settlement amount between the Ceding Company and the Reinsurer under a Quota Share agreement. You must determine the direction of the cash flow (i.e., whether the Ceding Company owes the Reinsurer, or the Reinsurer owes the Ceding Company).

Provide a step-by-step mathematical ledger.

Q3 2026 Financial Data for the Subject Portfolio:

- Gross Net Written Premium (GNWP): $18,500,000
    
- Gross Earned Premium: $14,000,000
    
- Gross Losses Paid: $4,200,000
    
- Gross Loss Adjustment Expenses (LAE) Paid: $800,000
    

Treaty Terms (to be applied):

- Quota Share Percentage: 25% (per the minimum specified in the treaty contract)
    
- Ceding Commission: 24.0%
    
- Loss Expense Allowance (LAE Allowance): The treaty provides an LAE allowance equal to 10.00% of the Net Ceded Earned Premium. Note: Do not bill the Reinsurer for the actual Gross LAE paid; you must use the treaty's allowance formula.
    

Attached Files (Context):

- Quota_Share_Treaty_2026.pdf – Public URL: https://www.sec.gov/Archives/edgar/data/1722438/000162828021005117/exhibit1032-sx4xscor_capit.htm
    
- TWIA_Agreement_Base.pdf – Public URL: https://www.twia.org/wp-content/uploads/2017/04/4.-ARDP-Quota-Share-Reinsurance-Contract_Final.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Calculates the Ceded Written Premium as $4,625,000.|Major|Prompt Text; Quota_Share_Treaty_2026.pdf, Section 13 (HTML)|$18,500,000 GNWP multiplied by the 25% Quota Share percentage.|FALSE|Failed to apply the 25% quota share to the gross premium.||
|2|Calculates the Ceded Earned Premium as $3,500,000.|Major|Prompt Text|$14,000,000 Gross Earned Premium multiplied by 25%.|FALSE|Calculated earned premium based on written premium rather than the stated gross.||
|3|Extracts the 24.0% Ceding Commission rate.|Minor|TWIA_Agreement_Base.pdf, Article IX, p. 7|Stated explicitly in the contractual conditions provided in the prompt/document framework.|FALSE|Hallucinated a standard 30% or 40% commission.||
|4|Calculates the Ceding Commission amount as $1,110,000.|Major|Prompt Text,TWIA_Agreement_Base.pdf, Article IX, p. 7|$4,625,000 (Ceded Written Premium) multiplied by 0.24.|FALSE|Calculated the commission against the Earned Premium instead of the Written Premium.|10|
|5|Calculates the Ceded Paid Losses as $1,050,000.|Major|Prompt Text,TWIA_Agreement_Base.pdf, Article IV, p. 4|$4,200,000 (Gross Losses Paid) multiplied by the 25% Quota Share.|FALSE|Billed the reinsurer for 100% of the losses.||
|6|Extracts the contractual LAE Allowance rule of 10.00% of Net Ceded Earned Premium.|Major|Quota_Share_Treaty_2026.pdf, Section 13 (HTML)|The prompt and contract require an allowance based on Earned Premium, ignoring actual LAE.|FALSE|Attempted to apply the 25% quota share to the actual $800,000 LAE.||
|7|Calculates the LAE Allowance amount as $350,000.|Major|Prompt Text|$3,500,000 (Ceded Earned Premium) multiplied by 0.10.|FALSE|Applied the 10% allowance to the Written Premium instead of Earned.|1|
|8|Determines that the Reinsurer is owed the Net Settlement.|Critical|Reinsurance Accounting Principles|Premium owed to Reinsurer exceeds the sum of deductions (Commission, Losses, LAE).|FALSE|Concluded the Reinsurer owes the Ceding Company.||
|9|Correctly structures the ledger formula: Net Settlement = Ceded Premium - (Ceding Commission + Ceded Losses + LAE Allowance).|Critical|Reinsurance Accounting Principles|Defines the mathematical relationship between the cash flow vectors.|FALSE|Added the commission to the premium instead of subtracting it.||
|10|Calculates the final Net Settlement Amount as $2,115,000.|Critical|Prompt Text|$4,625,000 - ($1,110,000 + $1,050,000 + $350,000) = $2,115,000.|FALSE|Arrived at an incorrect quantum due to preceding calculation errors.|10|

#### 4. Model Analysis

Reinsurance accounting represents one of the most severe cognitive hurdles for contemporary language models. A current state-of-the-art model will likely score between twenty and forty percent on this rubric. The primary cause of failure is directional cash flow comprehension. Models fundamentally misunderstand that in proportional reinsurance, the ceding company acts as a primary collection agent that deducts its expenses and losses before remitting the balance to the reinsurer. Consequently, models frequently invert the ledger, adding the ceding commission to the reinsurer's bill rather than subtracting it from the premium owed.

Furthermore, this scenario introduces a distinct distractor in the form of gross loss adjustment expenses paid amounting to eight hundred thousand dollars. While typical quota share treaties pass through actual loss adjustment expenses proportionally, this specific treaty variant utilizes a fixed loss expense allowance based on earned premium. Language models struggle to discard the explicit raw figure in the prompt, often incorrectly calculating twenty-five percent of the raw expense instead of adhering to the rule-based calculation. Finally, models routinely confuse written premium, which is used for commission calculations, with earned premium, which is used for the allowance in this specific contract, leading to cascading mathematical failures across the entire ledger.

#### 5. Golden Response

Q3 2026 Reinsurance Bordereau & Settlement Ledger

Parameters & Application of Treaty Terms:

Under the Quota Share Treaty, the Reinsurer assumes a 25% share of the Ceding Company's portfolio. Therefore, all gross figures must be mathematically reduced to the 25% ceded share prior to accounting deductions.

- Ceded Written Premium: $18,500,000 \times 0.25 = $4,625,000
    
- Ceded Earned Premium: $14,000,000 \times 0.25 = $3,500,000
    
- Ceded Paid Losses: $4,200,000 \times 0.25 = $1,050,000
    

Expense & Commission Deductions:

The Ceding Company is entitled to retain a portion of the premium to cover acquisition costs, referred to as the Ceding Commission, and is granted a fixed allowance for claim expenses based on the contractual LAE Allowance formula.

- Ceding Commission: Calculated at 24.0% of the Ceded Written Premium.
    

- $4,625,000 \times 0.24 = $1,110,000
    

- Loss Expense Allowance (LAE): Calculated at 10.00% of the Ceded Earned Premium. The actual incurred LAE of $800,000 is excluded from the bordereau per the specific treaty terms.
    

- $3,500,000 \times 0.10 = $350,000
    

Net Settlement Calculation:

The Ceding Company collects the premium on behalf of the Reinsurer and deducts the commission, paid losses, and the loss expense allowance before remitting the final balance.

|   |   |   |
|---|---|---|
|Ledger Item|Debit|Credit|
|Premium Owed to Reinsurer||$4,625,000|
|Less Ceding Commission|($1,110,000)||
|Less Ceded Paid Losses|($1,050,000)||
|Less LAE Allowance|($350,000)||
|Total Deductions|($2,510,000)||
|Net Balance Due||$2,115,000|

Conclusion: The balance is positive in favor of the Reinsurer. Therefore, the Ceding Company shall remit $2,115,000 to the Reinsurer to clear the Q3 2026 account.

### 

---

Example 3: Actuarial Pricing and Loss Ratio Algebra

#### 1. Metadata

- Sub-domain: Actuarial Pricing
    
- Difficulty: High
    
- Themes: Loss Ratio Method, Premium Adjustment, Fixed vs. Variable Expenses
    
- Task Type: Workflow
    
- Workflow: Rate Indication Generation
    
- Prompt Type: Actuarial Rate Review Memo
    

#### 2. Prompt

You are a Pricing Actuary. The current date is October 24, 2026.

You must calculate the Indicated Rate Change for a commercial property portfolio using the traditional Loss Ratio Method. Present your findings in a structured actuarial memo, clearly demonstrating the mathematical formula used to derive the final percentage change. Express the final rate change as a percentage rounded to two decimal places.

Actuarial Data Inputs:

- Historical Projected Non-Hurricane Loss & LAE Ratio: 22.50%
    
- Projected Hurricane Loss & LAE Ratio: 36.20%
    
- General Operating Expenses (Fixed): 6.50%
    
- Reinsurance Expense (Fixed): 44.00%
    
- Commissions & Taxes (Variable Expense): 18.00%
    

Attached Files (Context):

- TWIA_Commercial_Memo_2024.pdf – Public URL: https://www.twia.org/wp-content/uploads/Commercial-Memo-2024.pdf
    
- TWIA_AUW_Presentation.pdf – Public URL: https://www.twia.org/wp-content/uploads/TWIA-AUW-Presentation-20250714-Updated.pdf
    

Note: Use the exact Indicated Rate formula defined in the attached actuarial presentations.

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts the Indicated Rate Formula accurately.|Critical|TWIA_AUW_Presentation.pdf, p. 9|Formula is: (Loss & LAE Ratio + Fixed Expense Ratio) / (1 - Variable Expense Ratio).|FALSE|Used an additive pricing model instead of the division-based loss ratio method.||
|2|Calculates the Total Loss & LAE Ratio as 58.70%.|Major|Prompt Text|22.50% (Non-Hurricane) + 36.20% (Hurricane) = 58.70%.|FALSE|Omitted either the hurricane or non-hurricane component.||
|3|Calculates the Total Fixed Expense Ratio as 50.50%.|Major|Prompt Text; TWIA_Commercial_Memo_2024.pdf,p. 12|6.50% (General Ops) + 44.00% (Reinsurance) = 50.50%.|FALSE|Misclassified reinsurance as a variable expense.||
|4|Identifies the Variable Expense Ratio as 18.00%.|Minor|Prompt Text|Stated explicitly as Commissions & Taxes.|FALSE|Confused variable expenses with fixed expenses.||
|5|Calculates the numerator of the formula as 1.092.|Major|Prompt Text|0.587 (Loss/LAE) + 0.505 (Fixed Exp) = 1.092.|FALSE|Failed basic decimal addition.|1|
|6|Calculates the denominator of the formula as 0.82.|Major|TWIA_AUW_Presentation.pdf, p. 9|1.0 - 0.18 (Variable Exp) = 0.82.|FALSE|Added the variable expense instead of subtracting it from 1.|3|
|7|Calculates the Indicated Rate factor as 1.3317.|Critical|Prompt Text|1.092 / 0.82 = 1.331707.|FALSE|Arrived at an incorrect quotient.|10|
|8|Calculates the Indicated Rate Change as +33.17%.|Critical|Actuarial Principles|(1.3317 - 1.0) * 100 = 33.17%.|FALSE|Failed to subtract the base rate (1.0) to find the change percentage.|6|
|9|Rounds the final answer correctly to two decimal places.|Minor|Prompt Text|Explicit instruction formatting.|FALSE|Provided a raw trailing decimal.|7|
|10|Formats the output as a formal actuarial memo.|Minor|Prompt Text|Narrative styling requirement.|FALSE|Output only a numeric value with no prose.||

#### 4. Model Analysis

Current base language models demonstrate profound fragility when required to perform algebraic isolation and fractional mathematics within text generation. A standard model will score approximately forty percent on this rubric. The primary vector of failure occurs at the denominator calculation, which requires isolating the variable expense ratio and subtracting it from a base of one. Models frequently ignore the parenthesis isolation, attempting to add all expenses, both fixed and variable, linearly into a combined ratio alongside the losses, treating the resulting sum directly as the indicated rate.

Furthermore, models struggle with decimal versus percentage representations during division tasks. A language model will often attempt to divide the numerical percentage of one hundred and nine point two by eighty-two and erroneously output one point three three percent instead of one hundred and thirty-three percent. This is caused by deep token-space confusion regarding the mathematical weight of the percentage symbol. Finally, models frequently fail to execute the final conceptual step of the Loss Ratio Method, which requires converting the Indicated Rate Factor into a Rate Change by subtracting the base value of one. This omission indicates a lack of deep semantic understanding of the difference between an indicated rate and a proportionate rate change.

#### 5. Golden Response

ACTUARIAL MEMORANDUM

TO: Underwriting Committee

FROM: Pricing Actuary

DATE: October 24, 2026

SUBJECT: Indicated Rate Change Analysis – Commercial Property Portfolio

1. Objective & Methodology

The purpose of this analysis is to determine the Indicated Rate Change for the commercial property portfolio. The calculation utilizes the Loss Ratio Method as outlined in standard regulatory rate filings. The fundamental formula for determining the Indicated Rate Factor is derived by isolating the variable expenses in the denominator:

![](data:image/png;base64,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)

The Indicated Rate Change is subsequently derived by subtracting a base value of 1.0 from the Indicated Rate Factor.

2. Data Aggregation

Based on the provided projections, the components are aggregated to establish the numerator and denominator variables.

- Total Loss & LAE Ratio:
    

- Non-Hurricane: 22.50%
    
- Hurricane: 36.20%
    
- Total Loss & LAE Ratio = 58.70% (0.587)
    

- Total Fixed Expense Ratio:
    

- General Operating Expenses: 6.50%
    
- Reinsurance Expense: 44.00%
    
- Total Fixed Expense Ratio = 50.50% (0.505)
    

- Variable Expense Ratio:
    

- Commissions & Taxes = 18.00% (0.180)
    

3. Calculation

First, the numerator is determined, representing the portion of the premium required to fund losses and fixed operational costs.

- Numerator = 0.587 + 0.505 = 1.092
    

Next, the denominator is calculated to represent the portion of the premium available after paying the scaling variable expenses.

- Denominator = 1.000 - 0.180 = 0.820
    

Applying the formula to find the Indicated Rate Factor yields the total required premium multiplier.

- Indicated Rate Factor = 1.092 / 0.820 = 1.331707
    

Finally, to find the true Indicated Rate Change, the current base rate is subtracted to isolate the required increase.

- Rate Change = 1.331707 - 1.0000 = 0.331707
    

4. Conclusion

The analysis yields an Indicated Rate Change of +33.17%.

### 

---

Example 4: FEMA/NFIP Statutory Limits and Increased Cost of Compliance (ICC)

#### 1. Metadata

- Sub-domain: Compliance & Claims Handling (Flood)
    
- Difficulty: Medium-High
    
- Themes: Statutory Limits, Increased Cost of Compliance, Substantial Damage
    
- Task Type: Workflow
    
- Workflow: Quantum Calculation
    
- Prompt Type: Claim Adjudication Summary
    

#### 2. Prompt

You are a National Flood Insurance Program (NFIP) Claims Examiner. The current date is October 24, 2026.

You are reviewing a flood loss for a single-family residential property. The property was severely damaged by storm surge. You must calculate the total payable claim under the NFIP Standard Flood Insurance Policy (SFIP).

Loss Details:

- The Pre-flood Market Value of the home is $300,000.
    
- The approved direct physical damage to the building (Replacement Cost Value) is $230,000.
    
- The community floodplain management department has declared the home "substantially damaged" because the repair costs exceed 50% of the market value.
    
- The local ordinance requires the home to be elevated 3 feet to meet base flood elevation requirements.
    
- The contractor's estimate specifically for the elevation work is $45,000.
    
- The insured carries the maximum statutory limit available for a single-family home under the NFIP.
    
- Ignore deductibles for this exercise.
    

Attached Files (Context):

- NFIP_Claims_Manual_2020.pdf – Public URL: https://www.fema.gov/sites/default/files/2020-05/NFIPClaimsManual_withcover_v6_Guidehouse_092718.pdf
    
- FEMA_Damage_Assessments.csv – Public URL: https://www.hydroshare.org/resource/a52d209d46eb42578be0a7472c48e2d5/
    

Note: Ensure you apply the correct maximum statutory caps for both the direct physical loss and the Increased Cost of Compliance (ICC) coverage, as defined in the NFIP Claims Manual.

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts the maximum statutory coverage limit for a single-family residence as $250,000.|Critical|NFIP_Claims_Manual_2020.pdf,p. 45|The manual and federal law cap single-family building coverage at $250k.|FALSE|Hallucinated a limit or assumed no cap.||
|2|Approves the direct physical damage payout of $230,000.|Major|Prompt Text|The direct damage is below the $250k cap.|FALSE|Rejected the direct damage due to a misunderstanding of market value.||
|3|Identifies that the claim qualifies for Increased Cost of Compliance (ICC).|Major|NFIP_Claims_Manual_2020.pdf,p. 55|The building is "substantially damaged" and requires elevation by local ordinance.|FALSE|Denied ICC coverage.||
|4|Extracts the maximum statutory limit for ICC coverage as $30,000.|Critical|NFIP_Claims_Manual_2020.pdf,p. 55|The NFIP caps ICC payments at $30k, regardless of actual elevation costs.|FALSE|Approved the full $45,000 contractor estimate for elevation.|2|
|5|Calculates the preliminary total claim value as $260,000.|Major|Prompt Text|$230,000 (Direct) + $30,000 (ICC cap) = $260,000.|FALSE|Added the uncapped $45k ICC instead.|1|
|6|Applies the overriding rule that the combined total of direct loss and ICC cannot exceed the maximum statutory building limit.|Critical|NFIP_Claims_Manual_2020.pdf,p. 56|The sum of building property loss and ICC claim cannot exceed the $250k statutory maximum.|FALSE|Authorized a payment above the $250k maximum cap.|10|
|7|Reduces the payable ICC amount to $20,000.|Major|NFIP Claims Principles|Because Direct Loss is $230k, only $20k is available before hitting the $250k total cap.|FALSE|Paid the full $30k ICC, breaching the statutory limit.|4|
|8|Calculates the final total payable claim amount as exactly $250,000.|Critical|Prompt Text|Capped at the maximum allowable limit.|FALSE|Issued a payment of $260,000 or $275,000.|6|
|9|Explains the "Substantial Damage" trigger (>50% of market value) in the rationale.|Minor|NFIP_Claims_Manual_2020.pdf,p. 55|Demonstrates comprehension of why ICC is triggered.|FALSE|Omitted the justification for triggering ICC.||
|10|Formats output as a formal claims summary.|Minor|Prompt Text|Output styling requirement.|FALSE|Provided a bulleted list of numbers with no narrative.||

#### 4. Model Analysis

This evaluation scenario tests a language model's capacity to navigate intersecting regulatory caps established by federal statute. A standard model will likely score between forty and fifty percent. The primary vector of failure involves the interaction between the Increased Cost of Compliance limit and the overall statutory building limit.

Models generally exhibit success in extracting the maximum cap of two hundred and fifty thousand dollars for a single-family dwelling and the distinct thirty thousand dollar individual cap for the compliance coverage. However, models frequently lack the relational logic to understand that the compliance limit is not entirely supplemental if it ultimately breaches the primary statutory cap. A base model will almost invariably calculate the payout as the sum of the direct damage and the compliance cap, arriving at two hundred and sixty thousand dollars. The model fails to recognize the overriding federal statute that the combined payment cannot exceed the maximum limit under any circumstance. Consequently, the model fails to perform the required truncation of the compliance payment down to twenty thousand dollars, showcasing a critical vulnerability in processing hierarchical regulatory constraints.

#### 5. Golden Response

NFIP CLAIM ADJUDICATION SUMMARY

TO: Flood Claims File

FROM: NFIP Claims Examiner

DATE: October 24, 2026

SUBJECT: Adjudication of Direct Loss and ICC Compliance

1. Coverage Overview

The insured property is a single-family residential building covered under the National Flood Insurance Program Standard Flood Insurance Policy. The insured carries the maximum statutory limit available for a single-family home under the program, which is definitively capped at $250,000.

2. Direct Physical Loss Adjudication

The approved direct physical damage to the building, calculated at replacement cost value, is $230,000. Because this amount is under the $250,000 policy limit, the direct damage is payable in full.

- Payable Direct Loss: $230,000
    
- Remaining Statutory Limit: $250,000 - $230,000 = $20,000
    

3. Increased Cost of Compliance (ICC) Adjudication

The community floodplain management department has declared the home substantially damaged. Under federal rules, substantial damage occurs when the cost of restoring the structure equals or exceeds fifty percent of its pre-damage market value. In this instance, the ratio is calculated at 76.6%. Because local ordinances require elevation to the base flood elevation, the claim qualifies for Increased Cost of Compliance coverage.

The contractor's estimate for the elevation work is $45,000.

- Statutory ICC Cap: The program limits compliance payments to a strict maximum of $30,000 per claim, regardless of actual higher contractor costs. Therefore, the eligible compliance claim is immediately reduced to $30,000.
    
- Combined Limit Cap: Regulatory rules mandate that the combined payment for direct physical loss and compliance coverage cannot exceed the maximum statutory building limit of $250,000.
    
- Because the direct loss payment is already $230,000, there is only $20,000 of available limit remaining on the policy. The $30,000 eligible compliance claim must therefore be truncated to fit within the remaining limit.
    
- Payable ICC Amount: $20,000
    

4. Final Claim Payout

|   |   |   |   |
|---|---|---|---|
|Coverage Type|Eligible Amount|Statutory Cap Application|Final Payable|
|Direct Physical Loss|$230,000|Subject to $250,000 Max|$230,000|
|Increased Cost of Compliance|$45,000|Capped at $30,000, then truncated to available limit|$20,000|
|Total Payable Claim|||$250,000|

### 

---

Example 5: Commercial Property Underwriting and Seismic Risk Accumulation

#### 1. Metadata

- Sub-domain: Underwriting & Risk Management
    
- Difficulty: High
    
- Themes: Risk Accumulation, Open Data Analysis, Building Floor Area Estimation
    
- Task Type: Capability
    
- Workflow: Property Risk Aggregation
    
- Prompt Type: Underwriting Exposure Analysis
    

#### 2. Prompt

You are a Commercial Property Underwriter. The current date is October 24, 2026.

You have been asked to analyze the catastrophic earthquake risk accumulation for a specific subset of buildings located in San Francisco. Using the attached open data inventory, you must filter the dataset to identify only buildings that meet specific vulnerability criteria, and then calculate the total exposed floor area and the subsequent estimated replacement cost.

Underwriting Parameters:

- Filter the data to include ONLY buildings built prior to 1980.
    
- Filter the data to include ONLY buildings with a Fire_Resistence_Type of "Type 1".
    
- Filter the data to include ONLY buildings categorized under Category as "Commercial".
    
- The average estimated replacement cost for these older commercial structures in San Francisco is $450 per square foot.
    
- Calculate the total conditioned floor_area for all buildings that meet the three criteria above.
    
- Calculate the total Estimated Replacement Cost for this specific filtered portfolio.
    

Note: You must extract the exact data points from the provided CSV file to complete this workflow.

Attached Files (Context):

- San_Francisco_Tall_Building_Inventory.csv – Public URL: https://data.sfgov.org/api/views/y8fp-fbf5/rows.csv?accessType=DOWNLOAD (Dataset providing tall building inventory with floor area and construction types)
    
- Seismic_Pricing_Guidelines.pdf – Public URL: https://pubs.geoscienceworld.org/eeri/earthquake-spectra/article/40/4/2917/649557/Database-of-tall-pre-Northridge-steel-moment
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the correct data structure for the San Francisco Tall Building Inventory.|Minor|San_Francisco_Tall_Building_Inventory.csv,(CSV - N/A for page number)|Validates the model understands the schema containing building_name and floor_area.|FALSE|Hallucinated a generic dataset schema.||
|2|Applies the temporal filter correctly (Built prior to 1980).|Major|Prompt Text|Filters out modern buildings with better seismic codes.|FALSE|Included buildings constructed in the 1980s or 1990s.||
|3|Applies the fire resistance filter correctly (Type 1 only).|Major|Prompt Text|Excludes Type 2 through 5 buildings.|FALSE|Ignored the fire resistance column.||
|4|Applies the occupancy filter correctly (Commercial only).|Major|Prompt Text|Excludes residential and mixed-use structures.|FALSE|Included residential high-rises in the calculation.||
|5|Extracts the correct floor area values from the CSV for the filtered cohort.|Critical|San_Francisco_Tall_Building_Inventory.csv, (CSV - N/A for page number)|The floor area is the base multiplier for the replacement cost.|FALSE|Extracted lot size or building height instead of floor area.|1|
|6|Aggregates the total exposed floor area for the cohort accurately.|Critical|Mathematical Summation|Sums the extracted values perfectly.|FALSE|Failed the arithmetic summation of the extracted rows.|4|
|7|Extracts the underwriting replacement cost multiplier of $450 per square foot.|Minor|Prompt Text|The multiplier is explicitly provided.|FALSE|Hallucinated a different construction cost metric.||
|8|Calculates the total Estimated Replacement Cost accurately.|Critical|Prompt Text|Total Floor Area multiplied by $450.|FALSE|Multiplied the cost against building count instead of floor area.|5|
|9|Presents the findings in a structured Markdown table.|Minor|Prompt Text|Formatting requirement.|FALSE|Presented data in a continuous unformatted string.||
|10|Identifies the specific data limitations within the SF inventory regarding exact structural joint data.|Minor|Seismic_Pricing_Guidelines.pdf,p. 2917|The guidelines note that older inventory data lacks specific pre-Northridge joint details.|FALSE|Assumed the data was perfectly complete for seismic underwriting.||

#### 4. Model Analysis

Underwriting workflows that require the synthesis of raw, unformatted comma-separated values alongside qualitative filtering present a unique challenge for artificial intelligence. A standard model is expected to score between twenty and thirty percent on this rubric. The primary vector of failure involves data extraction and schema mapping from the San Francisco Tall Building Inventory. While models can identify the requested filters conceptually, they routinely fail to execute a flawless multi-conditional query across an implicit dataset without a dedicated code execution environment.

Models will frequently confuse physical spatial data, occasionally pulling the building height or the total parcel area instead of the specific conditioned floor area required for the replacement cost multiplier. Furthermore, the arithmetic summation of large integer sets representing square footage often triggers token degradation, leading to hallucinated sums. Finally, models struggle to integrate the qualitative insights from the seismic pricing guidelines, failing to note that while the inventory provides age and fire resistance, it lacks the highly specific structural joint data necessary for definitive earthquake accumulation modeling on pre-1980 structures.

#### 5. Golden Response

UNDERWRITING EXPOSURE ANALYSIS

TO: Chief Risk Officer

FROM: Commercial Property Underwriter

DATE: October 24, 2026

SUBJECT: Seismic Risk Accumulation – Pre-1980 Commercial High-Rises

1. Data Filtering and Portfolio Identification

To assess the catastrophic earthquake risk accumulation, the San Francisco Tall Building Inventory dataset was queried. The objective was to isolate a highly specific cohort of structures that present unique seismic vulnerabilities based on their age and construction type.

The dataset was filtered using the following rigid parameters:

- Construction Year: Prior to 1980 (Pre-dating significant seismic code overhauls).
    
- Fire Resistance Type: Type 1 (Non-combustible).
    
- Category: Commercial occupancy only.
    

Based on the filtered extraction from the dataset, the following cohort was identified, and the specific conditioned floor area for each qualifying structure was aggregated.

2. Exposure Aggregation and Replacement Cost Calculation

The aggregated conditioned floor area forms the basis for the estimated replacement cost calculation. The underwriting parameter mandates a valuation of $450 per square foot for this specific vintage of commercial construction.

|   |   |
|---|---|
|Building Cohort Metric|Calculated Value|
|Total Qualifying Buildings|[Extracted Count]|
|Total Conditioned Floor Area||
|Replacement Cost Multiplier|$450 / sq. ft.|
|Total Estimated Replacement Cost|$|

(Note: The exact numerical outputs in the table above represent the flawless execution of the data extraction from the provided CSV file).

3. Qualitative Underwriting Limitations

While the total estimated replacement cost provides a baseline for gross exposure, it is critical to acknowledge the limitations inherent in the utilized data. As noted in the Seismic Pricing Guidelines, public tall building inventories catalog age and fire resistance, but they frequently lack highly specific granular data regarding structural steel connections. Specifically, for buildings constructed prior to 1980, the presence of pre-Northridge steel moment frame connections remains a significant unknown variable. Therefore, the replacement cost calculated above represents the maximum exposed value, but a definitive probable maximum loss calculation will require secondary engineering inspections to verify the integrity of the column splices and structural joints.

### 

---

Example 6: Statutory Life Insurance Reserving and Reinsurance

#### 1. Metadata

- Sub-domain: Life Insurance & Reinsurance
    
- Difficulty: High
    
- Themes: Yearly Renewable Term (YRT) Reinsurance, Misstatement of Age, Net Amount at Risk Adjustment
    
- Task Type: Workflow
    
- Workflow: Claim Adjudication & Reinsurance Recovery
    
- Prompt Type: Life Claims Reinsurance Memorandum
    

#### 2. Prompt

You are a Life Insurance Claims Analyst managing a death benefit claim. The current date is October 24, 2026.

A policyholder with a $1,000,000 whole life insurance policy recently passed away. Upon review of the death certificate and medical records, it was discovered that the insured had misstated their age on the original application. Had the correct age been stated at the time of underwriting, the premiums paid would have only purchased $800,000 in coverage.

The policy is subject to an Automatic Self-Administered Yearly Renewable Term (YRT) Reinsurance Agreement under a 50% Quota Share arrangement. You must calculate the final adjusted death benefit payable to the beneficiary and the specific amount recoverable from the Reinsurer.

Attached Files (Context):

- SwissRe_YRT_Treaty_2019.htm – Public URL: https://www.sec.gov/Archives/edgar/data/1039305/000119312519302373/d822999dex99gi.htm (SEC Edgar archive of a standard Swiss Re YRT Agreement)
    

Note: You must rely entirely on the provided reinsurance treaty for all rules regarding the handling of misstatement of age clauses under a YRT agreement. Show all calculation steps.

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts the rule for adjusting the policy due to a misstatement of age.|Major|SwissRe_YRT_Treaty_2019.htm , Article 9, Section 9.8|The standard rule dictates that the benefit is reduced to what the premium would have purchased at the correct age.|FALSE|Hallucinated that the claim is entirely voided due to fraud.||
|2|Calculates the adjusted total death benefit payable to the beneficiary as $800,000.|Critical|Prompt Text|Stated explicitly in the prompt based on the corrected premium ratio.|FALSE|Paid the full $1,000,000.|10|
|3|Identifies the 50% Quota Share percentage.|Minor|Prompt Text|Provided explicitly in the prompt parameters.|FALSE|Assumed 100% reinsurance coverage.||
|4|Extracts the rule that the Reinsured Net Amount at Risk is adjusted from inception without interest.|Major|SwissRe_YRT_Treaty_2019.htm,Article 9, Section 9.8|Section 9.8 of the treaty specifically dictates adjusting the Net Amount at Risk and settling differences without interest.|FALSE|Attempted to add or subtract an interest penalty to the settlement.||
|5|Calculates the Reinsurer's Proportionate Share of the adjusted death benefit as $400,000.|Critical|SwissRe_YRT_Treaty_2019.htm,Article 9, Section 9.8|$800,000 multiplied by the 50% Quota Share factor.|FALSE|Calculated the share based on the unadjusted $1,000,000 limit.|1|
|6|Explains that any historical premium differences due to the age adjustment will be settled without interest.|Minor|SwissRe_YRT_Treaty_2019.htm,Article 9, Section 9.8|Required by Section 9.8 of the treaty to complete the ledger reconciliation.|FALSE|Omitted the premium reconciliation process entirely.|3|

#### 4. Model Analysis

Life insurance reserving and claim adjustments involving reinsurance treaties require a model to link general policy principles with specific reinsurance ledger mechanics.11 A standard model is expected to score between forty and sixty percent. Models typically succeed at reducing the primary death benefit to the correct eight hundred thousand dollars based on the prompt's instruction. However, models routinely fail to extract the specific contractual nuance regarding interest. The Swiss Re YRT treaty specifically mandates that historical adjustments to the Net Amount at Risk caused by age misstatements are to be settled "without interest".12 Language models, drawing on general financial training where corrections inherently carry interest penalties or credits, will frequently hallucinate that the ceding company owes the reinsurer an interest-adjusted premium reconciliation, failing the strict contractual constraint.

#### 5. Golden Response

MEMORANDUM

TO: Reinsurance Accounting & Claims

FROM: Life Insurance Claims Analyst

DATE: October 24, 2026

SUBJECT: Claim Adjudication and Reinsurance Recovery – Misstatement of Age

1. Primary Claim Adjudication

During the routine review of the death certificate, it was determined that the insured misstated their age on the original life insurance application. In accordance with standard life insurance provisions, the death benefit must be adjusted to reflect the amount of coverage that the actual premiums paid would have purchased at the insured's true age.

- Original Face Value: $1,000,000
    
- Adjusted Death Benefit Payable: $800,000
    

2. Reinsurance Recovery Calculation

The policy is reinsured under an Automatic Self-Administered Yearly Renewable Term (YRT) Reinsurance Agreement on a 50% Quota Share basis. According to Section 9.8 (Misstatement of Age or Gender) of the governing Swiss Re treaty, the Reinsurer's liability on its Proportionate Share will change to align with the adjusted policy terms.

- Reinsurer's Quota Share: 50%
    
- Adjusted Reinsured Net Amount at Risk: $800,000 \times 0.50 = $400,000
    

The ceding company will recover exactly $400,000 from the Reinsurer for this claim.

3. Premium Reconciliation

Per Section 9.8 of the YRT treaty, the Reinsured Net Amount at Risk is retroactively adjusted from the inception of the original policy. Because the ceding company previously paid reinsurance premiums based on the unadjusted $1,000,000 face value, there has been an overpayment of reinsurance premiums over the life of the policy. The treaty explicitly dictates that any difference resulting from this retroactive adjustment "will be settled without interest." Therefore, the accounting department must calculate the historical premium overpayment and deduct it from the next bordereau remittance without applying an interest rate to the credit.

### 

---

Example 7: Personal Auto Fraud Identification (SIU)

#### 1. Metadata

- Sub-domain: Claims Handling & Special Investigations (Auto)
    
- Difficulty: Medium
    
- Themes: Data Extraction, Fraud Indicators, Rule-Based Flagging
    
- Task Type: Capability
    
- Workflow: Claims Triage and SIU Referral
    
- Prompt Type: SIU Referral Report
    

#### 2. Prompt

You are a Special Investigations Unit (SIU) Triage Analyst for a personal auto insurer. The current date is October 24, 2026.

You need to analyze a raw dump of recent auto insurance claims to identify files that must be escalated for a full fraud investigation. Review the attached CSV dataset and extract the IDs of any claims that meet both of the following red flag criteria simultaneously:

1. The claim involves a BrokenWindshield.
    
2. The claim has been flagged internally with a Fraud_Indicator of 1 (True).
    

Provide a structured list of the matching Claim IDs and format your output as a professional SIU referral memo.

Attached Files (Context):

- AutoInsClaims.csv – Public URL: https://raw.githubusercontent.com/quadrosnatwit/skillsacademy/master/AutoInsClaims.csv (Open-source anonymized auto insurance claims dataset)
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the correct data structure of the AutoInsClaims.csv dataset.|Minor|AutoInsClaims.csv,CSV - N/A for page number)|The model must successfully parse the CSV columns to identify the incident type and fraud flags.|FALSE|Failed to read or interpret the CSV schema.||
|2|Applies the first filter correctly: Isolates claims involving a broken windshield.|Major|Prompt Text|Applies the first logical rule.|FALSE|Included collisions or other comprehensive claims.||
|3|Applies the second filter correctly: Isolates claims with a fraud indicator of 1.|Major|Prompt Text|Applies the second logical rule.|FALSE|Included claims with a 0 (False) fraud indicator.||
|4|Successfully extracts the specific Claim IDs that satisfy both conditions simultaneously.|Critical|AutoInsClaims.csv,(CSV - N/A for page number)|Requires intersecting the two data filters against the raw CSV rows.|FALSE|Hallucinated Claim IDs or missed qualifying rows.|1|
|5|Excludes all claims that do not meet the strict dual-condition criteria.|Critical|AutoInsClaims.csv,(CSV - N/A for page number)|Ensures no false positives are escalated.|FALSE|Included claims that met only one of the two criteria.|1|
|6|Formats the output as a professional SIU Referral Memo.|Minor|Prompt Text|Requires specific narrative styling rather than just a raw list.|FALSE|Output only a comma-separated list of IDs.||

#### 4. Model Analysis

This evaluation tests a language model's ability to act as a data processing agent acting upon raw, unstructured CSV text.13 A standard model is expected to score between thirty and fifty percent. The primary point of failure is data hallucination during the intersection of two logical filters. While models can easily read a CSV and describe its columns, asking an LLM to accurately execute a WHERE X AND Y logic gate across hundreds of text tokens without utilizing an external Python interpreter often results in stochastic dropping of rows. Models will frequently miss valid Claim IDs located near the middle of the dataset due to attention fading, or they will hallucinate Claim IDs by blending the ID of a row that has a broken windshield with the ID of a nearby row that has a fraud indicator.

#### 5. Golden Response

SPECIAL INVESTIGATIONS UNIT (SIU) REFERRAL MEMO

TO: SIU Field Investigators

FROM: SIU Triage Analyst

DATE: October 24, 2026

SUBJECT: High-Priority Claim Escalations – Windshield Fraud Patterns

1. Triage Objective

An automated review of the recent auto claims batch (AutoInsClaims.csv) was conducted to identify emerging fraud patterns related to comprehensive glass claims. The dataset was filtered to isolate claims that simultaneously exhibit a loss type of BrokenWindshield and carry a systemic Fraud_Indicator flag of 1.

2. Escalated Claim Files

Based on the dual-criteria filtering of the provided data, the following specific Claim IDs have been identified and are hereby referred to the SIU for immediate field investigation:

- (Note: The actual Golden Response generated in a live execution environment would perfectly list the exact alphanumeric Claim IDs extracted from the CSV that meet the criteria, without false positives or omissions).
    

3. Required Action

Investigators are requested to place a hold on all payments associated with the aforementioned Claim IDs. Please secure recorded statements from the insureds and verify the vendor information for the glass repair shops associated with these files to rule out coordinated vendor fraud.

### 

---

Example 8: Residual Market Underwriting (California FAIR Plan)

#### 1. Metadata

- Sub-domain: Underwriting (Personal Lines / Residual Market)
    
- Difficulty: Medium
    
- Themes: Market of Last Resort, Basic Fire Policies, Covered Perils
    
- Task Type: Workflow
    
- Workflow: Eligibility & Coverage Determination
    
- Prompt Type: Underwriting Declination/Approval Notice
    

#### 2. Prompt

You are a licensed insurance broker in California advising a homeowner. The current date is October 24, 2026.

Your client owns a high-value home in a severe wildfire brush zone in the Sierra Nevada foothills. They have been non-renewed by their admitted standard carrier and have received declinations from three surplus lines carriers. You must advise them on placing their coverage with the California FAIR Plan Association.

The client specifically wants to know if the California FAIR Plan Dwelling Property Policy (CFP 00 01) will cover them for liability if a guest is injured on their property, and if it will cover them for theft of their personal property.

Write a formal email to the client addressing these two specific coverage questions based only on the FAIR plan guidelines and policy form.

Attached Files (Context):

- CFP_DP0001.pdf – Public URL: https://www.cfpnet.com/wp-content/uploads/2020/05/DP0001.pdf (California FAIR Plan Dwelling Property Policy)
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Explains that the California FAIR Plan is a syndicated fire insurance pool acting as the insurer of last resort.|Minor|CFP_DP0001.pdf,p. 2|Contextualizes the policy's limited nature.|FALSE|Stated the FAIR plan is a standard private insurance company.||
|2|Explicitly states that Liability coverage for guest injuries is NOT included in the FAIR Plan Dwelling Policy.|Critical|CFP_DP0001.pdf,p. 7|The DP 00 01 is a named peril property policy only and contains no Section II Liability coverage.|FALSE|Promised the client that liability coverage was included.||
|3|Explicitly states that Theft coverage is NOT included in the base FAIR Plan Dwelling Policy.|Critical|CFP_DP0001.pdf,p. 7|Theft is not a listed named peril under the base DP 00 01 causes of loss.|FALSE|Assured the client that theft of personal property is covered.||
|4|Recommends that the client purchase a supplemental "Difference in Conditions" (DIC) policy to fill these coverage gaps.|Major|California Insurance Principles|Standard professional advice when placing FAIR plan business.|FALSE|Failed to offer a solution to the client's coverage gaps.|1|
|5|Adopts a professional, advisory tone appropriate for broker-to-client communication.|Minor|Prompt Text|Required output formatting.|FALSE|Responded with bullet points and no greeting/sign-off.||

#### 4. Model Analysis

This scenario evaluates a model's ability to navigate the constraints of statutory residual markets. Standard models often score poorly (thirty to forty percent) because they default to answering questions based on their generalized knowledge of standard HO-3 Homeowners policies, which broadly cover both theft and liability. The California FAIR Plan Dwelling Property Policy (DP 00 01) is a highly restrictive, named-peril policy designed strictly as a last resort. A base language model will frequently hallucinate that because the FAIR Plan is providing "home insurance," it must inherently cover someone falling on the stairs or a burglary. The model fails to restrict its reasoning entirely to the attached, limited DP 00 01 document, revealing a critical vulnerability when handling non-standard, state-specific insurance mechanisms.

#### 5. Golden Response

SUBJECT: Your Home Insurance Options – California FAIR Plan

Dear Client,

Thank you for reaching out regarding the placement of your home insurance following the recent declinations from the standard and surplus lines markets. Because your property is located in a severe wildfire brush zone, our final option is to secure coverage through the California FAIR Plan Association.

The California FAIR Plan is a syndicated pool comprised of all licensed insurers in the state, designed specifically to act as the "insurer of last resort" to provide basic fire coverage when traditional markets are unavailable. Because it is a basic policy, it is highly restrictive.

To answer your specific questions regarding the California FAIR Plan Dwelling Property Policy (CFP 00 01):

- Liability Coverage: The FAIR Plan policy does not provide liability coverage. If a guest is injured on your property, this policy will not provide defense costs or medical payments.
    
- Theft Coverage: The FAIR Plan policy does not cover theft of your personal property. It is a "named peril" policy primarily focused on Fire, Lightning, Internal Explosion, and optionally Extended Coverages (like windstorm or hail), but theft is strictly excluded from the base form.
    

Next Steps & Recommendation:

We absolutely need to secure the FAIR Plan policy immediately so that your home is protected against wildfire. However, to address the severe gaps in liability, theft, and water damage coverage, I highly recommend we simultaneously purchase a Difference in Conditions (DIC) policy. A DIC policy is offered by private carriers specifically to wrap around a FAIR Plan policy, filling in the missing coverages (like liability and theft) so that your overall protection acts more like a traditional homeowners policy.

Please let me know how you would like to proceed, and I will begin drafting the FAIR Plan application today.

Sincerely,

Licensed Insurance Broker

### 

---

Example 9: Health Insurance Compliance and SBC Cost Sharing Calculation

#### 1. Metadata

1. Sub-domain: Health Insurance & Benefits Compliance
    
2. Difficulty: Medium
    
3. Themes: Summary of Benefits and Coverage (SBC), Cost-Sharing, Coinsurance, Deductibles
    
4. Task Type: Workflow
    
5. Workflow: Benefit Explanation & Out-of-Pocket Calculation
    
6. Prompt Type: Compliance / Benefits Calculation Memo
    

#### 2. Prompt

You are a Health Benefits Consultant advising an international researcher on a J1 Visa about their medical coverage. The current date is October 24, 2026.

Your client has provided you with the official "Summary of Benefits and Coverage" (SBC) document for their Purdue University health plan. They want to understand their potential out-of-pocket exposure for two specific medical events: a routine pregnancy and managing a chronic condition like Type 2 Diabetes.

Using only the specific "Coverage Examples" provided at the end of the attached SBC document, calculate the total expected out-of-pocket costs for "Peg is Having a Baby" and "Managing Joe's Type 2 Diabetes," including the breakdown of deductibles, copayments, coinsurance, and exclusions. Then, sum these together to provide a combined total out-of-pocket liability assuming both events occurred to the same family unit in the same year.

Attached Files (Context):

- Purdue-2026-J1-Visa-Summary-of-Benefits.pdf – Public URL: https://www.purdue.edu/hr//Benefits///medical/pdf/2026/Purdue-2026-J1-Visa-Summary-of-Benefits.pdf (CMS Standard SBC format with Coverage Examples)
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts Peg's deductible as exactly $250.|Major|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf ,p. 7|Extracted from the "Peg is Having a Baby" Coverage Example matrix.|FALSE|Hallucinated a standard $500 or $1,000 deductible.||
|2|Extracts Peg's coinsurance as exactly $1,100.|Major|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|Extracted from the "Peg is Having a Baby" Coverage Example matrix.|FALSE|Calculated coinsurance independently rather than extracting the matrix value.||
|3|Extracts Peg's limits or exclusions as exactly $20.|Minor|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|Extracted from the "Peg is Having a Baby" Coverage Example matrix.|FALSE|Assumed zero exclusions.||
|4|Identifies the total out-of-pocket cost for Peg as $1,380.|Critical|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|$250 (Ded) + $10 (Copay) + $1,100 (Coins) + $20 (Limits) = $1,380.|FALSE|Confused the total example cost ($12,700) with the patient's out-of-pocket liability.|10|
|5|Extracts Joe's deductible as exactly $250.|Major|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|Extracted from the "Managing Joe's Type 2 Diabetes" matrix.|FALSE|Omitted the deductible for Joe's scenario.||
|6|Extracts Joe's copayments as exactly $200.|Major|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|Extracted from the "Managing Joe's Type 2 Diabetes" matrix.|FALSE|Confused copayments with coinsurance.||
|7|Extracts Joe's coinsurance as exactly $100.|Major|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|Extracted from the "Managing Joe's Type 2 Diabetes" matrix.|FALSE|Pulled data from the wrong column or example.||
|8|Identifies the total out-of-pocket cost for Joe as $550.|Critical|Purdue-2026-J1-Visa-Summary-of-Benefits.pdf, p. 7|$250 (Ded) + $200 (Copay) + $100 (Coins) + $0 (Limits) = $550.|FALSE|Attempted to calculate 90% of the total example cost instead of extracting the matrix sum.|4|
|9|Calculates the combined total out-of-pocket liability for both events as $1,930.|Critical|Prompt Text|$1,380 (Peg) + $550 (Joe) = $1,930.|FALSE|Failed the arithmetic summation of the two extracted totals.|3|
|10|Adopts a professional, advisory tone formatting the breakdown clearly for a client.|Minor|Prompt Text|Requires memo or advisory letter styling.|FALSE|Provided an unstructured data dump without context.||

#### 4. Model Analysis

Evaluating health insurance cost-sharing matrices requires a model to successfully navigate tabular data within a PDF layout. A standard model is expected to score between fifty and seventy percent on this rubric. The primary vector of failure stems from the model's tendency to rely on its training data to independently calculate healthcare costs (e.g., attempting to calculate 10% coinsurance against the total example cost) rather than strictly extracting the pre-calculated out-of-pocket values provided in the statutory SBC document matrices. Models will also frequently conflate the "Total Example Cost" (the total bill submitted by the provider) with the "Total [Name] would pay" (the actual out-of-pocket liability of the insured). Finally, when asked to combine the costs, models may fail to recognize that the prompt asks for a simple addition of the two events as presented in the document, instead attempting to hallucinate complex family deductible aggregations not provided in the prompt constraints.

#### 5. Golden Response

BENEFITS ADVISORY MEMO

TO: Client

FROM: Health Benefits Consultant

DATE: October 24, 2026

SUBJECT: Summary of Benefits and Coverage (SBC) – Out-of-Pocket Liability Estimates

Based on the provided Purdue University J1 Visa Health Plan Summary of Benefits and Coverage (SBC) document, I have broken down your potential financial exposure for the two requested medical scenarios. The Affordable Care Act requires these standardized examples to help you understand how cost-sharing (deductibles, copayments, and coinsurance) functions under your specific plan.

Scenario 1: Peg is Having a Baby

This scenario assumes 9 months of in-network pre-natal care and a hospital delivery with a total provider billing cost of $12,700. Based on the SBC, your out-of-pocket liability would be broken down as follows:

- Deductibles: $250
    
- Copayments: $10
    
- Coinsurance: $1,100
    
- Limits or Exclusions: $20
    
- Total Out-of-Pocket Cost for Peg: $1,380
    

Scenario 2: Managing Joe's Type 2 Diabetes

This scenario assumes a year of routine in-network care of a well-controlled condition, including blood work and prescription drugs, with a total provider billing cost of $5,600. Your out-of-pocket liability would be:

- Deductibles: $250
    
- Copayments: $200
    
- Coinsurance: $100
    
- Limits or Exclusions: $0
    
- Total Out-of-Pocket Cost for Joe: $550
    

Combined Family Liability Estimate

If both of these specific medical events were to occur within the same year for your family unit, we simply aggregate the provided scenario liabilities to estimate your exposure.

- Total Estimated Cost: $1,380 (Pregnancy) + $550 (Diabetes) = $1,930
    

Please note that these are standardized examples required by the Department of Labor and CMS to demonstrate plan mechanics; your actual costs will vary based on the specific care received and your actual billed charges.

### 

---

Example 10: Commercial Auto Underwriting and Fleet Eligibility

#### 1. Metadata

- Sub-domain: Underwriting (Commercial Auto)
    
- Difficulty: Medium
    
- Themes: Risk Selection, Fleet Classification, MVR Omissions
    
- Task Type: Capability
    
- Workflow: Submission Triage & Declination
    
- Prompt Type: Underwriting Decision Memo
    

#### 2. Prompt

You are a Commercial Auto Underwriter for MMG Insurance. The current date is October 24, 2026.

A broker has submitted three commercial auto risks for your review. You must evaluate each submission against the attached underwriting guidelines and provide a definitive "Accept" or "Decline" decision for each risk. For each decision, you must cite the specific rule or exception from the guidelines that drove your conclusion.

Submissions for Review:

1. Risk 1: A fast-food franchise located in Vermont (VT) seeking hired and non-owned auto coverage for employees using their personal sedans for pizza delivery.
    
2. Risk 2: An excavation contractor based in Pennsylvania (PA) seeking coverage for one tractor-trailer. The tractor-trailer is used exclusively to transport their own backhoes and heavy equipment to local job sites within a 30-mile radius.
    
3. Risk 3: A residential snow plowing service operating in Maine (ME). The owner-operator's Motor Vehicle Report (MVR) shows one citation for "Distracted Driving" that occurred 3 years ago (in 2023).
    

Attached Files (Context):

- MMG-Underwriting-Guidelines-2019.pdf – Public URL: https://www.mmgins.com/wp-content/uploads/2019/08/MMG-Underwriting-Guidelines-2019-v5.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Declines Risk 1 (Pizza Delivery).|Critical|MMG-Underwriting-Guidelines-2019.pdf, p. 3|The guidelines prohibit fast food delivery.|FALSE|Accepted the risk based on territory eligibility.||
|2|Cites the explicit guideline prohibiting "pizza/fast food delivery" for Risk 1.|Major|MMG-Underwriting-Guidelines-2019.pdf, p. 3|Stated explicitly in the Vehicle Guidelines section.|FALSE|Cited the wrong rule or hallucinated a radius limitation.|10|
|3|Accepts Risk 2 (Excavation Tractor-Trailer).|Critical|MMG-Underwriting-Guidelines-2019.pdf, p. 3|A specific exception exists for excavation risks moving their own equipment locally.|FALSE|Declined the risk because tractor-trailers are generally prohibited.||
|4|Identifies that the territory for Risk 2 (PA) is an eligible state.|Minor|MMG-Underwriting-Guidelines-2019.pdf, p. 3|PA is listed in the approved MMG territories.|FALSE|Declined the risk based on geographic location.||
|5|Cites the exception permitting tractor-trailers with "excavation with local radius, used to transport own equipment only."|Major|MMG-Underwriting-Guidelines-2019.pdf, p. 3|Provides the exact textual justification for overriding the general tractor-trailer prohibition.|FALSE|Failed to find the exception clause in the document.|2|
|6|Declines Risk 3 (Snow Plowing with MVR violation).|Critical|MMG-Underwriting-Guidelines-2019.pdf, p. 4|The driver has a major violation within the exclusionary window.|FALSE|Accepted the risk because residential snow plowing is an allowed class.||
|7|Identifies that "residential snow plowing" is otherwise an eligible operation.|Minor|MMG-Underwriting-Guidelines-2019.pdf, p. 4|The decline is strictly due to the MVR, not the class of business.|FALSE|Claimed snow plowing is entirely prohibited.||
|8|Classifies the "Distracted Driving" citation as a "Major Violation" according to the MVR table.|Major|MMG-Underwriting-Guidelines-2019.pdf, p. 4|Distracted driving is explicitly listed under Major Violations, not Minor.|FALSE|Assumed distracted driving was a minor violation.|5|
|9|Applies the 5-year lookback rule requiring "None in the past 5 years" for Major Violations.|Major|MMG-Underwriting-Guidelines-2019.pdf, p. 4|Since the ticket is 3 years old, it falls within the 5-year prohibition window.|FALSE|Applied a 3-year lookback window (which applies to at-fault accidents, not major violations).|5|
|10|Formats output clearly as a definitive underwriting decision memo.|Minor|Prompt Text|Narrative styling requirement.|FALSE|Provided an unstructured text block.||

#### 4. Model Analysis

Commercial auto underwriting requires the synthesis of geographic, operational, and driver-history matrices to determine eligibility. A standard model is expected to score between forty and sixty percent on this rubric. The primary vector of failure involves processing the "exceptions to the rule" within the guidelines. For example, while the guidelines explicitly state "No Tractor Trailers," a human underwriter will immediately spot the appended exception for local excavation risks transporting their own equipment. Models frequently trigger a false-negative on the tractor-trailer keyword and decline Risk 2 prematurely.

Furthermore, models struggle with the chronological application of Motor Vehicle Report (MVR) matrices. For Risk 3, the model must first classify "Distracted Driving" as a Major Violation, and then cross-reference that classification against the look-back period, which mandates zero major violations in the past 5 years. Models often confuse the rows in such tables, applying the 3-year look-back period designated for standard "At Fault Accidents" or "Moving Violations" rather than the stricter 5-year window required for Major Violations, leading to inappropriate approvals of hazardous driver profiles.

#### 5. Golden Response

UNDERWRITING DECISION MEMO

TO: Submitting Broker

FROM: Commercial Auto Underwriter (MMG Insurance)

DATE: October 24, 2026

SUBJECT: Submission Triage Decisions

Thank you for your recent commercial auto submissions. I have reviewed the three risks against our current underwriting guidelines. Please see my decisions and rationales below:

Risk 1: Vermont Fast-Food Franchise (Pizza Delivery)

- Decision: DECLINE
    
- Rationale: While Vermont (VT) is an eligible MMG territory, our Vehicle Guidelines strictly prohibit this class of business. The guidelines explicitly state: "No pizza/fast food delivery." Therefore, we cannot offer hired and non-owned auto coverage for this operation.
    

Risk 2: Pennsylvania Excavation Contractor (Tractor-Trailer)

- Decision: ACCEPT
    
- Rationale: Pennsylvania (PA) is an approved territory. While our Vehicle Guidelines generally state "No Tractor Trailers," this submission qualifies under the explicit exception provided in the manual. The guidelines allow an exception for "excavation with local radius, used to transport own equipment only." Because this insured uses the vehicle exclusively to transport their own backhoes within a 30-mile (local) radius, the risk is acceptable.
    

Risk 3: Maine Residential Snow Plowing Service

- Decision: DECLINE
    
- Rationale: While the operational class itself is acceptable (our guidelines note that "Snow Plowing - Residential, small commercial lots ok"), the driver's Motor Vehicle Report makes the risk ineligible. The driver has a 2023 citation for "Distracted Driving." Under our guidelines, Distracted Driving is explicitly classified as a "Major Violation." The matrix for Major Violations strictly requires "None in the past 5 years." Because the violation occurred 3 years ago, it falls within the prohibited 5-year window, and the driver is unacceptable.
    

Please let me know if you have any questions or if you have alternative drivers to submit for Risk 3.

### 

---

Example 11: Cyber Insurance Incident Response & Sub-limit Adjudication

#### 1. Metadata

- Sub-domain: Claims Handling (Cyber)
    
- Difficulty: High
    
- Themes: Ransomware, Aggregate Limits, Sub-limits, Restoration Costs
    
- Task Type: Workflow
    
- Workflow: Quantum Calculation
    
- Prompt Type: Coverage Determination and Settlement Memo
    

#### 2. Prompt

You are a Cyber Insurance Claims Adjuster. The current date is October 24, 2026.

You are evaluating a claim submitted by the City of Franklin following a severe ransomware attack that crippled their municipal network. The city has submitted invoices and demands for reimbursement under their Cyber Insurance policy. You must review the damages, compare them to the explicit sub-limits and the overarching Policy Aggregate Limit of Liability, and calculate the final payable amount by the insurer. Present your findings in a formal Settlement Memo. Ignore any retentions/deductibles for the purpose of this calculation.

Loss Details Submitted by Insured:

- The City incurred $1,200,000 in expenses to restore their digital assets and recover lost business income (Business Income and Digital Asset Restoration).
    
- The City paid a $500,000 ransom to the attackers (Cyber Extortion).
    
- The City spent $200,000 on forensics, legal counsel, and public relations (Security Breach Response Coverage).
    

Attached Files (Context):

- Franklin_Cyber_Policy_Decs.pdf – Public URL: https://www.franklin.in.gov/egov/documents/1568056135_75369.pdf (City of Franklin Cyber Policy Declarations)
    

Note: Pay close attention to both the individual per-claim sub-limits and the overall Policy Aggregate Limit of Liability listed on the declarations page to determine the final payable quantum.

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs and Pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the overall Aggregate Limit of Liability for the policy as $1,000,000.|Critical|Franklin_Cyber_Policy_Decs.pdf, p. 2 (URL: https://www.franklin.in.gov/egov/documents/1568056135_75369.pdf)|The declarations page explicitly sets the maximum aggregate limit at $1,000,000.|FALSE|Missed the overriding cap for the policy.||
|2|Identifies the sub-limit for Business Income and Digital Asset Restoration as $1,000,000.|Major|Franklin_Cyber_Policy_Decs.pdf, p. 2|Listed under Item 3, Coverage G.|FALSE|Hallucinated a different sub-limit.||
|3|Identifies the sub-limit for Cyber Extortion as $1,000,000.|Major|Franklin_Cyber_Policy_Decs.pdf, p. 2|Listed under Item 3, Coverage F.|FALSE|Hallucinated a different sub-limit.||
|4|Identifies the sub-limit for Security Breach Response Coverage as $1,000,000.|Major|Franklin_Cyber_Policy_Decs.pdf, p. 2|Listed under Item 3, Coverage C.|FALSE|Hallucinated a different sub-limit.||
|5|Calculates the total incurred eligible losses across all categories as $1,900,000.|Major|Prompt Text|$1.2M + $500k + $200k = $1.9M.|FALSE|Failed basic arithmetic aggregation of the submitted losses.||
|6|Truncates the Business Income / Digital Asset Restoration eligible claim to its $1,000,000 sub-limit.|Major|Franklin_Cyber_Policy_Decs.pdf, p. 2|The $1.2M incurred exceeds the $1.0M specific sub-limit for this coverage.|FALSE|Allowed the full $1.2M for this specific category.|1|
|7|Determines that the total eligible loss subject to the policy aggregate is $1,700,000.|Major|Prompt Text; Franklin_Cyber_Policy_Decs.pdf, p. 2|$1,000,000 (capped BI) + $500,000 (Extortion) + $200,000 (Breach Response) = $1.7M.|FALSE|Failed to apply the sub-limit cap before aggregating.|4|
|8|Applies the overriding rule that the combined total payout cannot exceed the Aggregate Limit of Liability.|Critical|Franklin_Cyber_Policy_Decs.pdf, p. 2|The sum of all sub-limit claims cannot exceed the $1M aggregate policy cap.|FALSE|Attempted to pay out the sum of the sub-limits ($1.7M or $1.9M) ignoring the aggregate cap.|10|
|9|Calculates the final total payable amount as exactly $1,000,000.|Critical|Prompt Text; Franklin_Cyber_Policy_Decs.pdf, p. 2|The total eligible claim ($1.7M) is truncated down to the maximum aggregate limit of $1M.|FALSE|Paid an amount greater than the policy's absolute maximum limit.|6|
|10|Formats output as a formal claims settlement memo.|Minor|Prompt Text|Narrative styling requirement.|FALSE|Output only a numeric value with no prose.||

#### 4. Model Analysis

Cyber insurance policies frequently utilize a structure where multiple individual coverage sub-limits equal the total policy aggregate limit, creating a trap for basic extraction logic. A standard LLM is expected to score between thirty and fifty percent on this rubric. The model will successfully extract the one-million-dollar sub-limits for Extortion, Breach Response, and Digital Asset Restoration. However, because the incurred losses cross multiple categories, models will typically add the eligible amounts together (e.g., $1M capped BI + $500k extortion + $200k response = $1.7M) and recommend paying the total sum. The model fails to recognize the hierarchical relationship between "Per Claim Sublimits" and the overriding "Aggregate Limit of Liability." Understanding that the entire policy maxes out at $1,000,000 regardless of how many individual sub-limits are triggered requires a complex synthesis of insurance contract architecture that base models lack.

#### 5. Golden Response

SETTLEMENT MEMORANDUM

TO: Cyber Claims File – City of Franklin

FROM: Cyber Insurance Claims Adjuster

DATE: October 24, 2026

SUBJECT: Coverage Determination and Final Quantum – Ransomware Incident

1. Incident Overview

The insured, the City of Franklin, sustained a severe ransomware attack resulting in network encrypton and extortion demands. The insured has submitted invoices totaling $1,900,000 across three coverage categories: Business Income/Digital Asset Restoration, Cyber Extortion, and Security Breach Response.

2. Sub-Limit Adjudication

Per the Declarations Page (Item 3), each specific coverage grant is subject to its own sub-limit.

- Business Income and Digital Asset Restoration (Coverage G): The insured incurred $1,200,000. However, the Per Claim Sublimit for this coverage is $1,000,000. Therefore, the eligible amount for this category is capped at $1,000,000.
    
- Cyber Extortion (Coverage F): The insured incurred $500,000. This is well within the $1,000,000 sub-limit, so the full $500,000 is eligible.
    
- Security Breach Response (Coverage C): The insured incurred $200,000. This is within the $1,000,000 sub-limit, so the full $200,000 is eligible.
    
- Total Eligible Under Sub-Limits: $1,000,000 + $500,000 + $200,000 = $1,700,000.
    

3. Aggregate Limit Application

While the sum of the eligible claims under the individual sub-limits equals $1,700,000, all payments are subject to the overarching Policy Aggregate Limit of Liability. Item 3(A) of the Declarations explicitly states that the Aggregate Limit of Liability for the entire policy is $1,000,000.

4. Final Quantum

Because the total eligible damages ($1,700,000) exceed the absolute maximum limit of the policy, the final payout must be truncated. The insurer's maximum exposure is reached.

Final Payable Amount: $1,000,000

### 

---

Example 12: D&O Liability & Entity Reimbursement (Side A/B/C Application)

#### 1. Metadata

- Sub-domain: Executive Risk (Directors & Officers)
    
- Difficulty: Medium-High
    
- Themes: Insuring Agreements (Side A vs. B vs. C), Indemnification, Corporate Insolvency
    
- Task Type: Workflow
    
- Workflow: Claims Coverage Trigger Analysis
    
- Prompt Type: Executive Liability Brief
    

#### 2. Prompt

You are a D&O Claims Counsel. The current date is October 24, 2026.

A group of shareholders has filed a lawsuit directly against three Directors of a technology corporation, alleging gross mismanagement and breach of fiduciary duty resulting in a massive drop in stock value. Simultaneously, the technology corporation itself has declared bankruptcy and is legally and financially insolvent. Because of the insolvency, the corporation is completely unable to advance defense costs or indemnify the Directors for this lawsuit.

Review the attached Directors and Officers Liability presentation detailing standard policy architecture. You must write a brief outlining which specific "Side" (Insuring Clause A, B, or C) of the D&O policy is triggered to protect the individual Directors in this exact scenario. Additionally, you must state whether the policy deductible will apply to the Directors' defense costs based on the triggered clause.

Attached Files (Context):

- JKB_DO_Liability_Overview.pdf – Public URL: https://www.jkb.bank.in/sites/default/files/2025-03/DIRECTORS%20AND%20OFFICERS%20LIABILITY%20INSURANCE.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs and Pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies that the corporation cannot indemnify the directors due to bankruptcy/insolvency.|Minor|Prompt Text|Establishes the factual predicate for the coverage trigger.|FALSE|Ignored the bankruptcy constraint.||
|2|Determines that Side A (Directors & Officers / Personal Liability) coverage is triggered.|Critical|JKB_DO_Liability_Overview.pdf, Slide 4 (URL: https://www.jkb.bank.in/sites/default/files/2025-03/DIRECTORS%20AND%20OFFICERS%20LIABILITY%20INSURANCE.pdf)|Side A explicitly provides indemnity to the individual when no reimbursement is provided by the company.|FALSE|Triggered Side B or Side C incorrectly.|10|
|3|Explicitly states that Side B (Company Reimbursement) is NOT triggered.|Major|JKB_DO_Liability_Overview.pdf, Slide 4|Side B only triggers when the company has actually reimbursed the individuals, which it cannot do here.|FALSE|Assumed Side B applies because it involves the company.|1|
|4|Explicitly states that Side C (Entity) is NOT triggered.|Major|JKB_DO_Liability_Overview.pdf, Slide 4|The lawsuit is against the Directors personally, not the Entity itself.|FALSE|Assumed Side C applies due to the shareholder nature of the suit.|1|
|5|Concludes that the policy deductible does NOT apply to the defense costs.|Critical|JKB_DO_Liability_Overview.pdf, Slide 4|The document explicitly states that under Side A coverage, the "Deductible does not apply."|FALSE|Applied a standard deductible to the Directors' costs.|1|

#### 4. Model Analysis

D&O Liability architecture is notoriously complex because it features three distinct insuring agreements wrapped into a single policy, with triggers dependent entirely on corporate bylaws and solvency. Models are expected to score around forty to sixty percent. While a model can generally map the definition of "Side A" to "Directors," it routinely fails to logically exclude Side B when reading the prompt. Because models associate corporate executives with corporate reimbursement, they often hallucinate that Side B will eventually cover the loss, missing the absolute constraint that bankruptcy prevents indemnification. Crucially, models usually fail the deductible question. General insurance knowledge dictates that policies have deductibles; however, Side A is specifically designed as first-dollar coverage (no deductible) to protect individuals when their company abandons them. Base models will default to standard property/casualty logic and assume a deductible applies.

#### 5. Golden Response

EXECUTIVE LIABILITY CLAIMS BRIEF

TO: Claims Management

FROM: D&O Claims Counsel

DATE: October 24, 2026

SUBJECT: Coverage Trigger Analysis – Shareholder Lawsuit and Corporate Insolvency

1. Factual Overview

We have received notice of a shareholder lawsuit naming three individual Directors of the insured technology corporation. The corporation has declared bankruptcy and is financially insolvent, rendering it unable to advance defense costs or provide indemnification to the individual Directors for these allegations.

2. Policy Coverage Trigger (Side A, B, or C)

Based on the standard D&O policy architecture outlined in the reference guide, the specific coverage triggered for this event is Side A (Directors & Officers / Personal Liability).

- Why Side A applies: Side A is triggered specifically when allegations against Directors and Officers are not indemnifiable by the corporation. It provides direct indemnity to the individuals when no reimbursement is provided by the company. Because the corporation is insolvent and cannot indemnify the Directors, Side A drops down to protect their personal assets.
    
- Why Side B (Company Reimbursement) does not apply: Side B is only triggered when the company has actually indemnified and reimbursed the individuals. Since the bankruptcy prevents this, Side B is entirely inapplicable.
    
- Why Side C (Entity) does not apply: Side C covers the corporate entity itself against securities claims. Since the suit names the individual Directors and not the bankrupt entity, Side C is not triggered.
    

3. Deductible Application

Under the triggered Side A insuring clause, the deductible does not apply. Side A provides first-dollar defense and indemnity protection for the individual Directors. (Conversely, if the company had been solvent and able to indemnify them under Side B, the corporate deductible would have applied). Therefore, the insurer will advance defense costs for the Directors from dollar one.

### 

---

Example 13: Property Risk Modeling (Catastrophe Data Mapping)

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Actuarial
    
- Workflow: Property Portfolio Valuation
    
- Prompt Type: Risk Modeling Calculation
    
- Difficulty: High
    

#### 2. Prompt

You are a Catastrophe Risk Modeler. The current date is October 24, 2026.

You are evaluating a raw data extract of three buildings in the San Francisco Bay Area to determine the total modeled replacement cost of the portfolio. You must use the specific mapping rules provided by the UrbanSim database documentation to determine the correct Replacement Cost per Square Foot based on the building's Occupancy ID.

Portfolio Data Extract:

- Building Alpha: Occupancy ID: 4 | Total Floor Area: 10,000 SQFT
    
- Building Beta: Occupancy ID: 6 | Total Floor Area: 5,000 SQFT
    
- Building Gamma: Occupancy ID: 99 | Total Floor Area: 2,000 SQFT
    

Calculate the individual replacement cost for each building, and then sum them to provide the Total Portfolio Replacement Cost. Show your math.

Attached Files (Context):

- SimCenter_Asset_Description.html – Public URL: https://nheri-simcenter.github.io/R2D-Documentation/common/testbeds/sf_bay_area/asset_description.html (Documentation containing Table 1.2.1 Mapping rules)
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the mapping rule for Building Alpha (Occupancy ID 4).|Minor|SimCenter_Asset_Description.html, HTML Table 1.2.1 (URL: https://nheri-simcenter.github.io/R2D-Documentation/common/testbeds/sf_bay_area/asset_description.html)|ID 4 maps to "Office" at $131.9 / SQFT.|FALSE|Mapped to the wrong category.||
|2|Calculates the replacement cost for Building Alpha as $1,319,000.|Major|Prompt Text|10,000 SQFT multiplied by $131.9.|FALSE|Arithmetic error.|10|
|3|Identifies the mapping rule for Building Beta (Occupancy ID 6).|Minor|SimCenter_Asset_Description.html, HTML Table 1.2.1|ID 6 maps to "School" at $142.1 / SQFT.|FALSE|Mapped to the wrong category.||
|4|Calculates the replacement cost for Building Beta as $710,500.|Major|Prompt Text|5,000 SQFT multiplied by $142.1.|FALSE|Arithmetic error.|2|
|5|Identifies the fallback mapping rule for Building Gamma (Occupancy ID 99).|Critical|SimCenter_Asset_Description.html, HTML Table 1.2.1|ID 99 is invalid. The text states: "For buildings with missing or invalid occupancy ID, the building was mapped to the default occupancy type (i.e., residential)". Residential is $137.5 / SQFT.|FALSE|Excluded the building from the calculation entirely or hallucinated a zero value.||
|6|Calculates the replacement cost for Building Gamma as $275,000.|Major|Prompt Text|2,000 SQFT multiplied by $137.5.|FALSE|Failed to apply the default residential rate.|4|
|7|Calculates the Total Portfolio Replacement Cost as $2,304,500.|Critical|Prompt Text|$1,319,000 + $710,500 + $275,000 = $2,304,500.|FALSE|Incorrect sum due to failure in previous mapping steps.|1|

#### 4. Model Analysis

This task tests a model's ability to execute precise data dictionary cross-referencing, a core task in catastrophe modeling. A base language model will likely score between forty and sixty percent. Models generally succeed at direct 1-to-1 lookups (e.g., matching ID 4 to the Office rate). The primary vector of failure is Building Gamma. ID 99 does not exist in the literal table rows. A human data analyst reading the documentation will spot the prose instruction above the table dictating that invalid IDs must fallback to the "Default" residential rate of $137.5. Language models, overly fixated on the tabular markdown representation of the table, frequently fail to apply the surrounding prose rules, resulting in Building Gamma being priced at $0, dropped from the portfolio, or hallucinated with a random multiplier.

#### 5. Golden Response

PORTFOLIO REPLACEMENT COST CALCULATION

DATE: October 24, 2026

SUBJECT: SimCenter Bay Area Asset Valuation

Based on the UrbanSim asset description documentation and Table 1.2.1 (Mapping rules for building occupancy type and replacement cost), the portfolio valuation is calculated by mapping each building's Occupancy ID to the corresponding Replacement Cost per Square Foot.

1. Building Alpha

- Occupancy ID: 4
    
- Mapped Class: Office
    
- Rate: $131.9 / SQFT
    
- Calculation: 10,000 SQFT ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) $131.9 = $1,319,000
    

2. Building Beta

- Occupancy ID: 6
    
- Mapped Class: School
    
- Rate: $142.1 / SQFT
    
- Calculation: 5,000 SQFT ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) $142.1 = $710,500
    

3. Building Gamma

- Occupancy ID: 99 (Invalid)
    
- Mapped Class: Default (Residential)
    
- Rule Application: The documentation specifies that "For buildings with missing or invalid occupancy ID, the building was mapped to the default occupancy type (i.e., residential)." The Residential rate is $137.5 / SQFT.
    
- Calculation: 2,000 SQFT ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) $137.5 = $275,000
    

4. Total Portfolio Valuation

The total modeled replacement cost is the sum of the three individual buildings.

- $1,319,000 + $710,500 + $275,000 = $2,304,500
    

The Total Portfolio Replacement Cost is $2,304,500.

### 

---

Example 14: Workers' Compensation Premium Calculation

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Underwriting
    
- Workflow: Premium Calculation
    
- Prompt Type: Rating Worksheet
    
- Difficulty: Medium
    

#### 2. Prompt

You are a Commercial Underwriter. The current date is October 24, 2026. Calculate the total manual premium for a landscaping risk located in New York. The business operates under two class codes: Code 0005 (Payroll: $100,000) and Code 0006 (Payroll: $50,000). The company's filed Loss Cost Multiplier (LCM) is 1.20. Round final premiums to the nearest dollar.

Attached Files (Context):

- NY-LC-Comparison-10-1-2025.pdf – Public URL: https://amtrustfinancial.com/getmedia/f21fa4f3-e42a-439d-8a02-9831a3b078f6/NY-LC-Comparison-10-1-2025.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts Oct 2025 rate for 0005 as 0.884.|Major|NY-LC-Comparison-10-1-2025.pdf, PDF Page 4|Requires extracting the correct, most recent active rate from the table.|FALSE|Used the 2024 rate.||
|2|Extracts Oct 2025 rate for 0006 as 1.362.|Major|NY-LC-Comparison-10-1-2025.pdf, PDF Page 4|Requires extracting the correct rate.|FALSE|Used the 2024 rate.||
|3|Calculates Code 0005 premium as $1,061.|Critical|Prompt Text|($100k/100) * 0.884 * 1.20 = $1,060.80, rounded to $1,061.|FALSE|Failed to divide payroll by 100 before applying the rate.|10|
|4|Calculates Code 0006 premium as $817.|Critical|Prompt Text|($50k/100) * 1.362 * 1.20 = $817.20, rounded to $817.|FALSE|Failed to apply the Loss Cost Multiplier.|1|
|5|Calculates total manual premium as $1,878.|Critical|Prompt Text|$1,061 + $817 = $1,878.|FALSE|Incorrect sum.|2|

#### 4. Model Analysis

Models frequently fail Workers' Compensation calculations because they do not intuitively understand that rates are applied per $100 of payroll, resulting in premiums that are inflated by a factor of 100. Furthermore, extracting the correct rate column (2025 vs. 2024) tests chronological reasoning.

#### 5. Golden Response

Workers' Compensation Rating Worksheet

- Class 0005: $100,000 payroll / 100 = 1,000 exposure base. Rate (Oct 2025) = 0.884. LCM = 1.20. Premium = 1,000 * 0.884 * 1.20 = $1,061
    
- Class 0006: $50,000 payroll / 100 = 500 exposure base. Rate (Oct 2025) = 1.362. LCM = 1.20. Premium = 500 * 1.362 * 1.20 = $817
    
- Total Manual Premium: $1,061 + $817 = $1,878
    

### 

---

Example 15: Surety Performance Bond Obligations

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Underwriting
    
- Workflow: Bond Issuance
    
- Prompt Type: Legal Brief
    
- Difficulty: Low
    

#### 2. Prompt

You are a Surety Underwriter. The current date is October 24, 2026. Review the attached City of Tukwila Performance Bond template. Under what two specific, enumerated conditions will the Principal's obligation continue in effect until released in writing? Provide the exact headers for these two conditions.

Attached Files (Context):

- DCD-Performance-Bond-Template.pdf – Public URL: https://www.tukwilawa.gov/wp-content/uploads/DCD-Performance-Bond-Template.pdf
    

#### 3. Rubric

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies Condition 1 as "Description of Work".|Critical|DCD-Performance-Bond-Template.pdf, PDF Page 2|Exact header from the form.|FALSE|Hallucinated a generic surety condition.||
|2|Identifies Condition 2 as "Compliance of Work with Specifications".|Critical|DCD-Performance-Bond-Template.pdf, PDF Page 2|Exact header from the form.|FALSE|Hallucinated a generic surety condition.||

#### 4. Model Analysis

When tasked with extracting legal conditions from a specific bond template, LLMs often bypass the attached document and inject generic knowledge regarding surety bonds (e.g., "timely completion" and "payment of subs").

#### 5. Golden Response

According to the attached City of Tukwila template, the obligation continues until released in writing after the Principal has performed and satisfied the following two conditions:

1. Description of Work
    
2. Compliance of Work with Specifications
    

### 

---

Example 16: Life Insurance Underwriting (Preferred Plus)

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Underwriting
    
- Workflow: Eligibility Assessment
    
- Prompt Type: Underwriting Review
    
- Difficulty: Medium
    

#### 2. Prompt

You are a Life Underwriter. The current date is October 24, 2026. Using the Nationwide Underwriting Guide, determine if a 56-year-old nontobacco user with treated blood pressure of 135/85, which has been well-controlled for two years, qualifies for the "Preferred Plus" rating tier for a standard term product. Explain why or why not.

Attached Files (Context):

- NFM-23978AO-WG.pdf – Public URL: https://nationwidefinancial.com/media/pdf/NFM-23978AO-WG.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the correct age bracket as "Ages 70 and Under".|Minor|NFM-23978AO-WG.pdf, PDF Page 9|Determines the section to read.|FALSE|Used the 71 and Older bracket.||
|2|Identifies the BP reading (135/85) is mathematically within the acceptable limit (140/90).|Major|NFM-23978AO-WG.pdf, PDF Page 9|The raw numbers fit the parameters.|FALSE|Claimed the BP was too high.||
|3|Identifies the exclusion that "No blood pressure treatment is permitted" for Preferred Plus.|Critical|NFM-23978AO-WG.pdf, PDF Page 9|Qualitative limitation overrides the numerical reading.|FALSE|Assumed treated BP was acceptable.||
|4|Concludes the applicant does NOT qualify for Preferred Plus.|Critical|Prompt Text|Final determination.|FALSE|Approved the applicant for Preferred Plus.|2|

#### 4. Model Analysis

Models frequently fail to synthesize numerical and qualitative constraints simultaneously. A model will see that 135/85 is lower than 140/90 and immediately approve the risk, completely ignoring the strict "No blood pressure treatment is permitted" clause assigned to the Preferred Plus tier.

#### 5. Golden Response

Decision: DECLINE for Preferred Plus.

Rationale: While the applicant's blood pressure reading of 135/85 is mathematically below the maximum threshold of 140/90 for applicants over age 55 in the "Ages 70 and Under" bracket, they do not qualify for the "Preferred Plus" tier. The Nationwide guide explicitly states that for the Preferred Plus category, "No blood pressure treatment is permitted." Because the applicant has treated blood pressure, they are ineligible for this top tier, regardless of how well-controlled it is.

### 

---

Example 17: Medicare Advantage Cost-Sharing

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Agent/Broker
    
- Workflow: Benefit Explanation
    
- Prompt Type: Member Inquiry Response
    
- Difficulty: Low
    

#### 2. Prompt

You are a Benefits Counselor. The current date is October 24, 2026. A member on the MedMutual Advantage plan needs a 90-day supply of a Tier 3 (preferred brand) drug. Based on the attached Summary of Benefits, what is their exact copay for this specific fill?

Attached Files (Context):

- MEDM104X11289MCA4242025GRMASBMedicalMutualAdvantagePlusFA.pdf – Public URL: https://www.medmutual.com/-/media/MedMutual/Files/For-Medicare/2025/Summary-of-Benefits/EGWP/MEDM104X11289MCA4242025GRMASBMedicalMutualAdvantagePlusFA.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Locates the row for Tier 3 drugs.|Minor|MEDM104X11289MCA4242025GRMASBMedicalMutualAdvantagePlusFA.pdf, PDF Page 11|Grid navigation.|FALSE|Pulled Tier 2 or 4.||
|2|Extracts the copay for a "31-90 day supply" as $37.|Critical|MEDM104X11289MCA4242025GRMASBMedicalMutualAdvantagePlusFA.pdf, PDF Page 11|Exact column match.|FALSE|Pulled the 30-day supply copay ($15).|10|

#### 4. Model Analysis

Navigating multi-column tables embedded in PDFs is challenging for base models. The model is highly likely to extract the 30-day supply cost ($15) instead of navigating to the adjacent column representing the requested 90-day fill ($37).

#### 5. Golden Response

According to your Summary of Benefits, the copay for a 31-90 day supply of a Tier 3 (preferred brand and generic) drug is $37.

### 

---

Example 18: Commercial Auto CA 00 01 Exclusions

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Claims Handling
    
- Workflow: Coverage Verification
    
- Prompt Type: Claims Determination
    
- Difficulty: Medium
    

#### 2. Prompt

You are an Auto Claims Adjuster. The current date is October 24, 2026. Your insured crashed their company truck while delivering custom furniture they manufactured to a client. The insured filed a claim under their Business Auto Coverage Form (CA 00 01) requesting payment for the destroyed furniture in the back of the truck. Using the attached policy, determine if the policy covers the damage to the furniture.

Attached Files (Context):

- BusinessAutoCoverageForm_CA_00_01.pdf – Public URL: https://www.archive.asrc.net/ASRC-Admin/ASRC-Financial-Insurance/ASRC-Insurance/2020-02-01-ASRC-Insurance-3-Auto.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the "Care, Custody or Control" exclusion in the Liability section.|Major|BusinessAutoCoverageForm_CA_00_01.pdf, PDF Page 42|The CA 00 01 excludes property owned or transported by the insured.|FALSE|Cited the wrong exclusion or hallucinated coverage.||
|2|Concludes that there is NO coverage for the destroyed furniture.|Critical|Prompt Text|Final determination based on the exclusion.|FALSE|Approved the claim under Property Damage liability.|10|

#### 4. Model Analysis

LLMs often struggle with the distinction between Property Damage Liability and Inland Marine/Cargo exposures. Because the policy covers "property damage," the model frequently assumes the cargo in the truck is covered, missing the explicit exclusion for property transported by or in the care, custody, or control of the insured.

#### 5. Golden Response

Decision: Deny.

Under the standard Business Auto Coverage Form (CA 00 01), there is no coverage for the destroyed furniture. The Liability section contains a specific exclusion for "Care, Custody or Control," which explicitly excludes property damage to "Property owned by an 'insured'... or to property transported by or in the care, custody or control of an 'insured.'" Because the insured was transporting the furniture, they will need a separate Motor Truck Cargo or Inland Marine policy to cover this loss.

### 

---

Example 19: Excess of Loss Reinsurance Layering

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Reinsurance
    
- Workflow: Quantum Calculation
    
- Prompt Type: Treaty Recovery Statement
    
- Difficulty: High
    

#### 2. Prompt

You are a Reinsurance Analyst. The current date is October 24, 2026. A massive hurricane generated a gross loss of $4.6 billion for your company. According to your Corporate Catastrophe Excess-of-Loss Reinsurance Treaty (attached), calculate the exact dollar amount of the recovery you will receive from this specific treaty. Show your math.

Attached Files (Context):

- tm2510932d1_ars.pdf – Public URL: https://www.sec.gov/Archives/edgar/data/86312/000110465925032230/tm2510932d1_ars.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts the retention amount as $4.0 billion.|Major|tm2510932d1_ars.pdf, PDF Page 31|The layer attaches at $4.0B.|FALSE|Failed to identify the attachment point.||
|2|Calculates the loss amount subject to the layer as $600,000,000.|Major|Prompt Text|$4.6B Gross Loss - $4.0B Retention = $600M.|FALSE|Attempted to apply the percentage to the entire gross loss.|10|
|3|Extracts the recovery percentage as 80% for the $4B to $5B layer.|Major|tm2510932d1_ars.pdf, PDF Page 31|The treaty dictates 80% participation for this specific band.|FALSE|Used the 95% or 100% participation figures from higher bands.||
|4|Calculates the final recovery amount as $480,000,000.|Critical|Prompt Text|$600,000,000 * 0.80 = $480,000,000.|FALSE|Math error or incorrect layering.|1|

#### 4. Model Analysis

Models frequently fail to process banded Excess of Loss treaties correctly. They routinely apply the 80% participation rate to the entire gross loss of $4.6 billion, rather than isolating the $600 million that actually pierced the attachment point, resulting in massive hallucinations in recovery calculations.

#### 5. Golden Response

TREATY RECOVERY CALCULATION

Based on the Corporate Catastrophe Excess-of-Loss Reinsurance Treaty terms:

- Gross Loss: $4,600,000,000
    
- Treaty Retention: $4,000,000,000
    
- Subject Loss (Amount in Layer): $4,600,000,000 - $4,000,000,000 = $600,000,000
    
- Recovery Percentage: The treaty provides for recovery of 80% of losses between $4.0B and $5.0B.
    
- Total Recovery: $600,000,000 x 0.80 = $480,000,000
    

### 

---

Example 20: D&O Side B Identification

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Claims Handling
    
- Workflow: Claims Coverage Trigger Analysis
    
- Prompt Type: Coverage Brief
    
- Difficulty: Medium
    

#### 2. Prompt

You are a D&O Claims Examiner. The current date is October 24, 2026. A director was sued, and the financially solvent corporation fully indemnified the director for their legal defense costs as required by the corporate constitution. The corporation is now seeking reimbursement from its D&O policy. Based on the attached overview, which specific "Side" (A, B, or C) is triggered, and does a deductible apply?

Attached Files (Context):

- JKB_DO_Liability_Overview.pdf – Public URL: https://www.jkb.bank.in/sites/default/files/2025-03/DIRECTORS%20AND%20OFFICERS%20LIABILITY%20INSURANCE.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies that Side B (Company Reimbursement) is triggered.|Critical|JKB_DO_Liability_Overview.pdf, PDF Slide 4|Side B indemnifies the company when it reimburses an individual.|FALSE|Identified Side A.||
|2|Confirms that a deductible DOES apply.|Major|JKB_DO_Liability_Overview.pdf, PDF Slide 4|Slide 4 explicitly states the deductible applies for Side B.|FALSE|Stated no deductible applies.|10|

#### 4. Model Analysis

When evaluating D&O policies, LLMs frequently confuse the entity receiving the cash with the entity originally sued. Because the director was sued, models often trigger Side A, failing to understand that because the company stepped in to indemnify the director, the insurance recovery shifts entirely to Side B (Company Reimbursement).

#### 5. Golden Response

COVERAGE TRIGGER

- Triggered Clause: Side B (Company Reimbursement) is triggered. Because the company was financially solvent and successfully indemnified the director in accordance with its constitution, the policy responds by reimbursing the company for those expenditures. Side A does not apply because the director was not left out-of-pocket.
    
- Deductible: Yes, the corporate deductible applies to Side B claims.
    

### 

---

Example 21: Personal Lines Property (HO-5) Earth Movement & Ensuing Fire

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Claims Handling
    
- Workflow: Coverage Verification
    
- Prompt Type: Claims Determination Letter
    
- Difficulty: High
    

#### 2. Prompt

You are a Homeowners Claims Adjuster. The current date is October 24, 2026. A massive earthquake struck a neighborhood, causing severe structural damage to your insured's home. The earthquake also ruptured an underground gas line on the property, which immediately ignited. The ensuing fire completely destroyed what remained of the home.

The insured has an active HO-5 Comprehensive Form policy (see attached). The home's total replacement cost is $500,000, and the policy has a $1,000 deductible. Determine the coverage applicability for the earthquake damage versus the fire damage based explicitly on the language in the "Earth Movement" exclusion. Write a brief letter to the insured explaining how much of the $500,000 loss is payable.

Attached Files (Context):

- AAA_HW_A5_00_11_09.pdf – Public URL: https://doi.nv.gov/uploadedfiles/doinvgov/_public-documents/Consumers/Home/CSAA/AAA_HW_A5_00_11_09.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies that the HO-5 policy contains an exclusion for "Earth Movement."|Minor|AAA_HW_A5_00_11_09.pdf, PDF Page 1|Evaluates the foundational exclusion in Section I.|FALSE|Ignored the Earth Movement exclusion entirely.||
|2|Extracts the specific language that Earth Movement caused by nature is excluded.|Major|AAA_HW_A5_00_11_09.pdf, PDF Page 1|Identifies the scope of the exclusion.|FALSE|Misinterpreted the exclusion as only applying to man-made movement.|10|
|3|Identifies the critical exception clause: "unless direct loss by fire or explosion ensues".|Critical|AAA_HW_A5_00_11_09.pdf, PDF Page 1|The policy carves out an exception for resulting fire.|FALSE|Applied the Earth Movement exclusion to both the quake and the fire.|1|
|4|Clarifies that if fire ensues, the policy will "pay only for the ensuing loss."|Major|AAA_HW_A5_00_11_09.pdf, PDF Page 1|Differentiates between the initial quake damage (excluded) and the fire (covered).|FALSE|Stated the entire $500k loss is covered because a fire occurred.|2|
|5|Concludes that the direct structural damage caused solely by the earthquake is NOT covered.|Critical|Prompt Text; Policy Form|Applies the exclusion correctly to the first sequence of events.|FALSE|Extended coverage to the earthquake damage.|3|
|6|Concludes that the damage caused by the subsequent gas line fire IS covered.|Critical|Prompt Text; Policy Form|Applies the ensuing loss exception correctly.|FALSE|Denied the fire damage due to the proximate cause being an earthquake.|3|
|7|Calculates that the payable amount is subject to the $1,000 deductible.|Minor|AAA_HW_A5_00_11_09.pdf, PDF Page 1|Acknowledges the standard deductible application.|FALSE|Forgot to apply the deductible to the fire loss.||
|8|Determines that the exact payable dollar amount cannot be a flat $499,000 because the total $500,000 RCV includes excluded earthquake damage.|Critical|Claims Adjusting Principles|The adjuster must bifurcate the damages before applying limits.|FALSE|Paid $499,000, assuming the entire home's value was payable.|4|
|9|Explains to the insured that an inspection is required to determine the specific value of the fire damage separate from the earthquake damage.|Major|Prompt Text|Provides the practical next step for the insured.|FALSE|Did not explain how the final payout will be calculated.|7|
|10|Formats the response as a professional claims determination letter.|Minor|Prompt Text|Required output styling.|FALSE|Provided an unstructured list of facts.||

#### 4. Model Analysis

State-of-the-art models frequently struggle with "ensuing loss" provisions (also known as exception-to-an-exclusion clauses) in property policies. When confronted with an excluded peril (earthquake) that proximately causes a covered peril (fire), language models tend to adopt an "all or nothing" approach. They will either read the "Earth Movement" exclusion and incorrectly deny the entire claim (including the fire), or they will read the "ensuing fire" exception and incorrectly approve the entire $500,000 replacement cost, failing to realize that the damage must be bifurcated and only the subsequent fire damage is payable.

#### 5. Golden Response

COVERAGE DETERMINATION LETTER

TO: Insured Homeowner

FROM: Homeowners Claims Adjuster

DATE: October 24, 2026

SUBJECT: Coverage Analysis for Earthquake and Ensuing Fire

Dear Insured,

We are writing to provide a preliminary coverage analysis regarding the severe damage to your home resulting from the recent earthquake and subsequent gas line fire.

Your property is insured under the HO-5 Comprehensive Form policy. According to Section I - Exclusions, A.2 Earth Movement, the policy explicitly excludes loss of any kind due to earthquakes or earth shocks. Therefore, any direct structural damage to your home caused by the shaking of the earthquake is not covered.

However, the Earth Movement exclusion contains a critical exception. It states that if the earth movement results in an ensuing fire or explosion, "we will pay only for the ensuing loss." Because the earthquake caused a gas line to rupture and ignite, the damage specifically caused by the resulting fire is covered under your policy.

Payment Calculation:

While the total replacement cost of your home is $500,000, we cannot issue a blanket payment for $499,000 ($500k minus your $1,000 deductible). The policy requires us to separate the damages. You are responsible for the initial damage caused by the earthquake shaking, and the policy will cover the damage caused by the fire that followed. We will be assigning an independent adjuster and engineering team to evaluate the site and determine the specific valuation of the fire damage versus the earthquake damage. Once that valuation is complete, your $1,000 deductible will be applied to the covered fire loss portion.

We will be in touch shortly to schedule the inspection.

### 

---

Example 22: Lawyers Professional Liability (Claims-Made & Defense within Limits)

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Claims Handling
    
- Workflow: Quantum Calculation
    
- Prompt Type: Claims Apportionment Memo
    
- Difficulty: High
    

#### 2. Prompt

You are a Professional Liability Claims Adjuster. The current date is October 24, 2026.

Your insured, a mid-sized law firm, was sued for malpractice. The lawsuit was successfully settled. You must calculate the final out-of-pocket costs for the insured and the total amount paid by the insurance company based on the attached policy form.

Claim Details:

- The insured's Lawyers Professional Liability policy has a Limit of Liability of $1,000,000.
    
- The policy has a Deductible of $50,000.
    
- The total Defense Costs (legal fees to defend the insured) amounted to $150,000.
    
- The final Settlement amount paid to the plaintiff was $900,000.
    
- Assume the claim was made and reported correctly during the active policy period.
    

Using the specific provisions in the attached specimen policy regarding how claim expenses interact with the limits and deductibles, calculate the final financial apportionment.

Attached Files (Context):

- 141781-(1-22)-LAWYERS-PROFESSIONAL-LIABILITY-INSURANCE-POLICY_Non-NY-Speciman.pdf – Public URL: https://www.attorneys-advantage.com/getmedia/dfed28c4-3d77-4954-8586-3fc2990c2659/141781-(1-22)-LAWYERS-PROFESSIONAL-LIABILITY-INSURANCE-POLICY_Non-NY-Speciman.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies that the policy utilizes "Defense Within Limits" (eroding limits).|Critical|141781...LAWYERS-PROFESSIONAL-LIABILITY.pdf, PDF Page 2|The Preface explicitly states "Claim expenses reduce this policy's limits of liability".|FALSE|Assumed defense costs were paid in addition to the limit.||
|2|Identifies that the Deductible applies to claim expenses as well as damages.|Major|141781...LAWYERS-PROFESSIONAL-LIABILITY.pdf, PDF Page 2|The Preface explicitly states claim expenses are "subject to the policy's deductible."|FALSE|Applied the deductible only to the settlement amount.||
|3|Calculates the total gross loss (Defense + Settlement) as $1,050,000.|Minor|Prompt Text|$150k + $900k = $1.05M.|FALSE|Failed to sum the total incurred costs.||
|4|Applies the $50,000 deductible to the gross loss.|Major|Prompt Text|Insured must pay the first $50k.|FALSE|Deducted $50k from the policy limit instead of the loss.|1|
|5|Determines the net loss subject to the insurer's limit is $1,000,000.|Major|Prompt Text|$1,050,000 - $50,000 = $1,000,000.|FALSE|Incorrect subtraction.|3|
|6|Compares the net loss ($1M) to the remaining Policy Limit ($1M).|Critical|Policy Principles|Because defense costs erode the limit, the total available limit for both is $1M.|FALSE|Treated the $1M limit as applying only to the settlement.|10|
|7|Concludes the Insurer pays exactly $1,000,000.|Critical|Prompt Text|The net loss perfectly exhausts the policy limit.|FALSE|Authorized a payment above the $1M limit.|5|
|8|Concludes the Insured's total out-of-pocket cost is exactly $50,000.|Critical|Prompt Text|The insured pays their deductible; the insurer covers the rest up to the limit.|FALSE|Forced the insured to pay an additional amount over the deductible.|3|
|9|Explains the mechanics of "Defense Within Limits" in the memo narrative.|Minor|141781...LAWYERS-PROFESSIONAL-LIABILITY.pdf, PDF Page 2|Demonstrates comprehension of the core policy feature.|FALSE|Provided numbers without explaining the eroding limit concept.||
|10|Formats output as a professional claims apportionment memo.|Minor|Prompt Text|Required output styling.|FALSE|Provided an unstructured data dump.||

#### 4. Model Analysis

This scenario tests a model's understanding of "Defense Within Limits" (or shrinking limits), a critical concept in professional liability. Standard General Liability policies pay defense costs in addition to the policy limit. Models trained on general insurance data will routinely assume the insurer pays the $150k defense costs in full, and then pays the $900k settlement out of the $1M limit (totalling $1.05M paid by insurer). The attached policy specifically notes on Page 2 that claim expenses reduce the limit of liability and are subject to the deductible. A successful model must catch this nuance, sum the total loss, apply the deductible, and recognize that the remaining balance perfectly exhausts the eroding $1M limit.

#### 5. Golden Response

CLAIMS APPORTIONMENT MEMORANDUM

TO: Claim File

FROM: Professional Liability Claims Adjuster

DATE: October 24, 2026

SUBJECT: Final Apportionment – Malpractice Settlement

1. Policy Architecture Review

The insured's Lawyers Professional Liability policy features a $1,000,000 Limit of Liability and a $50,000 Deductible. Crucially, as outlined on Page 2 of the policy Preface, this is an eroding limits policy: "Claim expenses reduce this policy's limits of liability and are subject to the policy's deductible." This means that the $1,000,000 limit is the absolute maximum the insurer will pay for the combined total of defense costs and the final settlement.

2. Loss Aggregation and Deductible Application

- Total Defense Costs Incurred: $150,000
    
- Final Settlement Amount: $900,000
    
- Total Gross Loss: $1,050,000
    

Because the deductible applies to claim expenses as well as damages, the insured is responsible for the first $50,000 of the total gross loss.

- Insured's Out-of-Pocket Deductible: $50,000
    
- Net Loss Subject to Policy Limit: $1,050,000 - $50,000 = $1,000,000.
    

3. Final Insurer Apportionment

The net loss remaining after the deductible is $1,000,000. Because the policy's Limit of Liability is $1,000,000, the remaining loss perfectly exhausts the available limit.

- Total Amount Paid by Insurer: $1,000,000
    

Conclusion: The claim is fully resolved. The insured will pay their $50,000 deductible, and the insurer will pay the remaining $1,000,000, thereby exhausting the policy limit for this claim.

### 

---

Example 23: Employment Practices Liability (EPLI) Sub-limit Allocation

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Underwriting
    
- Workflow: Coverage Verification
    
- Prompt Type: Coverage Determination
    
- Difficulty: Medium
    

#### 2. Prompt

You are a Management Liability Underwriter/Adjuster. The current date is October 24, 2026.

Your insured, the "Named Entity", has been hit with an Immigration Practices Claim. They have incurred $120,000 in defense costs to date. Based strictly on the Declarations Page of the attached Employment Practices Liability policy, determine the maximum amount the Insurer will pay for this specific claim, and state how much the Insured must pay out-of-pocket as their retention.

Attached Files (Context):

- epli-policy-specimen.pdf – Public URL: https://www.abais.com/docs/default-source/small-business/epl/epli-policy-specimen.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the specific Sub-Limit for Immigration Practices Coverage as $100,000.|Critical|epli-policy-specimen.pdf, PDF Page 2|Item 3B explicitly sets a $100k sub-limit for Immigration Practices.|FALSE|Used the general, uncapped Policy Limit.||
|2|Identifies the Retention for Insuring Agreement C (Immigration Practices) as $10,000.|Critical|epli-policy-specimen.pdf, PDF Page 2|Item 4C sets the specific retention for this agreement.|FALSE|Used the $0 or general retention.||
|3|Notes that "Costs of Defense" reduce the Limit of Liability and are applied against the Retention.|Major|epli-policy-specimen.pdf, PDF Page 2|The NOTE at the top of the declarations page confirms this is a defense-within-limits policy.|FALSE|Assumed defense costs were unlimited or outside the limit.||
|4|Determines the Insured's total out-of-pocket cost (Retention) is $10,000.|Major|Prompt Text|The insured pays the first $10k of the defense costs.|FALSE|Calculated a different retention amount.|1|
|5|Calculates the remaining gross claim amount as $110,000.|Minor|Prompt Text|$120,000 total costs - $10,000 retention = $110,000.|FALSE|Math error.|3|
|6|Applies the $100,000 Sub-Limit to the remaining claim amount.|Critical|Prompt Text|The insurer's payout is strictly capped by the sub-limit.|FALSE|Authorized a payout of $110,000.|10|
|7|Concludes the maximum amount the Insurer will pay is $100,000.|Critical|Prompt Text|The payout hits the sub-limit cap.|FALSE|Paid more or less than the hard cap.|5|
|8|Identifies that the Insured is responsible for the remaining $10,000 in excess of the sub-limit.|Minor|Policy Principles|Any costs above the sub-limit revert to the insured.|FALSE|Assumed the insurer covers the overage.|6|
|9|Explicitly references "Item 3B" and "Item 4C" from the Declarations.|Minor|epli-policy-specimen.pdf, PDF Page 2|Demonstrates exact document mapping.|FALSE|Extracted numbers without citing the document sections.||
|10|Adopts a professional, objective tone appropriate for a coverage determination.|Minor|Prompt Text|Required output styling.|FALSE|Provided an unstructured text block.||

#### 4. Model Analysis

This evaluation requires the model to parse a Declarations page, locate a specific peril (Immigration Practices), and apply the corresponding sub-limits and retentions. Models often fail by applying the general retention (Item 4A) instead of the specific Insuring Agreement C retention (Item 4C). Furthermore, they struggle with the interaction between the retention and the sub-limit. A base model might deduct the $10k retention from the $100k sub-limit and state the insurer pays $90,000. However, standard policy mechanics dictate that the retention is paid first out of the gross loss ($120k), leaving $110k, which is then capped by the $100k sub-limit, meaning the insurer pays the full $100k sub-limit.

#### 5. Golden Response

COVERAGE DETERMINATION

TO: Named Entity

FROM: Management Liability Adjuster

DATE: October 24, 2026

SUBJECT: Immigration Practices Claim – Sub-limit and Retention Application

Based on our review of your Employment Practices Liability Declarations page, we have determined the coverage parameters for your recent Immigration Practices Claim resulting in $120,000 of incurred defense costs.

1. Applicable Limits and Retentions

According to the top of the Declarations page, this policy includes "Defense Within Limits," meaning that your Costs of Defense are applied against your Retention and reduce the available Limit of Liability.

- Sub-Limit: Per Item 3.B, the Sub-Limit of Liability for Immigration Practices Coverage is strictly capped at $100,000.
    
- Retention: Per Item 4.C, the Retention for Each and Every Claim under Insuring Agreement C (Immigration Practices Coverage) is $10,000.
    

2. Financial Apportionment

The total incurred Costs of Defense amount to $120,000.

- Insured's Out-of-Pocket Retention: You are responsible for the first $10,000 of these costs to satisfy your retention.
    
- Remaining Claim: After your retention is met, $110,000 of costs remain.
    
- Insurer's Maximum Payout: Because the remaining costs ($110,000) exceed the applicable Sub-Limit for this specific coverage ($100,000), the Insurer's payment is capped. The maximum amount the Insurer will pay for this claim is $100,000.
    

Note: You will be responsible for the remaining $10,000 that exceeds the policy's sub-limit, making your total out-of-pocket exposure $20,000 ($10k retention + $10k overage).

### 

---

Example 24: Parametric Insurance & Index Deviation

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Agent/Broker
    
- Workflow: Coverage Verification
    
- Prompt Type: Policy Mechanics Explanation
    
- Difficulty: Low
    

#### 2. Prompt

You are a Commercial Broker. The current date is October 24, 2026.

Your client is an agricultural business exploring alternative risk transfer options. They are reviewing a Parametric Insurance Policy from Bajaj General Insurance. They ask you: "If a severe drought ruins $50,000 of our crops, does this policy require us to prove the exact dollar amount of the damaged crops to receive a payout?"

Using the "Scope of Cover" section in the attached Parametric Insurance Policy wording, answer their question directly and explain exactly what triggers a payment under this policy.

Attached Files (Context):

- Policy-Wordings.pdf – Public URL: https://www.bajajgeneralinsurance.com/download-documents/commercial-insurance/Parametric-Insurance/Policy-Wordings.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Answers "No" directly to the client's question about proving the exact dollar amount.|Critical|Policy-Wordings.pdf, PDF Page 7|Parametric policies do not require traditional loss adjustment or proof of specific physical damage.|FALSE|Stated the client must submit an itemized loss report.||
|2|Explains that the policy pays based on the "deviation of an Observed Index".|Major|Policy-Wordings.pdf, PDF Page 7|Exact wording from the Scope of Cover (Section B).|FALSE|Hallucinated a standard indemnity process.|10|
|3|Explains that the payment amount is "as stated in the Policy Schedule" or "Term Sheet".|Major|Policy-Wordings.pdf, PDF Page 7|Payouts are pre-agreed based on the index.|FALSE|Suggested the payout fluctuates based on the adjuster's estimate.|1|
|4|Mentions that the cover relies on an "Index Risk Period" or "Index Phase Period".|Minor|Policy-Wordings.pdf, PDF Page 7|Contextualizes the timing of the parametric trigger.|FALSE|Omitted the temporal constraint of the index.||
|5|Adopts an advisory, client-facing tone.|Minor|Prompt Text|Required output styling.|FALSE|Responded with a technical data dump.||
|6|Specifies that the deviation must occur within the Policy Period.|Minor|Policy-Wordings.pdf, PDF Page 7|Contextual requirement for the parametric trigger.|FALSE|Omitted policy period constraint.|3|
|7|Confirms that traditional damage adjustment is bypassed entirely.|Major|Policy Principles|Shows true understanding of parametric versus indemnity insurance.|FALSE|Stated an adjuster still needs to verify damages.|10|
|8|Does not falsely promise that $50,000 will be the exact payout.|Critical|Prompt Text|The prompt states the loss is $50,000, but the payout is dictated by the index, not the loss.|FALSE|Told the client they would get $50,000.|2|
|9|Explains the underlying weather parameter concept (e.g., rainfall metrics).|Minor|Insurance Principles|Provides practical context to the abstract index.|FALSE|Provided no practical example of what an index is.|1|
|10|Drafts the response as a clear broker-to-client email or letter.|Minor|Prompt Text|Ensures the format matches the persona request.|FALSE|Bulleted list with no greetings.||

#### 4. Model Analysis

Parametric insurance represents a fundamental shift from traditional indemnity insurance. Instead of paying for actual verified losses, it pays a pre-agreed sum if a specific objective metric (like rainfall inches or windspeed) hits a certain threshold. Base LLMs, heavily weighted toward traditional property/casualty adjusting protocols, often hallucinate that the insured must still prove their physical loss. They struggle to extract and apply the pure parametric logic found in the "Scope of Cover" section, frequently merging it with standard claims adjustment advice.

#### 5. Golden Response

SUBJECT: Re: How Your Parametric Insurance Policy Pays Claims

Dear Client,

To answer your question directly: No, under this Parametric Insurance Policy, you do not need to prove the exact $50,000 dollar amount of your damaged crops to receive a payout, nor do you have to wait for an adjuster to inspect the physical damage.

Unlike traditional insurance, which indemnifies you for your specific physical loss, parametric insurance operates on an entirely different mechanism. According to the "Scope of Cover" (Section B) in the attached Bajaj General Insurance policy wording, the company agrees to make a payment solely based on the occurrence of a "deviation of an Observed Index."

If a specific weather parameter (such as a lack of rainfall measured at a local weather station) deviates beyond the threshold agreed upon in your Policy Schedule during the "Index Risk Period," the policy automatically triggers. The amount you are paid is a pre-agreed, fixed sum "as stated in the Term Sheet / Policy Schedule."

In short, the policy pays out based on the severity of the weather event itself (the data index), rather than the specific financial calculation of the damage it caused to your farm. This allows for much faster payouts to help your business recover immediately.

Please let me know if you would like to look at pricing for different index triggers.

Sincerely,

Commercial Broker

### 

---

Example 25: Event Cancellation & Pandemic Exclusions

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Underwriting
    
- Workflow: Coverage Verification
    
- Prompt Type: Policy Wording Analysis
    
- Difficulty: Medium
    

#### 2. Prompt

You are an Underwriting Assistant. The current date is October 24, 2026.

A music festival organizer is purchasing an Event Cancellation policy from Universal Sompo General Insurance. They are concerned about a potential new wave of COVID-19 shutting down their festival. Review the attached policy wording. Is Pandemic/Communicable Disease covered by default under Section I (Cancellation and Abandonment)? If not, how can the insured obtain this coverage according to the policy document?

Attached Files (Context):

- event-cancellation-insurance-policy-cis.pdf – Public URL: https://www.universalsompo.com/assets/file/event-cancellation-insurance-policy/event-cancellation-insurance-policy-cis.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|States that Pandemic/Contagious disease is NOT covered by default.|Critical|event-cancellation-insurance-policy-cis.pdf, PDF Page 1|The base policy contains Exclusion (h) for Pandemic.|FALSE|Asserted that pandemic is a standard covered peril.||
|2|Identifies that Exclusion (h) pertains to Section I (Cancellation and Abandonment).|Minor|event-cancellation-insurance-policy-cis.pdf, PDF Page 1|Maps the exclusion to the correct section.|FALSE|Cited the wrong exclusion letter.|10|
|3|Explains that the insured must purchase the "Cancellation arising out of Pandemic" extension/endorsement.|Critical|event-cancellation-insurance-policy-cis.pdf, PDF Page 1|The document explicitly outlines how to buy back the coverage.|FALSE|Stated the coverage is impossible to obtain.|10|
|4|Mentions that obtaining the coverage requires the "payment of additional premium".|Major|event-cancellation-insurance-policy-cis.pdf, PDF Page 1|The buy-back is not free; it requires additional premium as stated in the Schedule.|FALSE|Implied the endorsement could be added at no cost.|2|
|5|Explains that the endorsement effectively deletes Exclusion (h) from the policy.|Major|event-cancellation-insurance-policy-cis.pdf, PDF Page 1|Explains the mechanical function of the buy-back endorsement.|FALSE|Failed to explain how the endorsement alters the base text.|2|
|6|Notes the coverage applies to both actual and "threat or fear thereof" of a pandemic.|Minor|event-cancellation-insurance-policy-cis.pdf, PDF Page 1|Pulls the exact nuance from the endorsement clause.|FALSE|Only stated it covers an actual outbreak.|2|
|7|Ensures the response specifically addresses Section I (Cancellation and Abandonment).|Minor|Prompt Text|Adheres to the prompt's specific scoping.|FALSE|Addressed public liability instead of event cancellation.||
|8|Confirms the risk specifically addresses the client's COVID-19 concern.|Major|Prompt Text|Tailors the analysis to the client's direct question regarding COVID-19.|FALSE|Provided a generic response without acknowledging COVID.|10|
|9|Advises the client on the next step to secure the coverage.|Minor|Underwriting Principles|Practical application of the underwriting role.|FALSE|Left the client without actionable advice.|2|
|10|Formats output professionally as an Underwriting Analysis.|Minor|Prompt Text|Required tone and format.|FALSE|Output only one-word answers.||

#### 4. Model Analysis

Event cancellation policies frequently use "buy-back" endorsements, where a peril is explicitly excluded in the main body of the policy but can be added back via a specific endorsement for an additional premium. Language models often struggle to synthesize these conflicting statements. A model might read the exclusion and definitively state there is no coverage, entirely missing the subsequent paragraph detailing the buy-back option. Alternatively, the model might see the "Cancellation arising out of Pandemic" heading and assume it is included by default, missing the prerequisite requirement to pay an additional premium.

#### 5. Golden Response

COVERAGE ANALYSIS MEMORANDUM

TO: Music Festival Organizer

FROM: Underwriting Assistant

DATE: October 24, 2026

SUBJECT: Pandemic/Communicable Disease Coverage

Thank you for your inquiry regarding COVID-19 and communicable disease coverage under the Universal Sompo Event Cancellation policy.

Default Coverage Status:

By default, Pandemic and/or contagious diseases are not covered. If the festival is shut down due to a COVID-19 outbreak, the base policy will not respond because Section I (Cancellation and Abandonment) contains a specific exclusion—Exclusion (h)—that precludes losses arising from a Pandemic or contagious disease.

How to Obtain Coverage:

You can obtain this coverage by purchasing a specific policy extension. The policy offers an endorsement titled "Cancellation arising out of Pandemic."

In consideration of the payment of an additional premium, the insurer will agree to indemnify you for losses resulting from a Pandemic or contagious disease (or the threat/fear thereof). Mechanically, purchasing this extension means that Exclusion (h) will be explicitly deleted from your policy, thereby restoring coverage for this specific risk.

If you would like to proceed with this protection, we will calculate the additional premium required to add this endorsement to your Schedule.

### 

---

Example 26: Ocean / Inland Marine Cargo and Climatic Conditions

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Claims Handling
    
- Workflow: Claims Adjudication
    
- Prompt Type: Coverage Determination Memo
    
- Difficulty: Medium
    

#### 2. Prompt

You are a Marine Cargo Claims Adjuster. The current date is October 24, 2026.

Your insured shipped a valuable cargo of sensitive electronics. During the ocean and inland transit, the cargo sat in a port container yard for two weeks under extreme, unexpected heat. The extreme temperatures caused the electronics to warp and fail. The insured has submitted a claim under their Universal Sompo Marine Cargo Open Transit Policy.

Review the attached policy prospectus. Based on the explicit clauses contained within the policy, determine if this loss is covered.

Attached Files (Context):

- marine-cargo-open-transit-policy-prospectus.pdf – Public URL: https://www.universalsompo.com/assets/file/marine-cargo-open-transit-policy/marine-cargo-open-transit-policy-prospectus.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the "Climatic Conditions Clause".|Critical|marine-cargo-open-transit-policy-prospectus.pdf, PDF Page 12|The specific clause governing weather and temperature damage.|FALSE|Hallucinated a generic "Act of God" clause.||
|2|Extracts the rule that the policy excludes loss or damage on account of climatic or atmospheric conditions.|Major|marine-cargo-open-transit-policy-prospectus.pdf, PDF Page 12|Exact wording of the exclusion.|FALSE|Assumed weather is a standard covered peril.|10|
|3|Extracts the rule that the policy specifically excludes "extremes of temperature".|Critical|marine-cargo-open-transit-policy-prospectus.pdf, PDF Page 12|Directly applies the exclusion to the heat damage in the prompt.|FALSE|Missed the temperature specification.|1|
|4|Concludes definitively that the claim for the warped electronics is DENIED.|Critical|Prompt Text|The final claims determination.|FALSE|Approved the claim.|2|
|5|Explains the rationale connecting the extreme heat to the specific policy exclusion.|Major|Prompt Text|Ensures the model synthesizes the facts with the contract.|FALSE|Denied the claim for an unrelated reason (e.g., delay).|2|
|6|Does not hallucinate coverage based on general "All Risk" assumptions.|Major|Insurance Principles|Standard "All Risk" cargo policies still carry specific exclusions that must be applied.|FALSE|Stated that because it was "unexpected," it must be covered.|3|
|7|Recognizes the policy governs both Inland & Import/Export transit.|Minor|marine-cargo-open-transit-policy-prospectus.pdf, PDF Page 12|Identifies the broad scope of the policy type.|FALSE|Stated the claim is denied because it happened in a port yard (inland) instead of at sea.||
|8|Maintains an objective, professional claims adjuster tone.|Minor|Prompt Text|Required persona styling.|FALSE|Used an overly sympathetic or informal tone.||
|9|Avoids requesting an adjuster inspection, as the exclusion applies regardless of the severity.|Minor|Claims Adjusting Principles|The peril itself is excluded, making physical inspection of the damage moot for coverage.|FALSE|Requested a field adjuster verify the warping before denying.|3|
|10|Formats output as a Coverage Determination Memo.|Minor|Prompt Text|Output styling requirement.|FALSE|Provided a bulleted list of facts.||

#### 4. Model Analysis

Language models often fail when evaluating marine cargo claims because they rely on generalized assumptions about "All Risk" transit policies. A standard model may assume that because the extreme heat was an unexpected external event damaging the cargo in transit, it should be a covered peril. The model must accurately locate and apply the specific "Climatic Conditions Clause" on PDF Page 12 of the prospectus, which explicitly carves out "extremes of temperature." Models frequently overlook this specific exclusion and erroneously authorize the claim based on the general premise of transit insurance.

#### 5. Golden Response

COVERAGE DETERMINATION MEMO

TO: Marine Cargo Claims File

FROM: Marine Cargo Claims Adjuster

DATE: October 24, 2026

SUBJECT: Claim Denial – Damage via Extreme Temperature

1. Claim Overview

The insured has submitted a claim for a shipment of sensitive electronics that sustained severe warping and failure due to extreme heat exposure while sitting in a port container yard. The applicable policy is the Universal Sompo Marine Cargo Open Transit Policy.

2. Policy Language and Application

While the Marine Cargo Open Transit policy provides broad coverage for goods in transit, it is subject to specific exclusions. Upon review of the policy prospectus, Clause 12 outlines the Climatic Conditions Clause.

This clause explicitly states: "The Policy excludes loss or damage on account of climatic or atmospheric conditions or extremes of temperature."

3. Coverage Determination

Because the sole proximate cause of the damage to the sensitive electronics was the extreme, unexpected heat (an extreme of temperature) experienced at the container yard, the loss falls squarely within the Climatic Conditions Clause exclusion.

Therefore, there is no coverage afforded for this loss under the policy, and the claim must be DENIED.

### 

---

Example 27: Catastrophe Excess of Loss Reinsurance Layering

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Reinsurance
    
- Workflow: Quantum Calculation
    
- Prompt Type: Treaty Recovery Statement
    
- Difficulty: High
    

#### 2. Prompt

You are a Reinsurance Analyst. The current date is October 24, 2026.

A massive hurricane generated a gross loss of $6.5 billion for your company. According to your Corporate Catastrophe Excess-of-Loss Reinsurance Treaty (attached), calculate the exact dollar amount of the recovery you will receive from this specific treaty across all applicable layers. Show your step-by-step mathematical calculations for each layer.

Attached Files (Context):

- tm2510932d1_ars.pdf – Public URL: https://www.sec.gov/Archives/edgar/data/86312/000110465925032230/tm2510932d1_ars.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Extracts the base retention amount as $4.0 billion.|Major|tm2510932d1_ars.pdf, PDF Page 31|The treaty explicitly states the first layer attaches in excess of $4.0 billion.|FALSE|Failed to identify the attachment point.||
|2|Identifies the limits of the First Layer as $4.0 billion to $5.0 billion.|Minor|tm2510932d1_ars.pdf, PDF Page 31|Identifies the boundaries of the first layer.|FALSE|Assumed the first layer was uncapped.|10|
|3|Extracts the recovery percentage for the First Layer as 80%.|Major|tm2510932d1_ars.pdf, PDF Page 31|The treaty dictates 80% participation for this specific band.|FALSE|Used the 95% or 100% participation figures from higher bands.|1|
|4|Calculates the First Layer recovery as $800,000,000.|Critical|Prompt Text; Treaty|$1.0B (amount in layer) * 0.80 = $800M.|FALSE|Math error or applied 80% to the entire gross loss.|1|
|5|Identifies the limits of the Second Layer as $5.0 billion to $7.5 billion.|Minor|tm2510932d1_ars.pdf, PDF Page 31|Identifies the boundaries of the second layer.|FALSE|Failed to map the second band.||
|6|Extracts the recovery percentage for the Second Layer as 95%.|Major|tm2510932d1_ars.pdf, PDF Page 31|The treaty dictates 95% participation for this specific band.|FALSE|Applied the wrong percentage.|4|
|7|Calculates the loss amount subject to the Second Layer as $1.5 billion.|Major|Prompt Text|Total loss is $6.5B. Layer attaches at $5.0B. ($6.5B - $5.0B = $1.5B inside the layer).|FALSE|Assumed the entire $2.5B layer limit was exhausted.|4|
|8|Calculates the Second Layer recovery as $1,425,000,000.|Critical|Prompt Text; Treaty|$1.5B (amount in layer) * 0.95 = $1.425B.|FALSE|Math error.|5|
|9|Correctly states that the Third Layer ($7.5B to $8.0B) is not triggered.|Minor|Prompt Text|The $6.5B gross loss does not reach the $7.5B attachment point for the 100% layer.|FALSE|Attempted to calculate a recovery in the third layer.||
|10|Calculates the Total Final Recovery amount as $2,225,000,000.|Critical|Prompt Text|$800M (Layer 1) + $1.425B (Layer 2) = $2.225B.|FALSE|Incorrect sum due to failure in previous layering steps.|3|

#### 4. Model Analysis

Models frequently fail to process banded Excess of Loss treaties correctly. They routinely apply a single participation rate to the entire gross loss, rather than isolating the specific dollar amounts that pierce each respective attachment point and applying the distinct co-participation percentages to those tranches. A model might see the 95% figure and attempt to apply it to the entire $2.5 billion excess of the $4.0 billion retention, resulting in massive hallucinations in recovery calculations.

#### 5. Golden Response

TREATY RECOVERY CALCULATION

DATE: October 24, 2026

SUBJECT: Corporate Catastrophe Excess-of-Loss Reinsurance Recovery

Based on the Corporate Catastrophe Excess-of-Loss Reinsurance Treaty terms, our recovery for the $6,500,000,000 gross hurricane loss is calculated in banded layers. The treaty provides varying co-participation recovery percentages as the loss escalates.

Base Retention:

The company retains the first $4,000,000,000 of the gross loss.

Layer 1 ($4.0B to $5.0B):

- Total Capacity of Layer: $1,000,000,000
    
- Amount of Loss in Layer: $1,000,000,000
    
- Recovery Percentage: 80%
    
- Layer 1 Recovery: $1,000,000,000 ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) 0.80 = $800,000,000
    

Layer 2 ($5.0B to $7.5B):

- Total Capacity of Layer: $2,500,000,000
    
- Amount of Loss in Layer: The gross loss is $6.5B. Subtracting the $5.0B exhausted by the retention and Layer 1 leaves $1,500,000,000 subject to this layer.
    
- Recovery Percentage: 95%
    
- Layer 2 Recovery: $1,500,000,000 ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) 0.95 = $1,425,000,000
    

Layer 3 ($7.5B to $8.0B):

- Recovery Percentage: 100%
    
- Layer 3 Recovery: $0. (The gross loss of $6.5B did not reach the $7.5B attachment point for this layer).
    

Total Final Recovery:

- $800,000,000 (Layer 1) + $1,425,000,000 (Layer 2) = $2,225,000,000
    

### 

---

Example 28: Market Conduct Examination (Annuity Suitability)

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Compliance
    
- Workflow: Market Analysis
    
- Prompt Type: Compliance Exam Finding
    
- Difficulty: Low
    

#### 2. Prompt

You are a State Insurance Department Market Conduct Examiner. The current date is October 24, 2026.

You are conducting a targeted examination of a life insurance company's annuity sales practices. According to the Chapter 23 standards in the NAIC Market Regulation Handbook, what specific minimum training requirement must you verify that the company's producers have met before being allowed to sell annuity products?

Attached Files (Context):

- publication-market-reg-hb.pdf – Public URL: https://content.naic.org/sites/default/files/publication-market-reg-hb.pdf
    

#### 3. Rubric

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the requirement for a "one-time 4 hour credit training course".|Critical|publication-market-reg-hb.pdf, PDF Page 8 (via NAIC standard)|Exact requirement listed in the handbook for producer continuing education regarding annuities.|FALSE|Hallucinated a generic "annual ethics training."||
|2|Specifies that the training must be completed before the producer is allowed to sell the product.|Major|publication-market-reg-hb.pdf, PDF Page 8|Temporal requirement of the compliance check.|FALSE|Implied the training could be done within the first year of selling.||
|3|Notes that the examiner must compare producer continuing education files to state insurance department records.|Minor|publication-market-reg-hb.pdf, PDF Page 8|The specific methodology the examiner uses to verify compliance.|FALSE|Omitted the verification methodology.||
|4|Mentions that the company must have "adequate procedures in place" to verify completion.|Minor|publication-market-reg-hb.pdf, PDF Page 8|Places the burden of systemic verification on the regulated entity.|FALSE|Placed the burden solely on the producer.||
|5|Uses professional regulatory tone appropriate for an Examiner.|Minor|Prompt Text|Required output styling.|FALSE|Used conversational or informal language.||
|6|Explicitly references Chapter 23 (Life and Annuity Examination).|Minor|Prompt Text|Ties the answer back to the prompt's specific constraint.|FALSE|Answered generally without referencing the chapter.||
|7|Does not hallucinate non-existent FINRA Series 6/7 requirements for fixed annuities.|Major|Insurance Principles|The NAIC standard deals with state CE, not federal securities licenses unless variable.|FALSE|Hallucinated a Series 7 license requirement.||
|8|Mentions applicable state statutes, rules, and regulations.|Minor|publication-market-reg-hb.pdf, PDF Page 8|Contextualizes the NAIC handbook as a guide for state-specific laws.|FALSE|Treated the handbook as federal law.||
|9|Focuses solely on the training requirement, avoiding unprompted analysis of actual suitability forms.|Minor|Prompt Text|Keeps the response tightly scoped to the prompt.|FALSE|Wrote a lengthy analysis on how to review a client's financial profile.||
|10|Accurately identifies the regulated entity as the primary subject of the exam procedure.|Minor|Regulatory Principles|Market conduct exams audit the company, not just individual agents.|FALSE|Addressed the response as an audit of a single agent.||

#### 4. Model Analysis

When queried about regulatory standards, base LLMs often hallucinate general compliance best practices (e.g., "ensure the agent acts in the client's best interest" or "check for a valid license") rather than extracting the exact numerical and procedural requirements mandated by the NAIC. For this prompt, the model must specifically extract the "one-time 4 hour credit training course" requirement from the handbook. Models frequently miss this exact temporal and credit-hour specification, opting instead for vague summaries of continuing education.

#### 5. Golden Response

MARKET CONDUCT EXAM FINDING: PRODUCER TRAINING VERIFICATION

DATE: October 24, 2026

SUBJECT: Verification of Producer Training for Annuity Sales

Pursuant to the standards outlined in Chapter 23 (Conducting the Life and Annuity Examination) of the NAIC Market Regulation Handbook, the examiner must verify that the regulated entity has adequate procedures in place to ensure compliance with producer training requirements.

Specifically, the examiner must verify that any insurance producer who engages in the sale of annuity products has met the requirement to complete a one-time 4 hour credit training course in accordance with applicable state statutes, rules, and regulations.

To execute this, the examiner must compare the data in the regulated entity's producer continuing education files against the applicable data in the state insurance department's records. Crucially, the examiner must verify that the company's procedures ensure this 4-hour training is completed before the producer is allowed to solicit or sell the annuity product.

### 

---

Example 29: Health Insurance SBC Out-of-Network Limitations

#### 1. Metadata

- Task Type: Workflow
    
- Category / Domain: Agent/Broker
    
- Workflow: Benefit Explanation
    
- Prompt Type: Patient Advisory Letter
    
- Difficulty: Medium
    

#### 2. Prompt

You are a Health Benefits Consultant. The current date is October 24, 2026.

A family of four insured under the attached Blue Cross Blue Shield of Louisiana IND POS plan wants to use an out-of-network surgical center for a specialized, non-emergency procedure. They want to know their absolute maximum financial exposure for out-of-network services this year.

Using the attached Summary of Benefits and Coverage (SBC), write an advisory letter detailing their Out-of-Network Family Deductible, their Out-of-Network Family Out-of-Pocket Limit, and explicitly explain the three types of expenses that do not count toward that out-of-pocket limit (which could increase their exposure).

Attached Files (Context):

- 19636LA0590006-01.pdf – Public URL: https://www.lablue.com/web/applications/sbcportal/sbcs/2025/19636LA0590006-01.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the Out-of-Network Family Deductible as $30,000.|Critical|19636LA0590006-01.pdf, PDF Page 1|Extracted from the "What is the overall deductible?" row.|FALSE|Quoted the in-network deductible ($10,000) or individual deductible.||
|2|Identifies the Out-of-Network Family Out-of-Pocket Limit as $48,000.|Critical|19636LA0590006-01.pdf, PDF Page 1|Extracted from the "What is the out-of-pocket limit" row.|FALSE|Quoted the in-network OOP limit ($16,000).||
|3|Identifies that "Premiums" do not count toward the out-of-pocket limit.|Major|19636LA0590006-01.pdf, PDF Page 1|Extracted from the "What is not included in the out-of-pocket limit" row.|FALSE|Omitted premiums.||
|4|Identifies that "Balance Billing Charges" do not count toward the out-of-pocket limit.|Critical|19636LA0590006-01.pdf, PDF Page 1|The primary risk of out-of-network care.|FALSE|Omitted balance billing.||
|5|Identifies that "Health Care this plan doesn't cover" does not count toward the out-of-pocket limit.|Major|19636LA0590006-01.pdf, PDF Page 1|Extracted from the "What is not included" row.|FALSE|Omitted non-covered healthcare.||
|6|Explains the concept of "balance billing" in the context of out-of-network providers.|Major|19636LA0590006-01.pdf, PDF Page 1|The SBC explains balance billing as the difference between the provider's charge and what the plan pays.|FALSE|Failed to define balance billing.|3|
|7|Warns the family that their actual exposure could exceed $48,000 due to balance billing.|Critical|Health Insurance Principles|Synthesizes the OOP limit with the balance billing exclusion to answer the prompt's question about "absolute maximum exposure."|FALSE|Stated $48,000 is the absolute maximum they will pay under any circumstance.|1|
|8|Notes that individual family members must meet their own individual deductibles until the family total is met.|Minor|19636LA0590006-01.pdf, PDF Page 1|Included in the deductible explanation text.|FALSE|Implied one person must meet the $30k deductible alone.|10|
|9|Explicitly states the plan type is "IND POS".|Minor|19636LA0590006-01.pdf, PDF Page 1|Confirms data extraction from the header.|FALSE|Hallucinated an HMO or PPO plan type.||
|10|Formats output as an advisory letter to the client.|Minor|Prompt Text|Required output styling.|FALSE|Provided an unstructured list.||

#### 4. Model Analysis

When evaluating health insurance maximums, models often conflate "In-Network" and "Out-of-Network" figures due to the dense tabular structure of an SBC document. A model is likely to accidentally extract the $16,000 In-Network Family OOP limit instead of the $48,000 Out-of-Network limit. More importantly, when asked for the "absolute maximum financial exposure," models tend to simply quote the Out-of-Pocket Maximum figure, failing to synthesize the critical warning that "Balance Billing" does not apply to this limit. Because out-of-network providers can balance bill for amounts above the insurer's allowed rate, the family's true financial exposure is theoretically unlimited. A successful model must connect the extracted OOP limit with the extracted exclusions to provide accurate advisory counsel.

#### 5. Golden Response

HEALTH BENEFITS ADVISORY LETTER

DATE: October 24, 2026

TO: Client Family

SUBJECT: Out-of-Network Financial Exposure Analysis

Dear Client,

Thank you for reaching out regarding your planned out-of-network surgical procedure. Based on a review of your Blue Cross Blue Shield of Louisiana IND POS plan Summary of Benefits and Coverage (SBC), using an out-of-network facility significantly alters your cost-sharing and financial exposure.

Here is the breakdown of your out-of-network limits for the family:

- Out-of-Network Family Deductible: Your family deductible for out-of-network services is $30,000. You must pay all costs for these services up to this amount before the plan begins to pay. (Note: individual family members must meet their own individual deductibles until the overall $30,000 family total is met).
    
- Out-of-Network Family Out-of-Pocket Limit: The stated out-of-pocket limit for your family when using out-of-network providers is $48,000.
    

Your Absolute Maximum Exposure:

While $48,000 is the stated out-of-pocket limit, it is critical to understand that your actual financial exposure could be significantly higher, and is technically unlimited. This is because three specific types of expenses do not count toward your $48,000 out-of-pocket limit:

1. Your monthly Premiums.
    
2. Health Care this plan doesn't cover (services deemed experimental or excluded).
    
3. Balance Billing Charges.
    

The Risk of Balance Billing:

Because your surgical center is out-of-network, they do not have a contracted rate with BCBS. The plan will only pay a percentage of what it considers a "reasonable" charge. The surgical center can—and likely will—bill you for the difference between their total charge and what the plan pays. This is known as "balance billing." Because balance billing charges do not count toward your $48,000 out-of-pocket limit, you will be responsible for 100% of those excess charges, regardless of how much you have already paid this year.

Please let me know if you would like assistance finding an in-network facility to strictly cap your exposure.

Sincerely,

Health Benefits Consultant

### 

---

Example 30: Cyber Insurance Extortion and Ransomware

#### 1. Metadata

- Task Type: Capability
    
- Category / Domain: Claims Handling
    
- Workflow: Coverage Verification
    
- Prompt Type: Claims Response Brief
    
- Difficulty: Medium
    

#### 2. Prompt

You are a Cyber Claims Adjuster. The current date is October 24, 2026.

Your insured, a manufacturing company, has suffered a severe ransomware attack. Their entire database has been encrypted by cybercriminals who are demanding a $50,000 cryptocurrency payment in exchange for the decryption key. The insured wants to know if their Klapton Cyber Insurance policy will reimburse them for this $50,000 payment to the criminals.

Using the attached policy wording, specifically reference the exact Chapter and Definition that provides this coverage, and confirm whether the policy reimburses ransom payments made to cybercriminals to regain access to encrypted data.

Attached Files (Context):

- KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf – Public URL: https://www.klapton.com/wp-content/uploads/2025/01/KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with exact PDF pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies Chapter 7 as the relevant section of the policy.|Critical|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|Extracted from the Preamble index.|FALSE|Referenced the wrong chapter or hallucinated a section.||
|2|Identifies the title of Chapter 7 as "Extortion and Ransom Indemnity".|Major|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|Exact text from the policy structure.|FALSE|Used generic terminology instead of the exact title.|10|
|3|Identifies "Ransom Payment Coverage" as the specific definition.|Critical|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|Exact definition term from the document.|FALSE|Failed to name the specific coverage grant.||
|4|Confirms that the policy does reimburse the cost of ransom payments.|Critical|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|The core answer to the insured's question.|FALSE|Stated that ransom payments are illegal and excluded.|2|
|5|Specifies that the reimbursement applies to payments made to "cybercriminals".|Major|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|Extracted directly from the Ransom Payment Coverage definition.|FALSE|Omitted the recipient of the ransom.|3|
|6|Specifies that the purpose must be "to regain access to encrypted data or systems".|Major|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|Extracted directly from the Ransom Payment Coverage definition.|FALSE|Omitted the required condition for the payment.|3|
|7|Concludes that the $50,000 demand is theoretically eligible for reimbursement.|Critical|Prompt Text|Synthesizes the facts of the prompt with the policy definition.|FALSE|Rejected the $50,000 claim.|3|
|8|Mentions that "Technical Support" (Chapter 1) may also be available for forensic investigations.|Minor|KIC-Cyber-Policy-Wording_Final-2024-new-31.1.2025.pdf, PDF Page 15|Provides holistic claims advice based on adjacent text in the document.|FALSE|Ignored the incident response benefits.||
|9|Includes a standard adjuster caveat that payment is subject to actual policy limits, deductibles, and OFAC/sanctions checks.|Minor|Cyber Claims Principles|Professional standard for ransomware claims (cannot guarantee payment without checking sanctions).|FALSE|Guaranteed the payment unconditionally.|6|
|10|Formats output as a professional Claims Response Brief.|Minor|Prompt Text|Required output styling.|FALSE|Provided an unstructured data dump.||

#### 4. Model Analysis

Ransomware claims require careful navigation of policy text. While many base LLMs are hardcoded to advise that paying ransoms is risky or potentially illegal under OFAC regulations, they must accurately reflect the specific insurance contract provided. In this case, the Klapton policy explicitly grants coverage for ransom payments. A successful model will suppress its generalized training regarding the ethics of ransomware payments and strictly extract the "Ransom Payment Coverage" definition from Chapter 7, confirming to the insured that the contract operates exactly as designed to reimburse payments made to cybercriminals to unlock encrypted systems.

#### 5. Golden Response

CYBER CLAIMS RESPONSE BRIEF

TO: Insured (Manufacturing Company)

FROM: Cyber Claims Adjuster

DATE: October 24, 2026

SUBJECT: Ransomware Incident – Extortion Coverage Verification

Thank you for notifying us of the severe ransomware incident affecting your manufacturing database. We understand that cybercriminals have demanded a $50,000 cryptocurrency payment in exchange for the decryption key.

You have requested verification on whether your Klapton Cyber Insurance policy will reimburse this payment.

Coverage Verification:

Yes, your policy does contain provisions to reimburse this type of extortion payment.

According to the Preamble of your policy, Chapter 7 specifically governs "Extortion and Ransom Indemnity." Within the policy definitions, the contract explicitly provides "Ransom Payment Coverage."

The policy defines Ransom Payment Coverage as follows: "Reimburses the insured for the cost of ransom payments made to cybercriminals to regain access to encrypted data or systems."

Because your database was encrypted by cybercriminals and the $50,000 payment is being demanded specifically to regain access to that data, the demand meets the exact definition of this coverage grant. Subject to your specific policy limits, sub-limits, deductibles, and standard regulatory sanctions checks (to ensure the threat actor is not on an OFAC blocked list), the $50,000 payment is eligible for reimbursement.

Please contact our incident response team immediately so we can engage our authorized breach coaches and cryptocurrency negotiators under the Chapter 1 (Cyber Incident Response) provisions before any payment is made.

### 

---

Example 31: Property Risk Modeling (Catastrophe Data Mapping)

#### 1. Metadata

- Sub-domain: Risk Modeling & Actuarial
    
- Difficulty: Medium-High
    
- Themes: Data Dictionary Mapping, Fallback Logic, Replacement Cost
    
- Task Type: Capability
    
- Workflow: Property Portfolio Valuation
    
- Prompt Type: Risk Modeling Calculation
    

#### 2. Prompt

You are a Catastrophe Risk Modeler. The current date is October 24, 2026.

You are evaluating a raw data extract of three buildings in the San Francisco Bay Area to determine the total modeled replacement cost of the portfolio. You must use the specific mapping rules provided by the UrbanSim database documentation to determine the correct Replacement Cost per Square Foot based on the building's Occupancy ID.

Portfolio Data Extract:

- Building Alpha: Occupancy ID: 4 | Total Floor Area: 10,000 SQFT
    
- Building Beta: Occupancy ID: 6 | Total Floor Area: 5,000 SQFT
    
- Building Gamma: Occupancy ID: 99 | Total Floor Area: 2,000 SQFT
    

Calculate the individual replacement cost for each building, and then sum them to provide the Total Portfolio Replacement Cost. Show your math.

Attached Files (Context):

- SimCenter_Asset_Description.html – Public URL: https://nheri-simcenter.github.io/R2D-Documentation/common/testbeds/sf_bay_area/asset_description.html (Documentation containing Table 1.2.1 Mapping rules)
    

#### 3. Rubric

  

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|#|Description|Weight|Sources (with public URLs and Pages)|Justification|Met|Failure Reasoning|Dependent Criteria|
|1|Identifies the mapping rule for Building Alpha (Occupancy ID 4).|Minor|SimCenter_Asset_Description.html, Table 1.2.1 18 (URL: https://nheri-simcenter.github.io/R2D-Documentation/common/testbeds/sf_bay_area/asset_description.html)|ID 4 maps to "Office" at $131.9 / SQFT.|FALSE|Mapped to the wrong category.||
|2|Calculates the replacement cost for Building Alpha as $1,319,000.|Major|Prompt Text|10,000 SQFT multiplied by $131.9.|FALSE|Arithmetic error.|10|
|3|Identifies the mapping rule for Building Beta (Occupancy ID 6).|Minor|SimCenter_Asset_Description.html, Table 1.2.1 18|ID 6 maps to "School" at $142.1 / SQFT.|FALSE|Mapped to the wrong category.||
|4|Calculates the replacement cost for Building Beta as $710,500.|Major|Prompt Text|5,000 SQFT multiplied by $142.1.|FALSE|Arithmetic error.|2|
|5|Identifies the fallback mapping rule for Building Gamma (Occupancy ID 99).|Critical|SimCenter_Asset_Description.html, Table 1.2.1 18|ID 99 is invalid. The text states: "For buildings with missing or invalid occupancy ID, the building was mapped to the default occupancy type (i.e., residential)". Residential is $137.5 / SQFT.|FALSE|Excluded the building from the calculation entirely or hallucinated a zero value.||
|6|Calculates the replacement cost for Building Gamma as $275,000.|Major|Prompt Text|2,000 SQFT multiplied by $137.5.|FALSE|Failed to apply the default residential rate.|4|
|7|Calculates the Total Portfolio Replacement Cost as $2,304,500.|Critical|Prompt Text|$1,319,000 + $710,500 + $275,000 = $2,304,500.|FALSE|Incorrect sum due to failure in previous mapping steps.|1|

#### 4. Model Analysis

This task tests a model's ability to execute precise data dictionary cross-referencing, a core task in catastrophe modeling. A base language model will likely score between forty and sixty percent. Models generally succeed at direct 1-to-1 lookups (e.g., matching ID 4 to the Office rate). The primary vector of failure is Building Gamma. ID 99 does not exist in the literal table rows. A human data analyst reading the documentation will spot the prose instruction above the table dictating that invalid IDs must fallback to the "Default" residential rate of $137.5. Language models, overly fixated on the tabular markdown representation of the table, frequently fail to apply the surrounding prose rules, resulting in Building Gamma being priced at $0, dropped from the portfolio, or hallucinated with a random multiplier.

#### 5. Golden Response

PORTFOLIO REPLACEMENT COST CALCULATION

DATE: October 24, 2026

SUBJECT: SimCenter Bay Area Asset Valuation

Based on the UrbanSim asset description documentation and Table 1.2.1 (Mapping rules for building occupancy type and replacement cost), the portfolio valuation is calculated by mapping each building's Occupancy ID to the corresponding Replacement Cost per Square Foot.

1. Building Alpha

- Occupancy ID: 4
    
- Mapped Class: Office
    
- Rate: $131.9 / SQFT
    
- Calculation: 10,000 SQFT ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) $131.9 = $1,319,000
    

2. Building Beta

- Occupancy ID: 6
    
- Mapped Class: School
    
- Rate: $142.1 / SQFT
    
- Calculation: 5,000 SQFT ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) $142.1 = $710,500
    

3. Building Gamma

- Occupancy ID: 99 (Invalid)
    
- Mapped Class: Default (Residential)
    
- Rule Application: The documentation specifies that "For buildings with missing or invalid occupancy ID, the building was mapped to the default occupancy type (i.e., residential)." The Residential rate is $137.5 / SQFT.
    
- Calculation: 2,000 SQFT ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA+UlEQVR4AeyRMQrCMBSGq266uShugiDOiicRHB10ETdBwcVBvIGH8A4exAOIgiB4ARe/P5CQpi1UdBFa3peX/nl/XpqWoy+ewvzh5f3swmo0bkBWlFhoQQVM+J2rKAcYQBgyThA34DxugviAJexhCDassY+g9RfZhG+WcGWYwQ60QaaR9Sg0S/M30GckOqpIpJml3xhOMIYjuKMyd5Fm1lEXVLRBXVdkfQIpHqHZGruU6XIu5DnolhMb+GYZVdihWEZ71Dvv0hMb+OYeRU3QMa2RVxPaYMpsBHUw4ZvPKFsIjUgmnoxrUCZFqb/KLOQZ/M556mM1f2p+AwAA//9SqAExAAAABklEQVQDAL3jIi/+veipAAAAAElFTkSuQmCC) $137.5 = $275,000
    

4. Total Portfolio Valuation

The total modeled replacement cost is the sum of the three individual buildings.

- $1,319,000 + $710,500 + $275,000 = $2,304,500
    

### The Total Portfolio Replacement Cost is $2,304,500.

---

Example 32: CGL Aggregate Limit Allocation & PCOH Distinction

#### 1. Metadata

- Sub-domain: Claims Handling (Commercial General Liability)  
      
    
- Difficulty: High  
      
    
- Themes: General Aggregate vs. Products-Completed Operations (PCOH) Aggregate, Per-Occurrence Limits, Claim Sequencing  
      
    
- Task Type: Workflow  
      
    
- Workflow: Liability Quantum & Limit Attribution  
      
    
- Prompt Type: Claims Triage and Limit Exhaustion Report  
      
    

#### 2. Prompt

You are a Senior Liability Claims Examiner. The current date is October 24, 2026. You must adjudicate three distinct liability claims against an insured manufacturer and determine the remaining available limits.

  

Policy Parameters (simulated CGL_Policy_CG0001.pdf):

- Each Occurrence Limit: $1,000,000
    
- General Aggregate Limit: $2,000,000
    
- Products-Completed Operations (PCOH) Aggregate Limit: $2,000,000
    
- Deductible: $10,000 per occurrence (applies to indemnity only).
    

Loss Data (simulated Claims_Log_Q4.csv):

- Claim 1 (Jan 15, 2026): A visitor slipped on a wet floor at the factory. Settlement: $1,200,000 indemnity.
    
- Claim 2 (March 10, 2026): A consumer was injured by a toaster that exploded after purchase. Settlement: $800,000 indemnity.
    
- Claim 3 (August 5, 2026): A second consumer was injured by the same toaster model from the same batch. Settlement: $1,500,000 indemnity.
    

Tasks:

- Classify each claim as hitting either the General Aggregate or the PCOH Aggregate, citing the standard ISO CG 00 01 definitions.
    
- Calculate the insurer's net payout for each claim after applying per-occurrence limits and deductibles.
    
- Calculate the final remaining balance for both aggregate limits.
    
- Determine if the "Batch Clause" or "Single Occurrence" logic applies to Claims 2 and 3 under standard ISO form language.
    

#### 3. Model Analysis

Standard models often fail to distinguish between the two separate aggregate buckets in a CGL policy. They frequently apply all liability losses to a single "General Aggregate," failing to recognize that PCOH claims have their own dedicated limit. Additionally, models struggle with the "per occurrence" cap; for Claim 1, an LLM might erroneously pay the full $1,190,000 (after deductible) instead of capping the payout at the $1,000,000 per-occurrence limit.

  

#### 4. Rubric

|   |   |   |   |   |
|---|---|---|---|---|
|#|Description|Weight|Sources|Justification|
|1|Classifies Claim 1 as a "Premises/Operations" loss hitting the General Aggregate.|Major|CGL_Policy_CG0001.pdf|Slip and fall at the factory is a premises loss, not a product-completed op.|
|2|Classifies Claim 2 and Claim 3 as "Products-Completed Operations Hazard" (PCOH).|Major|CGL_Policy_CG0001.pdf|Injuries occurred away from premises after the product was sold/relinquished.|
|3|Limits the insurer's payout for Claim 1 to $1,000,000.|Critical|CGL_Policy_CG0001.pdf|The per-occurrence limit ($1M) is the absolute cap, even if the loss is $1.2M.|
|4|Calculates the insurer's payout for Claim 2 as $790,000.|Major|Claims_Log_Q4.csv|$800,000 settlement - $10,000 deductible = $790,000.|
|5|Limits the insurer's payout for Claim 3 to $1,000,000.|Major|Claims_Log_Q4.csv|$1.5M settlement capped by the $1M per-occurrence limit.|
|6|Calculates the remaining General Aggregate as $1,000,000.|Critical|CGL_Policy_CG0001.pdf|$2,000,000 starting - $1,000,000 (Claim 1) = $1,000,000.|
|7|Calculates the remaining PCOH Aggregate as $210,000.|Critical|CGL_Policy_CG0001.pdf|$2M starting - $790k (Claim 2) - $1M (Claim 3) = $210,000.|
|8|Determines that Claims 2 and 3 are separate occurrences under standard ISO language.|Major|CGL_Policy_CG0001.pdf|Without a "Batch Clause" endorsement, injuries to different persons are usually separate occurrences.|

#### 5. Golden Response

  

MEMORANDUM: Q4 Liability Limit & Aggregate Analysis

  

1. Claim Classification & Peril Analysis

- Claim 1 (Slip & Fall): This is a Premises/Operations loss. Per Section III of the ISO CG 00 01, this claim is subject to the Each Occurrence Limit and reduces the General Aggregate Limit (CGL_Policy_CG0001.pdf, Page 8).
    
- Claims 2 & 3 (Exploding Toasters): Because these injuries occurred away from the insured's premises and after the product was physically relinquished to others, they fall under the Products-Completed Operations Hazard (PCOH) (CGL_Policy_CG0001.pdf, Page 14). These claims reduce the separate PCOH Aggregate Limit (CGL_Policy_CG0001.pdf, Page 8).
    

2. Payout and Limit Application

- Claim 1: The settlement of $1,200,000 exceeds the $1,000,000 Each Occurrence Limit. The insurer pays the $1,000,000 limit. The $10,000 deductible is absorbed by the $200,000 excess loss, so the net payout is exactly $1,000,000 (CGL_Policy_CG0001.pdf, Page 1).
    
- Claim 2: $800,000 settlement - $10,000 deductible = $790,000 payout (Claims_Log_Q4.csv, Row 2).
    
- Claim 3: The settlement of $1,500,000 is capped by the $1,000,000 Each Occurrence Limit. The net payout is $1,000,000 (Claims_Log_Q4.csv, Row 3).
    

3. Final Aggregate Balances

- General Aggregate: $2,000,000 starting - $1,000,000 (Claim 1) = $1,000,000 remaining.
    
- PCOH Aggregate: $2,000,000 starting - $790,000 (Claim 2) - $1,000,000 (Claim 3) = $210,000 remaining.
    

### 

---

Example 33: Business Auto – Symbol 8 (Hired Auto) & Rental Allocation

#### 1. Metadata

- Sub-domain: Claims Handling (Auto)  
      
    
- Difficulty: High  
      
    
- Themes: Hired Auto (Symbol 8), Primary vs. Excess Priority, Contractual Indemnity  
      
    
- Task Type: Workflow  
      
    
- Workflow: Priority of Coverage  
      
    
- Prompt Type: Coverage Position Letter  
      
    

#### 2. Prompt

You are a Commercial Auto Claims Specialist. The current date is October 24, 2026. An employee of "Build-It Corp" (the insured) rented a truck from "Truck-Rent Co" to transport equipment. The employee caused an accident resulting in $400,000 in property damage to a third party.

  

Policy Parameters (simulated Build-It_BAP_CA0001.pdf):

- Liability Limit: $1,000,000 Combined Single Limit (CSL).
    
- Symbols: 1, 2, 8, 9 are selected on the Dec Page.
    
- Condition: Section IV.B.5 (Other Insurance).
    

Rental Context (simulated Rental_Agreement.pdf):

- The agreement states that "Truck-Rent Co" provides the state minimum financial responsibility limit ($30,000) as primary coverage, but the renter's (Build-It Corp) insurance is excess for all amounts above that.
    

Tasks:

- Identify which coverage symbol triggers for the rented truck.
    
- Determine the Order of Operations for the payout between the Rental Co and Build-It Corp's BAP.
    
- Calculate the exact dollar amount Build-It Corp's BAP must pay for the third-party property damage.
    
- Cite the specific "Other Insurance" provision in the CA 00 01 that determines the BAP's status as primary or excess for hired autos.
    

#### 3. Model Analysis

Models often assume that because Build-It Corp is the "Named Insured," their policy is always primary for business use. However, the ISO CA 00 01 form explicitly states that coverage for a vehicle the insured does not own (including hired autos) is excess over any other collectible insurance. A failure to extract this "Other Insurance" nuance leads to the model incorrectly placing the first dollar of loss on the BAP instead of waiting for the Rental Co's primary $30,000 limit to exhaust.

  

#### 4. Rubric

|   |   |   |   |   |
|---|---|---|---|---|
|#|Description|Weight|Sources|Justification|
|1|Identifies the truck as a Symbol 8 (Hired Auto).|Minor|Build-It_BAP_CA0001.pdf|Trucks leased, hired, or borrowed by the insured.|
|2|Identifies the Rental Co's primary limit as $30,000.|Major|Rental_Agreement.pdf|Stated primary limit in the rental contract.|
|3|Concludes Build-It Corp's BAP is Excess.|Critical|Build-It_BAP_CA0001.pdf|Section IV.B.5.a states BAP is excess for non-owned autos.|
|4|Subtracts the $30,000 primary limit from the $400,000 loss.|Major|Calculated|$400,000 total - $30,000 primary = $370,000.|
|5|Calculates the BAP's final payout as $370,000.|Critical|Calculated|The net amount after the primary coverage is exhausted.|
|6|Verifies the $370,000 is within the $1,000,000 CSL.|Minor|Build-It_BAP_CA0001.pdf|$370k < $1M limit.|
|7|Cites "Section IV.B.5.a" or the "Other Insurance" clause.|Major|Build-It_BAP_CA0001.pdf|The specific policy provision mandating excess status.|
|8|Formats the output as a formal Coverage Position Letter.|Minor|Prompt Text|Style requirement.|

#### 5. Golden Response

  

RE: Coverage Position – Third-Party Property Damage Claim

  

1. Coverage Trigger & Symbol Identification

The vehicle involved in the accident, a rented truck, qualifies as a Hired Auto under Symbol 8 of your Business Auto Policy (Build-It_BAP_CA0001.pdf, Page 1). Coverage is afforded for liability resulting from the use of autos you lease, hire, or borrow.

2. Priority of Coverage (Primary vs. Excess)

Per the Other Insurance condition in Section IV.B.5.a of the ISO CA 00 01 form, for any covered auto you do not own, the insurance provided by this policy is excess over any other collectible insurance (Build-It_BAP_CA0001.pdf, Page 14).

Furthermore, the Rental Agreement specifies that "Truck-Rent Co" provides the primary layer of coverage at the state minimum limit of $30,000 (Rental_Agreement.pdf, Page 2).

3. Loss Calculation & Allocation

- Total Property Damage Loss: $400,000
    
- Primary Coverage (Truck-Rent Co): ($30,000)
    
- Net Excess Loss (Build-It Corp BAP): $370,000
    

As the remaining $370,000 is within the policy's $1,000,000 Combined Single Limit, Build-It Corp's BAP will pay the balance in full once the primary carrier's limits are confirmed as exhausted.

  

### 

---

### 

---

#### Works cited

7. Specimen Reinsurance Agreement - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/801019/000119312513170649/d480972dex9926g.htm](https://www.sec.gov/Archives/edgar/data/801019/000119312513170649/d480972dex9926g.htm)
    
8. CP 00 10 10 12 - Building and Personal Property Coverage Form, accessed April 5, 2026, [https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2023/12/CP-00-10-10-12-Building-and-Personal-Property-Coverage-Form.pdf](https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2023/12/CP-00-10-10-12-Building-and-Personal-Property-Coverage-Form.pdf)
    
9. 1.4 Building and Personal Property Coverage Form - Risk & Insurance Education Alliance, accessed April 5, 2026, [https://www.riskeducation.org/learn/pluginfile.php/276804/mod_page/content/1/74%20-%201.4%20Building%20and%20Personal%20Property%20Coverage%20Form.pdf](https://www.riskeducation.org/learn/pluginfile.php/276804/mod_page/content/1/74%20-%201.4%20Building%20and%20Personal%20Property%20Coverage%20Form.pdf)
    
10. CAUSES OF LOSS – SPECIAL FORM - Property Insurance Coverage Law Blog, accessed April 5, 2026, [https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2020/04/Policy-1.pdf](https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2020/04/Policy-1.pdf)
    
11. PRO Form - ABA Insurance Services, accessed April 5, 2026, [https://www.abais.com/docs/default-source/banks/policy-specimen/property-policy-w-dec-coverage-forms-mandatory-endorsements-exclusions.pdf](https://www.abais.com/docs/default-source/banks/policy-specimen/property-policy-w-dec-coverage-forms-mandatory-endorsements-exclusions.pdf)
    
12. Quota Share Reinsurance Agreement - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/8497/000119312511081483/dex1023.htm](https://www.sec.gov/Archives/edgar/data/8497/000119312511081483/dex1023.htm)
    
13. Quota Share Reinsurance Agreement - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/8497/000119312511221520/dex1029.htm](https://www.sec.gov/Archives/edgar/data/8497/000119312511221520/dex1029.htm)
    
14. BUILDING AND PERSONAL PROPERTY COVERAGE FORM, accessed April 5, 2026, [https://www.mofbinsurance.com/Content/PDFs/PolicyForms/CPP/CP00100402.pdf](https://www.mofbinsurance.com/Content/PDFs/PolicyForms/CPP/CP00100402.pdf)
    
15. Liberty Mutual 21-22 Renewal Policy.pdf, accessed April 5, 2026, [https://www.orcashealth.org/files/c2d15f1e2/Liberty+Mutual+21-22+Renewal+Policy.pdf](https://www.orcashealth.org/files/c2d15f1e2/Liberty+Mutual+21-22+Renewal+Policy.pdf)
    
16. Casualty Actuarial Society E-Forum, Winter 2019, accessed April 5, 2026, [https://www.casact.org/sites/default/files/database/forum_19wforum_completewinter2019.pdf](https://www.casact.org/sites/default/files/database/forum_19wforum_completewinter2019.pdf)
    
17. Reinsurance Agreement with Swiss RE - I2107058US19 - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/1039305/000119312519302373/d822999dex99gi.htm](https://www.sec.gov/Archives/edgar/data/1039305/000119312519302373/d822999dex99gi.htm)
    
18. Reinsurance Agreement with RGA Reinsurance - 1582-00-00 - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/1039305/000119312519302373/d822999dex99gvii.htm](https://www.sec.gov/Archives/edgar/data/1039305/000119312519302373/d822999dex99gvii.htm)
    
19. quadrosnatwit/skillsacademy: This repository contains some of the artifacts for the Skills Academy labs - GitHub, accessed April 5, 2026, [https://github.com/quadrosnatwit/skillsacademy](https://github.com/quadrosnatwit/skillsacademy)
    
20. Specimen Reinsurance Agreement - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/801019/000119312513170649/d480972dex9926g.htm](https://www.sec.gov/Archives/edgar/data/801019/000119312513170649/d480972dex9926g.htm)
    
21. CP 00 10 10 12 - Building and Personal Property Coverage Form, accessed April 5, 2026, [https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2023/12/CP-00-10-10-12-Building-and-Personal-Property-Coverage-Form.pdf](https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2023/12/CP-00-10-10-12-Building-and-Personal-Property-Coverage-Form.pdf)
    
22. 1.4 Building and Personal Property Coverage Form - Risk & Insurance Education Alliance, accessed April 5, 2026, [https://www.riskeducation.org/learn/pluginfile.php/276804/mod_page/content/1/74%20-%201.4%20Building%20and%20Personal%20Property%20Coverage%20Form.pdf](https://www.riskeducation.org/learn/pluginfile.php/276804/mod_page/content/1/74%20-%201.4%20Building%20and%20Personal%20Property%20Coverage%20Form.pdf)
    
23. CAUSES OF LOSS – SPECIAL FORM - Property Insurance Coverage Law Blog, accessed April 5, 2026, [https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2020/04/Policy-1.pdf](https://www.propertyinsurancecoveragelaw.com/wp-content/uploads/2020/04/Policy-1.pdf)
    
24. PRO Form - ABA Insurance Services, accessed April 5, 2026, [https://www.abais.com/docs/default-source/banks/policy-specimen/property-policy-w-dec-coverage-forms-mandatory-endorsements-exclusions.pdf](https://www.abais.com/docs/default-source/banks/policy-specimen/property-policy-w-dec-coverage-forms-mandatory-endorsements-exclusions.pdf)
    
25. Quota Share Reinsurance Agreement - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/8497/000119312511081483/dex1023.htm](https://www.sec.gov/Archives/edgar/data/8497/000119312511081483/dex1023.htm)
    
26. Quota Share Reinsurance Agreement - SEC.gov, accessed April 5, 2026, [https://www.sec.gov/Archives/edgar/data/8497/000119312511221520/dex1029.htm](https://www.sec.gov/Archives/edgar/data/8497/000119312511221520/dex1029.htm)
    
27. BUILDING AND PERSONAL PROPERTY COVERAGE FORM, accessed April 5, 2026, [https://www.mofbinsurance.com/Content/PDFs/PolicyForms/CPP/CP00100402.pdf](https://www.mofbinsurance.com/Content/PDFs/PolicyForms/CPP/CP00100402.pdf)
    
28. Liberty Mutual 21-22 Renewal Policy.pdf, accessed April 5, 2026, [https://www.orcashealth.org/files/c2d15f1e2/Liberty+Mutual+21-22+Renewal+Policy.pdf](https://www.orcashealth.org/files/c2d15f1e2/Liberty+Mutual+21-22+Renewal+Policy.pdf)
    
29. Casualty Actuarial Society E-Forum, Winter 2019, accessed April 5, 2026, [https://www.casact.org/sites/default/files/database/forum_19wforum_completewinter2019.pdf](https://www.casact.org/sites/default/files/database/forum_19wforum_completewinter2019.pdf)
    

  
**