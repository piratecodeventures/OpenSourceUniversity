### A Founder, FDE, and AI Engineer's Field Guide to Interrogating Problems and De-Risking Projects
*Adapted from the UK Government Digital Service (GDS) Service Manual and the Founder's Project Discovery Playbook*

---

## 1. What This Guide Is For

Use this guide before committing time, budget, and engineering talent to build a new product, custom feature, workflow automation, or AI/ML system. 

Discovery is for **learning and de-risking, not for writing production code**. The goal is to replace untested guesses with verifiable evidence before spending heavily on development. A project stopped or re-scoped during discovery is a major victory that saves months of wasted labor and thousands (or millions) of dollars.

### Who This Guide Is For

| Role | How This Guide Helps |
| :--- | :--- |
| **Founders & Business Leaders** | Protects capital, validates ROI, avoids premature engineering hires, and ensures tech investments directly fix business bottlenecks. |
| **Product Managers & Project Leads** | Provides a step-by-step interrogation script for stakeholder workshops, aligning teams on scope, baseline metrics, and delivery deadlines. |
| **Forward Deployed Engineers (FDEs)** | Interrogates enterprise client architecture, IT firewalls, legacy APIs, data ownership, and internal politics before design starts. |
| **AI / Machine Learning Engineers** | Audits ground-truth datasets, verifies deterministic baselines, checks hallucination blast radius, and validates token unit economics. |
| **Students & Capstone Teams** | Prevents selecting impossible, ill-defined, or un-evaluatable project scopes by testing assumptions on day one. |

---

## 2. Core Discovery Principles & Concepts Explained Simply

### A. The Doctor's Exam: "Look Before You Leap"
If you visit a doctor and demand knee surgery, a competent physician does not pick up a scalpel. They ask: *"Where does it hurt? When did it start? Let's run an X-ray first."* Discovery is that diagnostic exam. Never accept a client's or executive's pre-packaged technical request (e.g., *"We need an autonomous AI multi-agent swarm"*) without uncovering the root symptom first.

### B. The "Stated Solution" vs. The "Real Problem"
- **Stated Solution:** *"We need an interactive map on our portal showing branch offices."*
- **Real Problem:** *"Customers arrive at closed offices on weekends, get angry, and overwhelm our phone support center."*
- **Why this distinction matters:** The real problem might be solved in 30 minutes with an automated SMS confirmation or an update to Google Maps profiles, with zero custom software required.

### C. Hard Constraints vs. Soft Constraints
- **Hard Constraints ("The Brick Walls"):** Realities that cannot be changed by negotiation. Examples: Data privacy laws (GDPR/HIPAA), hardware limits, air-gapped security, or latency limits of physical networks.
- **Soft Constraints ("The Closed Doors"):** Habits, internal company preferences, or legacy procedures. Examples: *"Marketing reviews everything manually,"* or *"We only export CSVs on Thursdays."* These can be renegotiated if the business case is compelling.

### D. "Shadow IT" and the Hidden Spreadsheet
Official company manuals describe how work is *supposed* to happen. Real work often runs on an unmonitored Excel sheet, private chat group, or personal script maintained by one overworked specialist. Uncovering these hidden workarounds is where the true requirements live.

### E. Riskiest Assumption Testing (RAT)
Instead of building a large Minimum Viable Product (MVP), Discovery isolates the single assumption that could instantly kill the project:
$$\text{Project Risk} = P(\text{Assumption Fails}) \times \text{Impact of Failure}$$
If an AI system requires scanned invoices, but 85% of incoming files are blurry cell-phone snapshots that OCR cannot parse, write a 1-day script to test that assumption on Day 2—not in Month 4.

---

## 3. How to Run Discovery: Timelines & Execution Steps

A standard discovery runs between **2 to 6 weeks**, scaled to project risk and complexity:

```
[ Step 1: Frame the Problem & Establish Baselines ]
  ├── Interrogate the stated solution; define the 1-2 sentence core problem.
  └── Measure the current baseline (time, money, error rate, rework volume).
[ Step 2: Map the Real Work & Talk to Operators ]
  ├── Shadow the people who actually do the work; identify unwritten workarounds.
  └── Identify the final decision-makers, skeptics, and potential blockers.
[ Step 3: Audit Data, Systems, and Operating Limits ]
  ├── Inspect live sample data, verify database ownership, test API rate limits.
  └── Check security, network egress, IAM, and compliance boundaries.
[ Step 4: Run Riskiest Assumption Tests (RATs) ]
  ├── Build disposable spikes or small manual tests with pre-agreed pass/fail criteria.
  └── Test edge cases, fallback procedures, and AI error tolerances.
[ Step 5: Gate Review & Next Steps ]
  └── Make an evidence-backed choice: Move to Pilot/Alpha, Pivot Scope, or Stop.
```

---

## 4. The Master Discovery Question Bank

*Note: You do not need to ask every question. Select the pillars relevant to your project and stakeholders.*

---

### Pillar 1: The Problem, Baseline & Strategic Value
*Purpose: Agree on what is going wrong, who feels the pain, and how success will be measured before discussing solutions.*

#### Defining the Core Bottleneck
- [ ] **One-Sentence Definition:** *"What exact problem are we solving? Can we state it in one or two plain sentences without mentioning any technology or solution?"*
- [ ] **Who Experiences It:** *"Who experiences this problem, how frequently does it occur, and how many employees or customers are affected?"*
- [ ] **Trajectory:** *"Is this problem getting worse, staying the same, or improving over time?"*
- [ ] **Concrete Anchor Example:** *"Can you share one specific, real-world case from the past two weeks that illustrates this problem perfectly?"*
- [ ] **Uneven Pain Distribution:** *"Who feels the impact of this problem most acutely every day, and who in the organization might not notice it at all?"*
- [ ] **The "Why Now?" Probe:** *"Why are we addressing this today instead of six months ago or next year? What changed?"*
- [ ] **Cost of Doing Nothing (CODN):** *"If we choose to do absolutely nothing and walk away today, what breaks or degrades over the next 12 months?"*

#### Value, Scope & Low-Tech Alternatives
- [ ] **The "Magic Wand" & Outcome:** *"If this tool works with 100% perfection, what exact manual task does a human stop doing every morning?"*
- [ ] **The Smallest Useful Result:** *"What is the absolute smallest, simplest outcome we could deliver that would still create measurable relief?"*
- [ ] **Ongoing Value vs. Cost:** *"What would make the outcome valuable enough to justify its full total cost of ownership—including licenses, hosting, monitoring, and ongoing human support?"*
- [ ] **The Explicit Anti-Scope:** *"What is explicitly OUT of scope for this project, even if someone might reasonably assume it is included?"*
- [ ] **Alignment Check:** *"Are different stakeholders describing different problems or expecting conflicting outcomes?"*
- [ ] **Falsification Test:** *"What specific evidence or data would prove that our understanding of this problem is fundamentally wrong?"*
- [ ] **The Graveyard Check:** *"Has anyone inside this organization tried to fix or build this before? What happened, and what did we learn?"*
- [ ] **The Simpler Alternative:** *"Could we solve 80% of this problem with a checklist, a form, an existing SaaS tool, or a process update rather than custom software or AI?"*
- [ ] **The Baseline Metric:**
  $$\text{Current Baseline} = \{\text{Hours/Week spent}\} \times \{\text{Hourly Wage}\} + \{\text{Cost of Rework/Errors}\}$$
  *Do we have a validated numerical baseline today? If not, how will we measure it before writing code?*

---

### Pillar 2: Stakeholders, People, Politics & Adoption
*Purpose: Understand who does the work, who holds veto power, and what human incentives could cause the project to succeed or stall.*

#### Key Actors & Decision Rights
- [ ] **The Frontline Operators:** *"Who performs this task day-to-day, and who receives or depends on the final output?"*
- [ ] **Future Users & Administrators:** *"Who will use the new solution directly, and who will be tasked with administering, configuring, and supporting it?"*
- [ ] **The True Champion:** *"Who is funding this initiative, and whose reputation or bonus is directly tied to its outcome?"*
- [ ] **Long-Term Accountability:** *"Who is accountable for the business outcome after the implementation or project team moves on?"*
- [ ] **Veto & Blocker Authority:** *"Who has the authority to step in at the last minute and block deployment (e.g., Legal, Information Security, Compliance, IT Infrastructure, Union/Worker councils)?"*
- [ ] **Tie-Breaking Mechanism:** *"When two senior leaders disagree on scope, speed, or quality, who makes the binding final decision?"*
- [ ] **Unrepresented Groups:** *"Are there downstream teams or user groups who will be impacted by this change but are not represented in current discussions?"*

#### Organizational Dynamics & Change Resistance
- [ ] **The Skeptic Check:** *"Who in the company thinks this project is a waste of time, money, or effort? What are their specific arguments?"*
- [ ] **Competing Priorities:** *"What other projects or company initiatives are actively competing for the same personnel, attention, or budget?"*
- [ ] **The 1-Year Pre-Mortem:** *"Imagine we fast-forward one year from today. The project has failed or been abandoned. What was the root cause?"*
- [ ] **Adoption Resistance:** *"Even if the new system works as designed, why might frontline staff prefer their old, familiar habits?"*
- [ ] **Trust & Transparency:** *"What would make users trust the new system? What single misstep would destroy their trust?"*
- [ ] **Unintended Downstream Work:** *"Could automating this task inadvertently create extra manual work or verification burden for another team?"*
- [ ] **Feedback & Defect Loops:** *"How will frontline users report errors, friction, or unexpected behavior once the tool is live?"*
- [ ] **Knowledge Silos & Single Points of Failure:** *"Does a single key employee hold critical process knowledge in their head that exists nowhere in writing?"*

---

### Pillar 3: Mapping the Real Work & Operational Workflows
*Purpose: Map every step, decision, handoff, and exception in the workflow before proposing automation.*

#### The Step-by-Step Reality
- [ ] **Live Walkthrough:** *"Can an operator walk us through a real, live task from start to finish while we observe?"*
- [ ] **Trigger & Input Requirements:** *"What event or document initiates the task, and what exact information must be present before work can begin?"*
- [ ] **Volume & Volatility:** *"How many cases or requests are processed daily, weekly, or monthly? How does this spike during peak hours, month-end, or holidays?"*
- [ ] **Completion Definition:** *"Where does data enter the workflow, and what explicit condition marks a case as 'completed'?"*
- [ ] **Friction & Latency Points:** *"Which steps take the longest, are repeated unnecessarily, or cause the most confusion?"*
- [ ] **Blocking Dependencies:** *"Which steps require waiting for another person, department, or external system before continuing?"*
- [ ] **Duplicate Entry:** *"Where do staff re-type, copy, paste, or cross-check information between different screens or systems?"*

#### Decisions, Knowledge & Exception Handling
- [ ] **Tacit Judgment vs. Explicit Rules:** *"At which points in the process do staff rely on personal intuition, memory, or experience rather than a documented rule?"*
- [ ] **Missing Information Handling:** *"What happens when incoming data is incomplete, corrupt, or contradictory? How is it resolved today?"*
- [ ] **Escalation Pathways:** *"Who resolves weird edge cases, and how are unresolved issues escalated?"*
- [ ] **Common vs. Rare Exceptions:** *"Which edge cases occur frequently enough that our solution must support them, and which are rare enough to route to manual review?"*
- [ ] **Unattended Fallback:** *"What happens when an exception occurs and no human specialist is available to resolve it?"*
- [ ] **Approval Gates:** *"Does an output require human review or sign-off before taking effect? Exactly how much verification is required?"*

#### Working Conditions & Environmental Constraints
- [ ] **Physical Context:** *"Where is this tool used (e.g., quiet office, noisy warehouse, moving vehicle, low-bandwidth location)?"*
- [ ] **Hardware & Devices:** *"What devices, operating systems, and screen sizes must be supported (e.g., enterprise desktops, legacy laptops, mobile devices, barcode scanners)?"*
- [ ] **Accessibility & Skills:** *"What digital literacy levels, language proficiencies, or accessibility requirements (e.g., WCAG 2.1 AA) must be met?"*

---

### Pillar 4: Data Lineage, Quality & Systems Reality
*Purpose: Ensure the required source data actually exists, is legally accessible, and is clean enough to support the solution.*

#### System of Record & Access
- [ ] **Trusted Source of Truth:** *"Which specific database, application, or ledger is the authoritative system of record when data fields disagree?"*
- [ ] **Data Ownership & Approvals:** *"Who technically owns each source system, and who has the administrative authority to grant database or API credentials?"*
- [ ] **Sample Data Velocity:** *"How quickly can our team get an anonymized, representative sample of live data? (If this takes $>3\text{ weeks}$, discovery is at risk)."*
- [ ] **Data Extraction Method:** *"How is data extracted today (e.g., real-time webhooks, direct SQL read replicas, batch SFTP, or manual CSV exports)?"*
- [ ] **Throughput Limits:** *"Are there rate limits, API call quotas, or peak-hour locks on source databases?"*

#### Cleanliness, Formats & Hidden Stores
- [ ] **Hygiene & Completeness:** *"What percentage of historical records have missing fields, corrupt schemas, or duplicate entities?"*
- [ ] **Freshness & TTL:** *"How fresh must the data be to be valuable (real-time sub-second, hourly, or overnight batch)? How old can data be before it is stale?"*
- [ ] **Error Correction Flow:** *"When a mistake is discovered in the source system, how is it corrected, and how does that correction propagate downstream?"*
- [ ] **Unrecorded Business Context:** *"Are there crucial business rules, customer tiers, or exceptions that exist only in someone's head and are not recorded in any database?"*
- [ ] **Schema Stability:** *"Are data formats and schemas documented, and how frequently do upstream systems change them without warning?"*
- [ ] **Historical Depth:** *"How many years of historical data exist, and has the format changed over that period?"*
- [ ] **Shadow IT Audit:** *"Are core operations dependent on unmanaged local Excel workbooks, Google Sheets, email threads, or desktop scripts?"*

#### Governance, Security & Compliance
- [ ] **Sensitive Data Types:** *"Does the data contain Personally Identifiable Information (PII), health records (HIPAA), credit card numbers (PCI), or confidential trade secrets?"*
- [ ] **Data Residency & Storage:** *"Are there geographic or regulatory restrictions on where this data can be stored, processed, or transferred?"*
- [ ] **Leakage Prevention:** *"How will we ensure sensitive information is never leaked into application logs, error monitoring, or model prompts?"*
- [ ] **Incident Response:** *"What is the protocol if data is corrupted, exposed, or mishandled during testing or production?"*

---

### Pillar 5: Computing Architecture, Infrastructure & Operating Limits
*Purpose: Understand hosting restrictions, network boundaries, and technical integration constraints.*

#### Environments & Networking
- [ ] **Hosting Environment:** *"Where must the application run (public cloud like AWS/Azure/GCP, client-managed private cloud, or an air-gapped on-premise datacenter)?"*
- [ ] **Network Egress & Firewalls:** *"Can the service connect to the public internet, or are all outbound connections blocked by corporate firewalls?"*
- [ ] **Identity & Access Management (IAM):** *"How do users authenticate (SAML SSO, Okta, Azure AD, OAuth2)? What Role-Based Access Control (RBAC) tiers are required?"*
- [ ] **Mandated Platforms & Vendors:** *"Are there mandated enterprise frameworks, vendors, or tools that must be used (or avoided)?"*
- [ ] **Legacy Interoperability:** *"Which legacy systems must we integrate with, and do they have modern REST/GraphQL APIs or legacy SOAP/RPC/file-drop protocols?"*
- [ ] **Degraded State Behavior:** *"What should happen to the application when an upstream dependency is slow, unresponsive, or down?"*

#### Performance, Scale & Operations
- [ ] **Concurrency:** *"How many concurrent users or transactions must the system support during normal operations versus peak surges?"*
- [ ] **Latency Budget:** *"What is the acceptable latency budget for end-to-end processing (e.g., $<500\text{ ms}$ for web requests, overnight batch for processing)?"*
- [ ] **Availability & RTO/RPO:**
  - What is the required uptime SLA (e.g., $99.9\%$ vs. $99.99\%$)?
  - What is the Recovery Time Objective (RTO) and Recovery Point Objective (RPO) if a server fails?
- [ ] **Observability & Network Isolation:** *"What operational telemetry (logs, metrics, traces) must be gathered, and can that telemetry leave the private network?"*
- [ ] **Post-Launch Maintenance:** *"Which specific team is responsible for monitoring, patching, and supporting the application after launch?"*

---

### Pillar 6: AI, Machine Learning & Automation Reality Checks
*Purpose: Verify whether AI is genuinely needed, evaluate risk and safety tolerances, and check unit economics.*

#### Necessity & Baseline Probes
- [ ] **The "Do You Actually Need AI?" Test:** *"Can this problem be solved effectively with deterministic rules, regular expressions, SQL queries, or standard search?"*
- [ ] **Output & Consumer:** *"What exact artifact does the AI generate (e.g., text, summary, classification, vector, code), and who acts on that output?"*
- [ ] **The "Golden Dataset":** *"Do we have at least 100 to 500 validated, clean examples of input data paired with the ground-truth ideal output?"*
- [ ] **Evaluation Agreement:** *"If two experienced human specialists look at the same input, do they agree on the correct answer? (If humans disagree, an AI cannot be evaluated)."*

#### Failure Modes, Risk & Guardrails
- [ ] **Type I vs. Type II Errors:** *"Which is more damaging to the business: a False Positive (spurious alert) or a False Negative (missed defect)?"*
- [ ] **Hallucination Blast Radius:** *"What is the worst-case consequence if the model generates a fabricated, misleading, or incorrect answer?"*
- [ ] **Confident Hallucinations:** *"Could a confident-sounding but incorrect answer deceive a user, and how will users verify accuracy?"*
- [ ] **Autonomous Boundaries:** *"What critical decisions must the system NEVER be permitted to make autonomously?"*
- [ ] **Human-in-the-Loop (HITL):** *"Which outputs must always require human review before execution, and which can be automated?"*
- [ ] **Uncertainty Fallback:** *"What should the system do when its confidence score falls below the required threshold?"*
- [ ] **Quantitative Acceptance Thresholds:**
  $$\text{Accuracy Target} \ge X\%, \quad \text{Precision} \ge Y\%, \quad \text{Recall} \ge Z\%$$
  *What are the explicit numerical targets required before this model is allowed near users?*
- [ ] **Drift & Regression Monitoring:** *"How will we monitor accuracy, drift, and edge-case degradation after deployment?"*

#### Latency, Data Privacy & Unit Economics
- [ ] **Latency Ceiling:** *"What is the maximum latency tolerated by the user experience (e.g., Time-to-First-Token $\le 300\text{ ms}$, total response $\le 3\text{ seconds}$)?"*
- [ ] **Unit Economics Formula:**
  $$\text{Net Value per Transaction} = \text{Human Labor Saved} - (\text{Inference Cost} + \text{Infrastructure Overhead} + \text{HITL Review Cost})$$
  *Is the net value per transaction comfortably positive at scale?*
- [ ] **Data Retention & Model Training Policies:** *"Do vendor terms allow them to use customer data to train their models? Are self-hosted open-weight models (e.g., Llama, Mistral) required for compliance?"*

---

### Pillar 7: Timing, Governance, Approvals & Procurement
*Purpose: Uncover structural, legal, and bureaucratic blockers that could stall delivery.*

#### Timelines & Critical Path
- [ ] **Fixed vs. Arbitrary Deadlines:** *"Is the target launch date anchored to an unmovable event (e.g., a new law, product launch, fiscal year-end, external audit), or is it an internal target?"*
- [ ] **Schedule Slip Impact:** *"What is the concrete financial or operational consequence if delivery is delayed by 90 days?"*
- [ ] **Review Cadence:** *"How frequently do review boards (Security, Architecture, Legal, Data Privacy) convene, and how many weeks does an approval cycle take?"*
- [ ] **Approval Artifact Requirements:** *"What exact documentation, diagrams, or security questionnaires must be submitted before approvers will review the project?"*
- [ ] **Standard Compliance:** *"What accessibility, industry-specific, safety, or trade union regulations must be met?"*

#### Procurement & Onboarding Velocity
- [ ] **Vendor & Tier Limitations:** *"Do existing vendor agreements restrict which cloud services or tools we can deploy? Will this require a new enterprise contract or tier upgrade?"*
- [ ] **Team Credential Velocity:** *"From day one, how many business days does it take for a new engineer to receive authorized laptops, VPN access, and staging database credentials?"*
- [ ] **Feedback Frequency:** *"Can key stakeholders and users commit to a 30-minute bi-weekly demo to review findings and working prototypes?"*

---

### Pillar 8: Riskiest Assumption Testing (RAT)
*Purpose: Use low-cost, disposable tests to validate or invalidate the assumptions most likely to kill the project.*

- [ ] **Assumption Inventory:** *"What core assumptions must prove true for this project to succeed commercially and technically?"*
- [ ] **Fatal Flaw Identification:** *"Which single assumption, if proven false, makes the project impossible, unsafe, or economically unviable?"*
- [ ] **Smallest Disposable Test:** *"What is the fastest, lowest-cost test (e.g., a 20-line Python script, a manual Wizard-of-Oz test, a 20-call sample audit) to evaluate this assumption?"*
- [ ] **Production Separation:** *"Are we keeping our test strictly disposable, avoiding building production architecture during discovery?"*
- [ ] **Pre-Agreed Pass/Fail Criteria:** *"What explicit, measurable result constitutes a 'Pass' vs. a 'Fail' before running the test?"*
- [ ] **Real-World Test Conditions:** *"Did the test use authentic, messy data and reflect real operational conditions rather than synthetic or sanitized inputs?"*
- [ ] **Pivot Trigger:** *"If the test fails or produces inconclusive results, what is our immediate fallback or pivot plan?"*
- [ ] **Sign-Off Authority:** *"Who evaluates the test data to decide whether the assumption is cleared?"*

---

## 5. The Discovery Gate Review: Making the Decision

At the end of discovery, hold a formal review with stakeholders, budget owners, and engineers. Discovery ends with an evidence-backed decision among three paths:

```
                           ┌───────────────────────────┐
                           │   Discovery Gate Review   │
                           └─────────────┬─────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌─────────────────┐            ┌───────────────────┐            ┌──────────────────┐
│ 1. MOVE TO PILOT│            │ 2. CHANGE THE PLAN│            │  3. STOP PROJECT │
│   (ALPHA PHASE) │            │   (PIVOT SCOPE)   │            │   (CANCEL/CLOSE) │
└────────┬────────┘            └─────────┬─────────┘            └────────┬─────────┘
         │                               │                               │
• Problem is real & costly.     • Problem is real, but initial  • Problem isn't severe.
• Users want a solution.          solution is wrong.            • Costs exceed value.
• Data & access confirmed.      • Blocker is broken data, lack  • Blocked by hard rules.
• Unit economics positive.        of access, or bad process.    • Simpler tool works.
• Key risks tested & passed.    • Focus on the foundation first.• Team saved resources.
```

### The Three Gate Decisions

| Decision | When to Choose This | What You Do Next |
| :--- | :--- | :--- |
| **1. Move to Pilot (Alpha)** | The problem is validated, the value exceeds the cost, users are engaged, data/access are confirmed, and the riskiest assumptions passed. | Launch a time-boxed 6–12 week Alpha phase. Build working, testable prototypes with a small user cohort under explicit success criteria. |
| **2. Change the Plan (Pivot)** | The problem matters, but the requested solution is not the right starting point. The real bottleneck is dirty data, a broken process, or missing access. | Re-scope the initiative. Solve the foundational blocker first (e.g., data pipeline consolidation, workflow redesign) before re-evaluating software/AI automation. |
| **3. Stop the Project (Cancel)** | The problem does not justify the total cost, data is unviable, technical/legal constraints block success, or a low-tech alternative is sufficient. | Document findings, archive test scripts, celebrate saving the company from a costly mistake, and redeploy resources to high-value opportunities. |

---

## 6. What "Discovery Done" Looks Like (Deliverables Checklist)

Before closing discovery, ensure these 5 concrete deliverables are completed:

- [ ] **1. Executive Summary (1–2 Pages):** Plain-language explanation of the real problem, who experiences it, the business baseline, and the final gate recommendation.
- [ ] **2. Current vs. Future Journey Map:** Visual diagram detailing the current manual process (including shadow spreadsheets, friction points, and handoffs) alongside the proposed workflow.
- [ ] **3. Data & Systems Readiness Matrix:** A 1-page scorecard grading existing data sources, schemas, API availability, and compliance clearances (Green / Yellow / Red).
- [ ] **4. Riskiest Assumption Test (RAT) Log:** A record of the specific tests conducted, the sample data used, the quantitative results, and the conclusions reached.
- [ ] **5. Pilot / Alpha Roadmap or Close-Out Note:**
  - *If moving to Pilot:* A prioritized backlog of user stories, hypotheses to test, required team members, and a timeline.
  - *If stopping or pivoting:* A concise summary of why the project stopped, key lessons learned, and recommended next steps.

---

## 7. Reusable Project Discovery Checklist & Sign-Off Template
*(Copy and paste this section directly into your project workspace, Notion, Jira, or shared document to run a project discovery)*

```markdown
# Project Discovery Workbook & Sign-Off

### Project Information
- **Project Name:** 
- **Date & Discovery Sprint:** 
- **Discovery Lead / FDE:** 
- **Executive Sponsor / Decision Owner:** 
- **Primary Users & Downstream Stakeholders:** 

---

### Step 1: The Problem & Value Validation
*Mark each item as [Yes], [No], or [Not Sure]. Any "Not Sure" requires investigation.*

- [ ] We can explain the core problem in 1–2 plain sentences without mentioning software or AI.
- [ ] We know who experiences the problem and how frequently it happens.
- [ ] We have inspected a real, recent example illustrating the issue.
- [ ] We have recorded a measurable baseline of the current process (time, cost, error rate, rework volume).
- [ ] We have quantified the cost of doing nothing for the next 12 months.
- [ ] We have defined what success looks like and how it will be measured.
- [ ] We verified whether a simpler, low-tech solution (process update, form, checklist) could solve this.

**Problem Statement:** 
> [Insert 1-2 sentence statement here]

**Current Baseline Metrics:**
- Manual time per task / week:
- Direct financial or rework cost:
- Current error or failure rate:

**Expected Improvement / Success Target:**
- Target metric improvement:

---

### Step 2: People, Workflows & Adoption
- [ ] We shadowed frontline workers performing the task live.
- [ ] We spoke with the people who depend on or receive the final output.
- [ ] We identified who funds the project, who makes final decisions, and who can veto it.
- [ ] We mapped the step-by-step workflow, including common edge cases and exceptions.
- [ ] We identified undocumented workarounds and shadow spreadsheets.
- [ ] We assessed why users might resist adopting the new system and what builds trust.
- [ ] We checked whether this project will create extra manual work for downstream teams.

**Primary Users & Impacted Groups:**
- Daily Operators:
- System Approver / Owner:
- Key Skeptic / Opposing Voice:

**Main Pain Points & Shadow Workarounds Discovered:**
- [List findings here]

---

### Step 3: Data, Systems & Operating Feasibility
- [ ] Authoritative systems of record have been identified.
- [ ] We verified that required data exists, is accessible, and has sufficient history.
- [ ] We secured an anonymized sample of authentic data for testing.
- [ ] Privacy, regulatory, and security requirements (PII, HIPAA, GDPR) are documented.
- [ ] Deployment environments (cloud, on-prem, air-gapped) and network egress constraints are confirmed.
- [ ] Authentication, user roles, and legacy API limitations are identified.
- [ ] (If AI/ML) We checked whether deterministic rules could solve this, evaluated hallucination risks, verified token economics, and confirmed golden test datasets.

**Key Technical Blockers & Governance Constraints:**
- [List findings here]

---

### Step 4: Riskiest Assumption Testing (RAT)
- [ ] We listed the assumptions critical to project viability.
- [ ] We identified the fatal assumption that could kill the project.
- [ ] We completed a small, disposable test with pre-agreed pass/fail criteria.
- [ ] The test was conducted using authentic, messy data under realistic conditions.

**Top Unknown / Riskiest Assumption Tested:**
- Assumption: 
- Test Method: 
- Success Threshold: 
- Actual Result: [Pass / Fail / Inconclusive]

---

### Step 5: Final Decision & Sign-Off

#### Gate Decision:
- [ ] **MOVE TO PILOT (ALPHA):** Proceed to small-scale pilot with defined user cohort.
- [ ] **CHANGE THE PLAN (PIVOT):** Address foundational data/process blocker before continuing.
- [ ] **STOP THE PROJECT (CANCEL):** Discontinue project; redeploy resources.

**Reason for Decision:**
> [Summarize the primary evidence supporting this decision]

#### Next Actions & Ownership:
| Action Item | Owner | Target Completion Date |
| :--- | :--- | :--- |
| 1. | | |
| 2. | | |
| 3. | | |

#### Sign-Off:
- **Decision Owner Name:** 
- **Signature / Approval:** 
- **Date:** 
- **Next Progress Review Date:** 
```
