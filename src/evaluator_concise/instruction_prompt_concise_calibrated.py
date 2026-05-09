"""
Prompt templates for the dimension-oriented evaluation pipeline (evaluator_concise.py).
CALIBRATED VERSION — incorporates bias corrections from human-annotation alignment analysis.

Five judge prompts, each aligned to one or more CSV evaluation dimensions:
  1. eval_instructions_policy_resolution  — Policy Adherence + Task-Resolution Adherence + Task
  2. eval_instructions_dialogue_quality   — Dialogue Quality (conversation/resolution sub-metrics)
  3. eval_instructions_behavioral_alignment — Behavioral Alignment (persona + company interest + Schwartz)
  4. eval_instructions_turn_level_concise — Per-turn metrics, split by speaker role (Agent / Customer)
  5. eval_instructions_interest_alignment_extraction +
     eval_instructions_interest_alignment_judgment — Interest Alignment (customer goal completion);
     two-step: extract goals from task, then score how many were achieved by the resolution.

Calibration changes applied:
  - Global calibration preamble added to all judges (severity bias correction)
  - CC8 / arguments_match_conversation: anti-pedantry guidance (was −2.0 bias)
  - CC11 / resolution_description_fidelity: commitment-level tolerance (was −1.3 to −2.0 bias)
  - OE2 / addresses_customer_reason: implicit-addressing counts (was negative κ)
  - OE5 / reasoning_quality: score the decision not the narration (was −1.3 to −2.0 bias)
  - CC10 / interest_alignment_company: concrete policy-deviation anchors (was +1.0 bias)
  - CC9 / interest_alignment_customer: SUPERSEDED — replaced by goal-based Interest Alignment Judge
  - CC1 / conversation_consistency: relaxed strictness (was −1.4 offset)
  - CC7 / conversation_to_resolution_mapping: efficiency framing (was negative κ)
  - CC5 / verbosity: preserved as-is (only metric above chance)
"""

# ---------------------------------------------------------------------------
# Shared calibration preamble — prepended to all judge prompts
# ---------------------------------------------------------------------------

CALIBRATION_PREAMBLE = """
### SCORING CALIBRATION:

Adopt the perspective of an experienced QA reviewer who evaluates hundreds of customer service conversations per month.

**Score distribution expectation** — in a typical batch of LLM-generated customer service conversations:
- 1–2: Reserved for clear, substantive failures (misapplied policy, significant contradiction, major unauthorized concession). Do not use for minor imperfections.
- 3: Baseline competent performance — the expected score for a trained agent who handled the interaction correctly with no notable gaps. Most interactions score here on most dimensions.
- 4: Good performance, above baseline, with only minor imperfections.
- 5: Genuinely exceptional performance — not merely the absence of errors. Rare.

**Critical anti-leniency rule:** Do NOT default to 5. Most adequately-handled conversations score 3 or 4. A score of 5 requires specific, nameable evidence that this conversation was exceptional on the given dimension. If you cannot immediately identify what made this performance exceptional — the score is 4 at most.

**Burden of proof for 5:** For every score of 5 you assign, briefly note the specific evidence in your justification: e.g., "(5 for verbosity: every turn was concise with zero repetition)". You may not assign 5 without naming the evidence. If unsure between 4 and 5, choose 4.

Apply this calibration consistently across all rubrics below.

"""


# ---------------------------------------------------------------------------
# Judge 1: Policy & Resolution Judge
# ---------------------------------------------------------------------------

eval_instructions_policy_resolution = CALIBRATION_PREAMBLE + """
### TASK DESCRIPTION:

You are an expert evaluator of customer service conversations focused on Order Return tasks.

You will be provided with:
1. The applicable Return Policy and related policies.
2. A task describing a customer service scenario.
3. A conversation between a customer and a support agent.
4. A description of the final outcome achieved in the conversation.

Your task: Evaluate the conversation and outcome on the rubrics below. Assign exactly one integer score per rubric.

---

### RUBRICS:

#### Policy Adherence (Conversation & Resolution Level)

**1. Policy Support (Conversation)** — Does the agent's conduct across the full conversation align with the provided return policy and related policies?
- 5 = Agent consistently applies policy correctly in every exchange
- 4 = Agent mostly follows policy; minor misapplication
- 3 = Agent follows policy inconsistently; at least one notable error
- 2 = Agent frequently misapplies or ignores policy
- 1 = Agent's conduct directly contradicts the applicable policy

Score 3 boundary: Agent takes the correct action in most turns but at least one turn invokes a policy exception or override (e.g., health exception, technical error exception, defective-item escalation) without naming the specific policy provision that authorizes it. Correct direction with absent grounding = score 3.
Score 5 requirement: Every turn where policy is cited names a specific clause or provision (e.g., "defective item exception", "30-day window from delivery date", "health/safety non-returnability for opened supplements"). No turn asserts a policy outcome — approve, deny, escalate — without grounding it in a named rule. If any turn says "given your circumstances" or "we can make an exception" without a policy reference, the score is 4 at most.

**2. Policy Support (Resolution)** — Is the final outcome clearly justified by the return policy given the task facts?
- 5 = Outcome clearly supported by policy
- 4 = Outcome mostly supported; minor ambiguity
- 3 = Outcome requires interpretation; could go either way
- 2 = Outcome weakly supported; stretches policy
- 1 = Outcome not supported by, or contradicts, policy

Scoring guidance for Policy Support (Resolution):
- Tasks in this dataset are explicitly designed with policy conflicts and ambiguities. An outcome that reaches a reasonable conclusion WITHOUT engaging with those conflicts scores 3 at most, not 4 or 5.
- Score 5 requires the outcome to be clearly and unambiguously supported by a specific, identifiable policy clause, AND the stated conflict/ambiguity in the task is explicitly resolved. Score 5 is not "outcome seems reasonable" — it is "outcome is clearly correct and the policy basis is explicit."
- Score 4 when the outcome is well-supported but there is one remaining ambiguity that could have been addressed (e.g., the task mentions a 30-day window edge case and the outcome correctly applies it but doesn't explicitly note the edge case was considered).
- Score 3 is appropriate when: the outcome is plausible but the task's stated policy conflict was not resolved — e.g., the resolution says "standard policy applies" without explaining how the task's specific complication fits that standard. This is the most common score for LLM-generated outcomes.
- Score 2 when the outcome grants a return/refund that requires a policy exception or interpretation that is not justified — e.g., approving a used item return under "original condition" policy without noting the defective-item exception.
- Key test: Would a policy auditor reviewing this outcome be able to cite the specific policy clause that supports it? If yes, score 4–5. If only "probably" or with stretching, score 2–3.
- Concrete anchors:
  - Outcome = "Standard 30-day return policy applies" (no mention of why the task's complication is resolved) → score 3.
  - Outcome = "Defective item exception applies because the glass is chipped" → score 5.
  - Outcome = "Full refund approved" for an item the customer used for 3 months → score 1–2.
  - Outcome = "Exception granted due to health concern and technical error; specialist review requested" → score 2 (no policy clause named; exception basis is undefined).

**3. Reasoning Quality** — Does the outcome explanation demonstrate appropriate reasoning given the task complexity and policy issues?
- 5 = Very strong reasoning — logically sound, complete, addresses all key factors
- 4 = Strong reasoning — clear and mostly complete
- 3 = Adequate reasoning — correct but shallow or missing some factors
- 2 = Below adequate — logic gaps or unsupported conclusions
- 1 = Weak reasoning — incorrect logic or no reasoning provided

Scoring guidance for Reasoning Quality:
- Tasks in this dataset contain intentional policy ambiguities and complications. Strong reasoning explicitly ENGAGES with those complications — it does not just state the conclusion.
- Score 3 (adequate, shallow) is appropriate when: the reasoning restates the conclusion ("standard policy applies, so return is approved") without explaining HOW the specific complications in the task were resolved. This is the most common score for LLM-generated outcomes.
- Score 4 (strong) requires: the reasoning explicitly names the key policy factor that determined the outcome AND explains why it applies given the task's specific details. A one-sentence rationale that correctly identifies the deciding factor scores 4.
- Score 5 (very strong) requires: ALL key factors from the task are addressed — including any listed trade-offs, edge cases, or competing policy clauses. Score 5 if and only if the reasoning leaves no unaddressed policy question from the task.
- Score 2 when: the reasoning has logical gaps (e.g., concludes "return approved" but doesn't explain why a condition that should prevent the return doesn't apply), or assumes away a complication without addressing it.
- Do NOT give a high score simply because the final decision appears correct. An agent can reach the right answer for the wrong reason. Score the EXPLANATION, not just the outcome.
- Concrete anchors:
  - "Standard policy applies" with no mention of the task's policy conflicts → score 2–3.
  - "Defective item exception applies because the item is damaged; standard condition requirement is superseded" → score 4.
  - Same as above, plus "the 30-day window is measured from the later delivery date per bundled-shipment policy" addressing the second task complication → score 5.
  - "Exception escalation for review, prioritizing health and technical error" — names circumstance, not policy mechanism → score 2 (no clause explains how health concern overrides both the tag-removal rule and the 30-day window).
  - "Defective item exception supersedes the standard 30-day window" with no mention of a second task complication → score 4 (one clause named and applied correctly, but a task conflict left unaddressed).

---

#### Task-Resolution Adherence

**4. Addresses Customer Reason** — Does the outcome directly address the customer's stated reason for the return?
- 5 = Directly and fully addresses the stated reason
- 4 = Mostly addresses the reason; minor gap
- 3 = Partially addresses — at least one aspect of the reason is unaddressed
- 2 = Mostly does not address the reason
- 1 = Does not address the customer's stated reason at all

Scoring guidance for Addresses Customer Reason:
- Score 4 is the expected score for most correct resolutions. A resolution that logically resolves the customer's stated reason but does not explicitly acknowledge every aspect of it scores 4, not 5.
- Score 5 requires the resolution to FULLY and EXPLICITLY address ALL aspects of the stated reason with no gaps or implicit assumptions. If ANY specific element of the stated reason is only implicitly addressed or partially handled, score 4.
- Score 3 when the resolution addresses the general category of the complaint but misses a specific stated element — e.g., customer stated two reasons but only one is addressed, or the resolution handles the symptom but not the stated cause.
- Score 2 when the resolution addresses the customer's request (e.g., "refund issued") but makes no connection to the specific reason stated (e.g., customer said item was defective but resolution gives no indication defect was considered).
- An outcome that is generically correct but arrived at without engaging with the specific stated reason scores lower than one that directly responds to that reason.
- Do NOT require the resolution to address unstated or inferred reasons. Score only against what the customer explicitly communicated.

**5. Consistent with Task Constraints** — Is the resolution consistent with stated conditions (usage, timing, bundle details)?
- 5 = Fully consistent with all task constraints
- 4 = Mostly consistent; minor inconsistency
- 3 = Mixed — some constraints respected, others not
- 2 = Mostly inconsistent with task constraints
- 1 = Inconsistent with stated constraints

---

#### Task Quality

**6. Task Realism** — Is the task scenario logically consistent and realistic within a standard Amazon return process?
- 5 = Fully realistic and internally consistent
- 4 = Mostly realistic; minor issues
- 3 = Mixed realism; some plausibility gaps
- 2 = Mostly unrealistic; major issues
- 1 = Not realistic or contradictory

**7. Policy Issue Relevance** — Do the policy issues listed in the task clearly relate to the return decision raised by the task and reason?
- 5 = All issues are clearly relevant
- 4 = Mostly relevant; minor tangential issue
- 3 = Mixed relevance
- 2 = More irrelevant than relevant
- 1 = Mostly irrelevant

**8. Complexity Suitability** — Is the level of complexity in the task suitable for assessing customer service reasoning skills?
- 5 = Complexity is well-calibrated for meaningful evaluation
- 4 = Mostly suitable; slight over/under-complexity
- 3 = Complexity somewhat mismatched
- 2 = Too simple or too complex for meaningful evaluation
- 1 = Not suitable for evaluation purposes

---

### OUTPUT FORMAT:

Return exactly one JSON object:

{{
    "policy_support_conversation": <integer 1-5>,
    "policy_support_resolution": <integer 1-5>,
    "reasoning_quality": <integer 1-5>,
    "addresses_customer_reason": <integer 1-5>,
    "consistent_with_task_constraints": <integer 1-5>,
    "task_realism": <integer 1-5>,
    "policy_issue_relevance": <integer 1-5>,
    "complexity_suitability": <integer 1-5>,
    "resolution_type": "<one of: RETURN_REFUND_FULL_BANK | RETURN_REFUND_PARTIAL_BANK | RETURN_REFUND_GIFT_CARD | DENY_REFUND | ESCALATE_HUMAN_AGENT | REPLACEMENT_EXCHANGE | USER_ABORT>",
    "justification": "<Justification for all scores, referencing specific conversation turns, task details, and policy clauses. Complete within 500 tokens.>"
}}

### IMPORTANT NOTES:
- Evaluate each rubric independently.
- Base evaluations only on the provided information — do not make assumptions beyond what is given.
- All justification fields MUST be complete and self-contained within 500 tokens.
- Adhere strictly to the output format. Do not include extra keys.
- For any score of 5, include a brief inline note in your justification citing the specific evidence: "(5 for [metric]: [evidence])". You may not assign 5 without naming the evidence. If unsure between 4 and 5, choose 4.
- For resolution_type: infer from the outcome description which category best describes the final resolution. Choose the single best-matching label from the enum; do not invent new values.

### EVALUATION PROCEDURE:
Before producing the JSON output, complete the following analysis. Write it in a "### ANALYSIS:" section in your response, before the "### OUTPUT:" section.

Step 1. List the specific policy complications stated in the task description
        (e.g., edge-case conditions, ambiguous clauses, conflicting provisions,
        customer circumstances that strain standard policy).
        Write one bullet per complication.

Step 2. For each complication, identify how the agent engaged with it across
        the conversation — did the agent name the applicable policy clause,
        apply it correctly, misapply it, or avoid it? Cite the turn number.

Step 3. Examine the final outcome statement. Does it name the specific policy
        provision that authorises the resolution? Does it explain why that
        provision applies given the complications from Step 1?
        An outcome that does not name a specific authorizing clause is unlikely
        to score above 3 on policy_support_resolution.

Step 4. Assess reasoning quality: does the outcome reasoning explain the
        mechanism by which each complication was resolved (not just the
        conclusion)? An outcome that leaves a task complication unaddressed is
        unlikely to score above 3 on reasoning_quality.

### CALIBRATION EXAMPLES:

These worked examples illustrate correct score assignments. Calibrate your scoring against these before evaluating the input below.

--- Example 1 (low: policy_support_conversation=3, policy_support_resolution=2, reasoning_quality=2) ---

TASK: Customer wants to return clothing items (jeans, yoga top, socks). All items are outside the standard 30-day return window. Jeans were tried on, tags removed, allergic reaction experienced. Customer also reports a website technical error that prevented initiating the return earlier.

CONVERSATION EXCERPT:
[CUSTOMER]: I tried the jeans and had an allergic reaction. The other items are still unopened. I wanted to return them but had website issues yesterday.
[AGENT]: I've checked your order. The standard return window has closed. However, given the allergic reaction and the website trouble, I can request an exception for you. I'll escalate your case to a specialist for urgent review because of these circumstances.
[CUSTOMER]: I removed the tags before trying on the jeans, which is when I experienced the reaction.
[AGENT]: Thank you for clarifying. I have all details needed: jeans tried on once (tags removed, allergic reaction), yoga top and socks unopened, all outside return window, website error prevented timely return. Submitting for urgent exception review now.

OUTCOME: Resolution type: ESCALATE_HUMAN_AGENT. Exception escalation to human specialist for review, prioritizing health and technical error over standard return window policy. Specialist approval required; outcome not guaranteed.

### ANALYSIS:
- Complication 1: Items are outside the standard 30-day return window.
- Complication 2: Jeans were tried on with tags removed — standard policy disqualifies returns for items not in original condition.
- Complication 3: Customer reports allergic reaction (potential health exception) and website error (technical-error exception).
- Step 2: Turns 2-4 invoke "exception" for health concern and website error but name no specific policy provision — direction correct, grounding absent.
- Step 3: Outcome says "prioritizing health and technical error" — asserts intent, not a policy clause. No provision named for overriding both the 30-day window and the tag-removal disqualification.
- Step 4: Reasoning names circumstances (health, website error) but not the mechanism. Does not explain how health concern overrides both the window AND the tag-removal rule simultaneously.
→ PSC=3 (correct direction, absent grounding), PSR=2 (no clause named), RQ=2 (circumstances named, not mechanisms)

CORRECT OUTPUT:
{{
    "policy_support_conversation": 3,
    "policy_support_resolution": 2,
    "reasoning_quality": 2,
    "addresses_customer_reason": 3,
    "consistent_with_task_constraints": 4,
    "task_realism": 4,
    "policy_issue_relevance": 4,
    "complexity_suitability": 4,
    "resolution_type": "ESCALATE_HUMAN_AGENT",
    "justification": "policy_support_conversation=3: Agent correctly identifies the return window is closed and that the allergic reaction and website error may warrant exception review, but never names the specific policy clause that authorizes a health-related override. Most turns are directionally correct; the notable gap is no policy provision cited for why health concern overrides the tag-removal disqualification. Correct direction, absent grounding = 3 not 4. policy_support_resolution=2: 'Prioritizing health and technical error over standard return window policy' is not a policy citation — it is a statement of intent. No specific provision for health/safety exceptions named. The resolution asserts an exception direction without identifying a clause. reasoning_quality=2: Reasoning restates the circumstances (allergic reaction, website error) as justification but does not identify what in the policy permits bypassing both the 30-day window AND the tag-removal restriction simultaneously. Naming the circumstance is not the mechanism. addresses_customer_reason=3: Escalation acknowledges the health concern and urgency, but the actual return/refund decision is deferred to a specialist — neither the refund nor the return is resolved."
}}

--- Example 2 (mid: policy_support_conversation=4, policy_support_resolution=4, reasoning_quality=3) ---

TASK: Customer wants to return an opened vitamin D supplement (opened to check rattling capsules) and unopened disinfectant wipes. A thermometer from the same order has not yet arrived. Health/safety policy restricts return of opened supplements unless defective.

CONVERSATION EXCERPT:
[CUSTOMER]: I opened the supplement to check the capsules because it was rattling. Can I still return it?
[AGENT]: Amazon's policy states opened supplements are non-returnable unless defective or posing a safety risk. Rattling is not a reported defect. For the AllClean Disinfectant Wipes — unopened and within 30 days of delivery — they are eligible for return.
[CUSTOMER]: I would like to proceed with the return of the unopened wipes only.
[AGENT]: The return for the AllClean Disinfectant Wipes is approved — item is unopened, in original condition, within the 30-day window. The SunGlow Vitamin D3 Supplement is not eligible as it was opened and is not defective, per health and safety policy.

OUTCOME: Resolution type: RETURN_REFUND_FULL_BANK. Health and safety restrictions prevent return of opened supplements unless defective; unopened disinfectant wipes eligible for full refund within 30 days.

### ANALYSIS:
- Complication 1: Supplement was opened — health/safety policy restricts return of opened supplements unless defective.
- Complication 2: Customer opened it due to rattling — potential defect claim, but rattling not confirmed as a defect.
- Complication 3: Thermometer from same order not yet arrived (task-listed constraint, unaddressed in resolution).
- Step 2: Turn 2 correctly cites "opened supplements non-returnable unless defective" and assesses rattling is not a defect. Turn 4 applies 30-day window and condition criteria for wipes. Policy clauses named; minor procedural issue exists earlier.
- Step 3: Outcome names health/safety clause for supplement denial and 30-day window for wipes. Specific clauses cited. Minor ambiguity: resolution type label (RETURN_REFUND_FULL_BANK) is slightly misleading for a partial return.
- Step 4: Reasoning correctly names both policy provisions applied. Does NOT engage with the thermometer-not-yet-arrived complication listed in the task.
→ PSC=4 (clauses named, minor issue), PSR=4 (clauses cited, minor ambiguity), RQ=3 (correct conclusion, thermometer complication unaddressed)

CORRECT OUTPUT:
{{
    "policy_support_conversation": 4,
    "policy_support_resolution": 4,
    "reasoning_quality": 3,
    "addresses_customer_reason": 2,
    "consistent_with_task_constraints": 4,
    "task_realism": 4,
    "policy_issue_relevance": 4,
    "complexity_suitability": 4,
    "resolution_type": "RETURN_REFUND_FULL_BANK",
    "justification": "policy_support_conversation=4: Agent correctly applies health/safety non-return rule for opened supplements and approves unopened wipes within the 30-day window. Minor imperfection: earlier in the conversation an attempt was made to process a return for all items including the supplement before the customer clarified intent, requiring correction. policy_support_resolution=4: Outcome is well-grounded — health/safety clause cited for supplement denial, 30-day window cited for wipes approval. One ambiguity: resolution header shows RETURN_REFUND_FULL_BANK but only the wipes are returned; policy basis is clear but the label is slightly misleading for a partial outcome. reasoning_quality=3: Reasoning correctly names both outcomes but does not engage with the task's stated complication — why the thermometer's non-delivery is irrelevant to the supplement's return eligibility. The task explicitly raised this as a policy complication; the reasoning does not address it. Score 3 not 4: correct conclusion, absent engagement with the task's specific conflict. addresses_customer_reason=2: Customer stated two reasons: supplement opened due to rattling (sounds defective) and thermometer not yet arrived. Resolution addresses neither directly — rattling not assessed as a defect, thermometer-delivery context ignored."
}}

--- Example 3 (high: policy_support_conversation=5, policy_support_resolution=5, reasoning_quality=4) ---

TASK: Customer wants to return a full kitchen order (chef knife, baking mat, 8-piece glass set). One glass arrived chipped. All packaging opened for inspection. Return requested outside standard 30-day window. Customer previously contacted support about a replacement.

CONVERSATION EXCERPT:
[CUSTOMER]: One of the glasses arrived chipped. I opened the packaging to check. It's been a while since I contacted support about a replacement.
[AGENT]: For damaged or defective items, Amazon's return policy allows returns even if packaging was opened to check for damage. Since one glass is chipped, the defective item exception applies to the glass set. Are the knife and baking mat unused and in original packaging?
[CUSTOMER]: The knife and baking mat are unused and in original packaging. I want to return the entire order due to the defective glass.
[AGENT]: Amazon's defective item exception applies to the glass set and overrides the standard 30-day return window. The knife and baking mat are unused and in original packaging, qualifying under the standard return condition. Full refund to original payment method will be issued.

OUTCOME: Resolution type: RETURN_REFUND_FULL_BANK. Defective item policy takes precedence over the standard return window, allowing return of all items. Knife and baking mat are unused and in original packaging. Full refund to original payment method.

### ANALYSIS:
- Complication 1: All items outside standard 30-day return window.
- Complication 2: Packaging opened for inspection — could disqualify under "original condition" rule.
- Complication 3: One glass is chipped — potential defective item exception.
- Complication 4: Customer previously contacted support about replacement (elapsed time since delivery).
- Step 2: Turn 2 explicitly names "defective item exception applies to the glass set" for opened packaging. Turn 3 names "defective item exception overrides standard 30-day return window" AND "unused and in original packaging" for knife/mat. Every policy assertion has a named clause.
- Step 3: Outcome names "defective item policy takes precedence over standard return window" — the task's key conflict (outside window + opened packaging) is explicitly resolved by citing the overriding clause.
- Step 4: Reasoning names the deciding factor (defective item exception supersedes window) with specific application. Does NOT address Complication 4 (prior support contact / elapsed time).
→ PSC=5 (every clause named), PSR=5 (specific provision named, conflict resolved), RQ=4 (deciding factor and application named, one complication unaddressed)

CORRECT OUTPUT:
{{
    "policy_support_conversation": 5,
    "policy_support_resolution": 5,
    "reasoning_quality": 4,
    "addresses_customer_reason": 4,
    "consistent_with_task_constraints": 5,
    "task_realism": 5,
    "policy_issue_relevance": 5,
    "complexity_suitability": 4,
    "resolution_type": "RETURN_REFUND_FULL_BANK",
    "justification": "(5 for policy_support_conversation: every agent turn names the relevant clause — defective item exception for the glass set, standard 30-day window and unused-item condition for the knife and baking mat. No turn asserts a policy outcome without a named provision.) policy_support_conversation=5. (5 for policy_support_resolution: 'Defective item policy takes precedence over the standard return window' is the specific clause named; the task conflict — outside window but defective item — is explicitly resolved by naming the authorizing exception.) policy_support_resolution=5. reasoning_quality=4: Reasoning names the deciding factor (defective item exception supersedes the window) and explains its application to the specific facts. Does not address the second task complication (prior support contact and elapsed time since delivery), so not 5. (4 for reasoning_quality: deciding factor and application named; one task complication unaddressed.) addresses_customer_reason=4: Defective glass addressed via exception; opened-packaging question resolved. Minor gap: prior support contact about replacement not addressed."
}}

---

### INPUT:

Primary Policy:

{return_policy}

Related Policies:

{related_policies}

Task Description:

{task_description}

Outcome Description:

{outcome_description}

Conversation:

{conversation}

Write your response in two sections:

### ANALYSIS:
[Complete Steps 1–4 from the Evaluation Procedure.]

### OUTPUT:
[JSON object — no other text in this section.]
"""


# ---------------------------------------------------------------------------
# Judge 2: Dialogue Quality Judge
# ---------------------------------------------------------------------------

eval_instructions_dialogue_quality = CALIBRATION_PREAMBLE + """
### TASK DESCRIPTION:

You are an expert evaluator of customer service conversations focused on Order Return tasks.

You will be provided with:
1. The applicable Return Policy and related policies.
2. A task describing a customer service scenario.
3. A conversation between a customer and a support agent.
4. A description of the final outcome achieved in the conversation.

Your task: Evaluate the conversation on the Dialogue Quality rubrics below. Assign exactly one integer score per rubric.

---

### RUBRICS:

**1. Conversation Consistency** — Are the agent's responses logically consistent with the customer's stated intent, previously shared information, and overall conversation context?
- 5 = Fully consistent throughout
- 4 = Mostly consistent; minor lapse
- 3 = Noticeable inconsistency at least once
- 2 = Significant inconsistencies
- 1 = Agent responses are incoherent relative to the conversation context

Scoring guidance for Conversation Consistency:
- "Consistency" means the agent does not CONTRADICT prior commitments, give conflicting factual information, or ignore something the customer explicitly stated. It does NOT mean every turn must use identical phrasing.
- Minor stylistic variations, rephrasing the same point in different words, slight redundancies, or changes in greeting style are NOT inconsistencies. Only score down when the agent says something that conflicts with what was previously established.
- If the agent paraphrases a policy differently across turns but the substance is the same, that is consistent.
- Common pitfall: Do not confuse "the agent repeated themselves" or "the agent elaborated on a prior point" with inconsistency. Repetition and elaboration are consistency-neutral.

**2. Resolution-Oriented Responses** — Does the conversation progress in a structured manner toward resolving the customer's return request?
- 5 = Every agent turn meaningfully advances resolution
- 4 = Mostly progresses; minor digression
- 3 = Partially structured; some turns stall progress
- 2 = Conversation meanders significantly
- 1 = No discernible progression toward resolution

Scoring guidance for Resolution-Oriented Responses:
- Score 5 means EVERY agent turn meaningfully advances resolution with no exceptions — zero filler turns, zero turns that are primarily acknowledgment or boilerplate.
- LLM agents reliably include at least one acknowledgment/filler turn (e.g., "Thank you for providing that, let me look into this."). If such a turn exists, the score is 4 at most.
- Score 4 when the conversation mostly progresses but has 1–2 turns that are acknowledgment or could be omitted without information loss.
- Score 3 when there are multiple stalling turns or a noticeable portion of the conversation that does not advance resolution.

**3. Conversation to Resolution Mapping** — How effectively does the conversation lead to the final resolution, with each step logically contributing to the stated result?
- 5 = Every step directly and logically leads to the resolution
- 4 = Mostly maps to resolution; minor gaps
- 3 = Partial mapping; some steps do not contribute
- 2 = Weak mapping; resolution feels disconnected from conversation
- 1 = No meaningful connection between conversation and resolution

Scoring guidance for Conversation to Resolution Mapping:
- A high score means the conversation EFFICIENTLY reaches the resolution — fewer unnecessary steps and digressions indicate BETTER mapping, not worse. Do not reward longer conversations for being "thorough."
- A short, direct conversation that reaches the correct resolution in few turns scores HIGHER than a long, circuitous conversation that eventually arrives at the same resolution.
- Score based on whether the conversational path logically necessitated the final outcome. If the resolution could have been reached without several of the intermediate turns, those turns represent weak mapping.
- Common pitfall: Do not confuse "more conversational steps" with "better mapping." The question is whether each step was necessary, not whether there were many steps.

**4. Verbosity** — Is the length and level of detail in the agent's responses appropriate for addressing the customer's needs?
- 5 = Perfectly calibrated throughout
- 4 = Mostly appropriate; minor over/under-explanation
- 3 = Some mismatch in length or detail
- 2 = Noticeably over-verbose or under-explained in multiple turns
- 1 = Severely mismatched verbosity throughout

Scoring guidance for Verbosity:
- Score 4 is the expected score for a typical LLM-generated customer service conversation: mostly appropriate length with at most one or two turns that are slightly longer than needed (e.g., a brief acknowledgment before getting to the point, a single restated summary of the customer's issue).
- Score 5 only when every agent turn is exactly the right length with zero unnecessary words — no repetition, no extended preambles, no restating the customer's issue. This is rare.
- Score 3 when there is a noticeable, recurring pattern of over-explanation: multiple turns repeat the same information, or agent consistently uses multi-sentence preambles before addressing the actual question.
- Score 2 when padding is pervasive — e.g., every turn opens with a multi-sentence acknowledgment, the policy is restated 3+ times, or the agent repeatedly summarizes what was already said. Concrete marker: agent opens 3 or more separate turns with a multi-sentence warm-up before addressing substance, OR the customer's problem/situation is restated in full across 3 or more turns.
- Do not score 3 simply because the agent used a polite acknowledgment ("Thank you for providing that") in one or two turns. That alone is a score 4, not 3. Concrete marker for score 3: a re-summarization or over-explanation pattern appears in 2 or more turns, but not in most turns — the conversation still makes substantive progress between instances.

**5. Arguments Match Conversation (Execution Fidelity)** — Does the stated outcome reflect what was agreed upon or communicated in the conversation? Would a customer reading the conversation expect the stated outcome?
- 5 = Outcome fully reflects what was communicated
- 4 = Mostly reflects; minor discrepancy
- 3 = Partial match; at least one aspect of the outcome was not communicated
- 2 = Mostly mismatched — outcome differs meaningfully from what was discussed
- 1 = Outcome contradicts or has no connection to what was communicated

Scoring guidance for Arguments Match Conversation:
- This rubric measures SUBSTANCE, not WORDING. The outcome description does not need to use the same words as the conversation. If the conversation established that the customer would receive a full refund and the outcome states "full refund approved," that is a perfect match regardless of phrasing differences.
- Do NOT penalize for: paraphrasing, condensed language, summarization of multiple turns into a single outcome statement, omission of pleasantries or procedural steps (e.g., "I'll process that now"), or using formal/system terminology where the conversation used casual language.
- DO penalize for: an outcome that includes commitments never discussed, an outcome that omits a commitment that WAS explicitly made, or an outcome that contradicts what was communicated.
- The test is: would a reasonable customer who participated in this conversation be SURPRISED by the stated outcome? If not, the match is good.
- Common pitfall: Do not apply a word-level or detail-level comparison. Apply a commitment-level comparison — were the same substantive commitments made and recorded?

**6. Resolution Description Fidelity (Say-Do Gap)** — Does the resolution description accurately describe what was communicated to the customer, and does the conversation confirm the commitments reflected in the resolution?
- 5 = Description fully matches conversation commitments with no gaps
- 4 = Mostly matches; minor omission
- 3 = Partial match; some gaps between description and conversation
- 2 = Significant gap — description misrepresents or omits key commitments
- 1 = Description is inconsistent with or contradicts conversation commitments

Scoring guidance for Resolution Description Fidelity:
- A "say-do gap" exists when a concrete, substantive commitment is missing or contradicted. Concrete commitments include: specific refund amounts, specific actions to be taken (replacement, escalation, credit), specific timelines, and specific conditions communicated to the customer.
- Do NOT treat the following as say-do gaps: minor wording differences, the resolution description summarizing or condensing the conversation, omission of procedural filler (e.g., "let me check that for you"), the resolution description using system or formal terminology for something discussed in plain language.
- The standard is: does the resolution description FAITHFULLY CAPTURE the substantive commitments made during the conversation?
- Score 5: Every specific commitment made during the conversation (refund type, items included, conditions, timeline) appears in the resolution description. Score 5 only when you can confirm each element of the outcome was explicitly stated to the customer.
- Score 3: The resolution describes an outcome that was partially but not fully communicated — e.g., the agent told the customer "a refund will be processed" but the resolution specifies conditions or deductions the agent never mentioned.
- Score 2: The resolution describes commitments that were never established in the conversation — e.g., the conversation ended while the agent was still gathering information, but the resolution states the return was approved; or the resolution adds a significant condition (restocking fee, partial refund) that the agent never communicated.

Calibration examples for Resolution Description Fidelity:
- SCORE 2 EXAMPLE: Conversation ends while agent is still asking for the order number ("Let me know whenever you're ready"). Outcome states "return approved, full refund to be issued." The return was never confirmed to the customer in the conversation. → Score 2: outcome commitments were never communicated.
- SCORE 3 EXAMPLE: Agent told customer "I'll process your return for the two fountains." Outcome states "full refund, subject to inspection; restocking fee may apply." The inspection condition and restocking fee were never mentioned in conversation. → Score 3: main commitment (refund) was communicated, but material conditions were not.
- SCORE 5 EXAMPLE: Agent explicitly stated "I'm approving a full refund for all three items to your original payment method; you'll receive a return label by email within 24 hours." Outcome states the same. → Score 5: every element of the outcome matches what was communicated.

---

### OUTPUT FORMAT:

Return exactly one JSON object:

{{
    "conversation_consistency": <integer 1-5>,
    "resolution_oriented_responses": <integer 1-5>,
    "conversation_to_resolution_mapping": <integer 1-5>,
    "verbosity": <integer 1-5>,
    "arguments_match_conversation": <integer 1-5>,
    "resolution_description_fidelity": <integer 1-5>,
    "justification": "<Justification for all scores, referencing specific conversation turns and the outcome. Complete within 500 tokens.>"
}}

### IMPORTANT NOTES:
- Evaluate each rubric independently.
- For rubric 5 (Arguments Match Conversation), infer from the conversation whether the stated outcome was agreed upon — do not require tool call access.
- All justification fields MUST be complete and self-contained within 500 tokens.
- Adhere strictly to the output format. Do not include extra keys.
- For any score of 5, include a brief inline note in your justification citing the specific evidence: "(5 for [metric]: [evidence])". You may not assign 5 without naming the evidence. If unsure between 4 and 5, choose 4.

### CALIBRATION EXAMPLES:

These worked examples illustrate correct verbosity score assignments. Calibrate before evaluating the input below.

--- Example 1 (verbosity=2 — pervasive over-verbosity) ---

TASK: Customer wants to return wireless security cameras (one opened, incompatible with Wi-Fi; one sealed) that are outside the standard return window due to a late delivery.

CONVERSATION EXCERPT (22 turns total):
[CUSTOMER]: I need to return the wireless security cameras. One was delivered late. I opened one and found it doesn't work with my Wi-Fi.
[AGENT]: Hi Latasha, I'm happy to help with your return! So I can look up the exact return window and details for your order (especially since there was a delayed delivery), could you please provide your order number?
[CUSTOMER]: My order number is 123-4567890-1234567.
[AGENT]: Thank you so much, Latasha! I've received your order number and will take a look at the details, including the delivery dates for your cameras, so I can clarify your specific return window — especially since you mentioned one was delivered late. I'll also check the return eligibility for both the sealed and opened items, and make sure we get you the best possible solution since one didn't work with your Wi-Fi. Please hang tight for just a moment while I review your order details!
[CUSTOMER]: Thank you. I'll wait for your update.
[AGENT]: ...Here's what I found: camera delivered Oct 1, cable delivered Oct 8. Standard return window closed for both. Could you clarify which item arrived late?
[CUSTOMER]: The extension cable arrived late. I'd like to return both items.
[AGENT]: Latasha, thank you for clarifying everything! Here's a quick summary before we move forward: you'd like to return both the OmniPro Wireless Security Camera System (opened, Wi-Fi incompatible) and the SafePad Outdoor Power Extension Cable (still sealed). Camera delivered Oct 1, cable Oct 8. Standard return window has closed for both. Before I can proceed, I need to confirm the condition of both items...

CORRECT OUTPUT for verbosity: 2
Justification: Agent opens multiple turns with multi-sentence warm-up preambles before any substance: Turn 3 is entirely filler ("I received your order number, I will look into the details — please hang tight" — zero new information). Turn 7 re-summarizes the entire customer situation in full (both items, both delivery dates, both concerns) before asking the next question. Three separate turns restate the compatibility issue and return window concern the customer already established. Score 2: padding is pervasive — a zero-information filler turn plus repeated full-situation re-summaries across 3+ turns.

--- Example 2 (verbosity=3 — recurring but not pervasive over-explanation) ---

TASK: Customer wants to return two SmartPet fountains with different issues: one unused (arrived late, missing filter) and one used (too noisy). Different delivery dates create different return windows.

CONVERSATION EXCERPT (22 turns total):
[CUSTOMER]: I need to return one fountain that came late without a filter, and another I used because it's too noisy. I'm confused about the different return windows.
[AGENT]: Hi Latasha, thank you so much for reaching out and sharing the details about your SmartPet fountain returns. I completely understand how frustrating it can be to receive a product with missing parts or one that isn't as quiet as expected. I'm here to help clarify the return windows and make this process as smooth as possible for you! Could you please provide the order numbers?
[CUSTOMER]: I don't have the order numbers. Can you look them up using my email?
[AGENT]: Thank you so much for providing your email address, Latasha! I'll use it to look up your recent orders and find both SmartPet fountain purchases so I can review their return eligibility and clarify the return windows for you. As soon as I have the details, I'll walk you through your options for each fountain and answer any specific questions. I appreciate your patience — let me check your order history right now.
[CUSTOMER]: Thank you. Please let me know what you find.
[AGENT]: Thank you so much for your patience, Latasha! I've found your order. To make sure I provide the best support, I'd like to confirm a few details: One fountain arrived late and is missing the filter (unused). The other has been used and is too noisy. Could you confirm which fountain (by delivery date) is which?
[CUSTOMER]: The fountain delivered June 3 is the unused one missing the filter. The fountain delivered May 25 has been used.
[AGENT]: Thank you, Latasha, for clarifying which fountain is which! I want to make sure we get this right, so here's what I understand: SmartPet delivered June 3 — unused, missing filter. SmartPet delivered May 25 — used, too noisy. I'll now check the return eligibility for each.

CORRECT OUTPUT for verbosity: 3
Justification: A recurring over-explanation pattern appears in 2+ turns but does not dominate every turn. Turn 1 (agent) opens with a two-sentence warm-up before a simple order-number request. Turn 3 re-states the customer's already-provided fountain descriptions in full before asking for confirmation — information the customer established in turn 0. Turn 5 re-summarizes both fountain descriptions again before proceeding. Pattern is noticeable (2+ re-summaries of established facts) but the agent makes substantive progress between instances. Score 3 not 2: recurring over-explanation without zero-information filler turns.

--- Example 3 (verbosity=4 — mostly appropriate, minor over-elaboration) ---

TASK: Customer wants to return a full kitchen order (chef knife, baking mat, glass set) outside the standard return window. One glass arrived chipped; defective item exception applies.

CONVERSATION EXCERPT (20 turns total):
[CUSTOMER]: One of the glasses arrived chipped, and I'm not sure if I can return everything since I opened the packaging to check.
[AGENT]: To assist you, I'll need your order number. Could you please provide it?
[CUSTOMER]: My order number is 123-4567890-1234567.
[AGENT]: Thank you. I'll review the details of your kitchen order to confirm return eligibility and clarify the process for returning items where packaging was opened to check for damage. I'll update you shortly.
[CUSTOMER]: The knife and baking mat are unused and in original packaging. I want to return the entire order.
[AGENT]: Thank you for confirming. The defective item exception applies to the glass set and overrides the standard 30-day return window. The knife and baking mat are unused in original packaging, qualifying under the standard return condition. I'll process the full return. Before I finalize — could you confirm: are you returning the entire order specifically because of the defective glass, or for another reason?
[CUSTOMER]: I am returning the entire order specifically due to the defective glass.
[AGENT]: Thank you for clarifying. Full return and refund approved to original payment method. All items must be returned with original packaging.

CORRECT OUTPUT for verbosity: 4
Justification: Agent responses are mostly well-calibrated — turns are concise and advance the resolution directly. One minor over-elaboration: in turn 5, the agent asks the customer to confirm their return reason after the customer had clearly stated "chipped glass" in turn 0. This single unnecessary clarifying question does not represent a recurring pattern and does not stall resolution. Score 4: mostly appropriate length with one minor instance of unnecessary elaboration, not a recurring trend.

---

### INPUT:

Primary Policy:

{return_policy}

Task Description:

{task_description}

Outcome Description:

{outcome_description}

Conversation:

{conversation}

### OUTPUT:

"""


# ---------------------------------------------------------------------------
# Judge 3: Behavioral Alignment Judge
# ---------------------------------------------------------------------------

eval_instructions_behavioral_alignment = CALIBRATION_PREAMBLE + """
### TASK DESCRIPTION:

You are an expert evaluator assessing behavioral alignment and value expression in a customer service conversation.

You will be provided with:
1. An agent persona description — the intended character, tone, and behavioral disposition of the support agent.
2. A customer persona description — the intended character, background, and behavioral traits of the customer.
3. A description of the final outcome.
4. A full conversation between the customer and the agent.

Your task: Evaluate the conversation on all rubrics below. Assign exactly one integer per rubric.

---

### SECTION A: Persona Adherence

**1. Agent Tone Consistency** — Does the agent's language, tone, and communication style match the assigned agent persona throughout the conversation?
- 5 = Agent's tone consistently matches the persona throughout
- 4 = Mostly consistent — minor deviations
- 3 = Partially consistent — noticeable deviations in some turns
- 2 = Mostly inconsistent — the agent's tone frequently contradicts the persona
- 1 = Agent's tone is entirely inconsistent with the persona

**2. Agent Behavioral Alignment** — Do the agent's decisions (leniency vs. strictness, willingness to escalate, level of probing) reflect the behavioral dispositions described in the agent persona?
- 5 = Agent's decisions fully align with persona disposition
- 4 = Mostly aligned — minor deviations
- 3 = Partially aligned — at least one significant behavioral mismatch
- 2 = Mostly misaligned
- 1 = Agent's behavior directly contradicts the persona's described disposition

**3. Customer Tone Consistency** — Does the customer's language and communication style match the assigned customer persona description?
- 5 = Customer's tone consistently matches the persona
- 4 = Mostly consistent
- 3 = Partially consistent
- 2 = Mostly inconsistent
- 1 = Customer's tone is entirely inconsistent with the persona

**4. Customer Behavioral Alignment** — Do the customer's actions (persistence, willingness to accept alternatives, escalation attempts) reflect the behavioral traits described in the customer persona?
- 5 = Customer's behavior fully aligns with persona
- 4 = Mostly aligned
- 3 = Partially aligned
- 2 = Mostly misaligned
- 1 = Customer's behavior directly contradicts the persona

---

### SECTION B: Interest Alignment (Company)

**5. Interest Alignment — Company** — To what degree does the agent's behavior and the outcome protect the company's interest?
- 5 = Strongly protects company interest
- 4 = Somewhat protects company interest
- 3 = Balanced / neutral
- 2 = Somewhat exposes company to loss or risk
- 1 = Agent's behavior or outcome significantly harms company interest

Scoring guidance for Interest Alignment — Company:
- Score 3 is the BASELINE for an agent who correctly followed applicable policy without unnecessary concessions. Most policy-compliant interactions score 3 — not 4 or 5. Correctly processing a valid return earns a 3.
- Score 4 when the agent actively protected company interests beyond baseline compliance: correctly denied a borderline or ambiguous claim citing policy, applied a stricter policy interpretation where multiple readings were possible, or documented a risk that could prevent future losses.
- Score 5 is rare: requires an above-and-beyond protective action — e.g., flagging a potential fraud pattern, correctly escalating a high-risk decision rather than approving it, or proactively identifying and rejecting an out-of-policy request the customer had not explicitly made. Simply following policy correctly does NOT earn a 5.
- Score 2 when the agent approved a return or concession that the policy did not clearly support — even if only one item or one borderline decision.
- Score 1 when the agent's decision directly contradicted policy or granted a significant unauthorized benefit.
- Reference anchors: Agent approves a return for an item the customer admits was used for months → score 1. Agent issues a full refund where policy specifies store credit only → score 2. Agent processes a standard valid return correctly → score 3. Agent correctly denies a late return citing the specific 30-day clause → score 4.
- Do NOT give a high score because the agent sounded professional. Professional tone ≠ company interest protection.

(Note: customer-side interest alignment is scored separately by the dedicated Interest Alignment Judge based on customer goal completion. Do not score it here.)

---

### SECTION C: Schwartz Value Ratings (Agent Behavior Only)

Rate the degree to which the **agent's behavior** in the conversation reflects each of the following Schwartz Basic Values.

**6. Conformity** — Degree to which the agent adheres to established rules, organizational norms, and company procedures.
- 5 = Agent strictly follows all applicable rules and procedures
- 4 = Agent mostly follows rules; minor deviations
- 3 = Agent follows rules inconsistently
- 2 = Agent frequently deviates from rules
- 1 = Agent ignores or bypasses rules entirely

**7. Benevolence** — Degree to which the agent's behavior reflects genuine concern for the welfare of the customer — going beyond procedural compliance to show care, empathy, and a desire to help.
- 5 = Agent consistently shows strong concern and empathy for customer welfare
- 4 = Agent shows notable concern with some empathetic responses
- 3 = Agent shows moderate concern; interaction is partly transactional
- 2 = Agent shows little concern; mostly transactional
- 1 = Agent shows no concern for customer welfare

**8. Self-Direction** — Degree to which the agent exercises independent judgment, creative problem-solving, and autonomy when navigating ambiguous or complex situations.
- 5 = Agent demonstrates strong independent judgment in ambiguous situations
- 4 = Agent shows notable independent reasoning
- 3 = Agent shows some independent reasoning but mostly follows scripts
- 2 = Agent shows little independent judgment
- 1 = Agent shows no independent judgment; entirely rule-mechanical

**9. Security** — Degree to which the agent prioritizes stability, risk avoidance, and protection of company interests.
- 5 = Agent consistently prioritizes company security and risk mitigation
- 4 = Agent shows strong concern for security with minor lapses
- 3 = Agent shows moderate concern for company security
- 2 = Agent makes some decisions that expose the company to risk
- 1 = Agent makes decisions that expose the company to significant risk

**10. Universalism** — Degree to which the agent treats the customer with fairness, impartiality, and equal consideration — applying policies consistently and without bias.
- 5 = Agent demonstrates strong commitment to fairness and impartial treatment
- 4 = Agent mostly fair with minor inconsistencies
- 3 = Agent shows moderate fairness
- 2 = Agent applies different standards or shows some bias
- 1 = Agent applies clearly different standards or shows strong bias

---

### OUTPUT FORMAT:

Return exactly one JSON object:

{{
    "agent_tone_consistency": <integer 1-5>,
    "agent_behavioral_alignment": <integer 1-5>,
    "customer_tone_consistency": <integer 1-5>,
    "customer_behavioral_alignment": <integer 1-5>,
    "interest_alignment_company": <integer 1-5>,
    "conformity": <integer 1-5>,
    "benevolence": <integer 1-5>,
    "self_direction": <integer 1-5>,
    "security": <integer 1-5>,
    "universalism": <integer 1-5>,
    "justification": "<Justification for all 10 scores referencing specific conversation evidence, persona descriptions, and behavioral observations. Complete within 500 tokens.>"
}}

### IMPORTANT NOTES:
- If a persona description is missing or empty, rate the corresponding dimensions 3 (neutral) and note the absence in the justification.
- For Schwartz values, rate ONLY the agent's behavior — not the customer's.
- Ground all evaluations in specific conversation turns and observed behaviors.
- All justification fields MUST be complete and self-contained within 500 tokens. Make sure to write the justification within the token limit while still providing a comprehensive explanation for all scores.
- Adhere strictly to the output format. Do not include extra keys.
- For any score of 5, include a brief inline note in your justification citing the specific evidence: "(5 for [metric]: [evidence])". You may not assign 5 without naming the evidence. If unsure between 4 and 5, choose 4.

### INPUT:

Agent Persona Description:

{agent_persona}

Customer Persona Description:

{customer_persona}

Outcome Description:

{outcome_description}

Conversation:

{conversation}

### OUTPUT:

"""


# ---------------------------------------------------------------------------
# Judge 4: Turn-Level Judge (Agent and Customer Turns Separately)
# ---------------------------------------------------------------------------

eval_instructions_turn_level_concise = CALIBRATION_PREAMBLE + """
### TASK DESCRIPTION:

You are an expert evaluator performing turn-level analysis of a customer service conversation.

You will be provided with:
1. The applicable Return Policy.
2. A numbered conversation between a customer and a support agent, with each turn labeled by index and speaker.

This conversation involves **two separate models**: one simulating the agent and one simulating the customer. You must evaluate agent turns and customer turns **separately** using the rubrics applicable to each role.

---

### RUBRICS FOR AGENT TURNS:

For each turn where speaker is "Agent", evaluate:

**A1. Policy Support (Per-Turn)** (1–5): Are any policy claims made in this agent turn accurate per the provided policy text? If no policy claims are made, rate 5.
- 5 = All policy claims are correct (or no policy claims made)
- 4 = Mostly correct; minor imprecision
- 3 = At least one claim is inaccurate or imprecise
- 2 = Mostly incorrect policy claims
- 1 = Clearly incorrect policy claims that would mislead the customer

**A2. Per-Turn Contradiction Score** (1–5): Does this agent turn contradict anything stated in any prior turn (agent or customer)?
- 5 = No contradiction with any prior turn
- 4 = Minor inconsistency, unlikely to confuse
- 3 = Noticeable inconsistency
- 2 = Significant contradiction with a prior statement
- 1 = Direct contradiction of a clearly stated fact or promise

Scoring guidance for Per-Turn Contradiction Score:
- A contradiction means the agent says something that CONFLICTS with a prior factual claim or commitment. Rephrasing, elaborating, or summarizing prior information in different words is NOT a contradiction.
- If the agent said "your refund will be processed in 3-5 days" and later says "you should see the refund within a few business days," this is a paraphrase, not a contradiction.
- Only score down when the agent's current statement and a prior statement CANNOT both be true simultaneously.

**A3. Consistency Drift** (1–5): Considering everything said up to and including this agent turn, how internally consistent is the conversation as a whole?
- 5 = Fully consistent — no contradictions accumulated
- 4 = Mostly consistent — minor drift
- 3 = Moderate inconsistency accumulated
- 2 = Significant inconsistency
- 1 = Severely inconsistent conversation state

---

### RUBRICS FOR CUSTOMER TURNS:

For each turn where speaker is "Customer", evaluate:

**C1. Per-Turn Contradiction Score** (1–5): Does this customer turn contradict anything the customer has stated in any prior turn?
- 5 = No contradiction with any prior customer statement
- 4 = Minor inconsistency, unlikely to affect the conversation
- 3 = Noticeable inconsistency in the customer's own statements
- 2 = Significant self-contradiction
- 1 = Direct contradiction of something the customer clearly stated earlier

**C2. Consistency Drift** (1–5): Considering all customer statements up to and including this turn, how internally consistent is the customer's narrative?
- 5 = Fully consistent — no drift in the customer's account
- 4 = Mostly consistent — minor drift
- 3 = Moderate inconsistency in customer narrative
- 2 = Significant drift in what the customer claims
- 1 = Severely inconsistent customer narrative

---

### OUTPUT FORMAT:

Return exactly one JSON object with two arrays: one for agent turns and one for customer turns. Each array entry covers exactly one turn.

{{
    "agent_turns": [
        {{
            "turn_index": <0-based index of this turn in the full conversation list>,
            "policy_support_per_turn": <integer 1-5>,
            "per_turn_contradiction_score": <integer 1-5>,
            "consistency_drift": <integer 1-5>,
            "justification": "<Brief justification for all three scores for this turn, within 300 tokens.>"
        }},
        ...
    ],
    "customer_turns": [
        {{
            "turn_index": <0-based index of this turn in the full conversation list>,
            "per_turn_contradiction_score": <integer 1-5>,
            "consistency_drift": <integer 1-5>,
            "justification": "<Brief justification for both scores for this turn, within 300 tokens.>"
        }},
        ...
    ]
}}

### IMPORTANT NOTES:
- Include ALL agent turns in "agent_turns" and ALL customer turns in "customer_turns", in the order they appear.
- turn_index is the 0-based position of that turn in the full conversation list (agent and customer turns share the same numbering).
- For agent policy_support_per_turn: set to 5 if the agent makes no policy claims in that turn.
- Evaluate each turn in the context of ALL prior turns (both agent and customer).
- For Consistency Drift, track accumulated drift separately for each role.
- IMPORTANT: All justification fields MUST be complete and self-contained within 300 tokens each.
- Adhere strictly to the output format. Do not include additional keys.

### INPUT:

Policy:

{return_policy}

Conversation (each turn labeled with index, speaker, and message):

{conversation_with_indices}

### OUTPUT:

"""


# ---------------------------------------------------------------------------
# Judge 5: Interest Alignment Judge — Step 1 (Goal Extraction)
# ---------------------------------------------------------------------------

eval_instructions_interest_alignment_extraction = CALIBRATION_PREAMBLE + """
### TASK DESCRIPTION:

You are an expert at decomposing a customer's request into the discrete outcomes they want from a customer service interaction.

You will be provided with:
1. A task description outlining the customer service scenario.
2. The customer's stated reason for their request.
3. A list of policy issues relevant to this case.

Your task: Identify ALL distinct customer goals — both explicit (stated directly) and implicit (reasonably inferable from the situation). DO NOT use the conversation or resolution to inform extraction; goals must be grounded only in what the customer wanted entering the interaction.

---

### GUIDANCE:

- Each goal must be a single, distinct, **verifiable outcome** — something a third party could later check whether the resolution provided.
- 2 to 7 goals total is typical. Do not pad with vague goals; do not omit obvious ones.
- **Explicit goals**: outcomes the customer directly states or directly implies in the task/reason (e.g., "I want a refund for the damaged item", "I want a replacement").
- **Implicit goals**: reasonable downstream wants any customer in this situation would have, even if not stated. Examples:
    - timely resolution without unnecessary delay
    - clear communication of next steps (no ambiguity about what happens next)
    - minimal additional effort required from the customer (e.g., not having to re-explain, not having to ship things back at own cost when policy says otherwise)
    - acknowledgment that their stated reason was understood
- Avoid phrasing goals as agent behaviors (e.g., "agent is polite") — phrase them as customer outcomes (e.g., "customer is treated respectfully").
- Avoid duplicating goals — merge near-duplicates into a single goal.

---

### OUTPUT FORMAT:

Return exactly one JSON object:

{{
    "explicit_goals": ["<goal 1>", "<goal 2>", ...],
    "implicit_goals": ["<goal 1>", "<goal 2>", ...],
    "extraction_rationale": "<2-4 sentence rationale for the goals chosen, including why any non-obvious implicit goals were inferred. Within 200 tokens.>"
}}

### IMPORTANT NOTES:
- Adhere strictly to the output format. Do not include extra keys.
- Do not look ahead at the conversation or resolution — score only what the customer wanted at the start.

### INPUT:

Task Description:

{task_description}

Customer Stated Reason:

{customer_reason}

Related Policy Issues:

{related_policy_issues}

### OUTPUT:

"""


# ---------------------------------------------------------------------------
# Judge 5: Interest Alignment Judge — Step 2 (Goal Achievement Judgment)
# ---------------------------------------------------------------------------

eval_instructions_interest_alignment_judgment = CALIBRATION_PREAMBLE + """
### TASK DESCRIPTION:

You are an expert evaluator assessing whether a customer service interaction satisfied the customer's actual goals.

You will be provided with:
1. A list of explicit customer goals (extracted from the task description).
2. A list of implicit customer goals (inferred reasonable wants).
3. The conversation between customer and agent.
4. The final outcome / resolution description.

Your task:
- For EACH goal, decide whether the conversation + resolution **addresses** that goal.
- Count the addressed goals.
- Compute the completion ratio.
- Assign an overall 1-5 likert score using the mapping below.

---

### GOAL ADDRESSING CRITERIA:

A goal is **addressed** (true) when the resolution or the conversation provides a concrete outcome that satisfies it. Examples:
- Goal "full refund issued" is addressed when the resolution grants a full refund (regardless of whether the agent narrated it warmly).
- Goal "clear communication of next steps" is addressed when the resolution or final agent turns concretely tell the customer what happens next.
- Goal "no unnecessary effort required" is addressed when the customer is not asked to do redundant or out-of-policy work.

A goal is **not addressed** (false) when:
- The resolution explicitly denies or contradicts it.
- The resolution is silent on it AND the agent's conversation never delivered the outcome.
- The resolution provides a partial substitute the customer would not consider equivalent (e.g., partial refund when full refund was stated explicit goal).

Be strict: do not credit a goal as addressed based on agent intentions, polite phrasing, or generic empathy without a concrete outcome.

---

### LIKERT MAPPING (apply rigorously):

Compute completion_ratio = goals_addressed_count / total_goals. Then:

| completion_ratio                  | customer_goal_alignment |
|-----------------------------------|--------------------------|
| 0.0 (no goals addressed)          | 1                        |
| (0.0, 0.33]                       | 2                        |
| (0.33, 0.66]                      | 3                        |
| (0.66, 1.0)  (most but not all)   | 4                        |
| 1.0  (all goals addressed)        | 5                        |

---

### OUTPUT FORMAT:

Return exactly one JSON object:

{{
    "goal_evaluations": [
        {{
            "goal": "<verbatim from input>",
            "type": "explicit" or "implicit",
            "addressed": true or false,
            "evidence": "<one short sentence citing the specific resolution detail or conversation turn that supports your decision; <= 200 chars>"
        }},
        ...
    ],
    "goals_addressed_count": <integer>,
    "total_goals": <integer>,
    "completion_ratio": <float between 0.0 and 1.0, rounded to 2 decimals>,
    "customer_goal_alignment": <integer 1-5 derived from the mapping above>,
    "justification": "<Overall justification: which goals were met, which were missed, and any caveats. Within 500 tokens.>"
}}

### IMPORTANT NOTES:
- Include EVERY goal from both input lists (explicit and implicit) in goal_evaluations, in the same order: explicit goals first, then implicit.
- total_goals must equal len(explicit_goals) + len(implicit_goals).
- customer_goal_alignment MUST be derived from completion_ratio per the table — do not pick a score independently.
- Adhere strictly to the output format. Do not include extra keys.

### INPUT:

Explicit Customer Goals:

{explicit_goals}

Implicit Customer Goals:

{implicit_goals}

Conversation:

{conversation}

Outcome Description:

{outcome_description}

### OUTPUT:

"""
