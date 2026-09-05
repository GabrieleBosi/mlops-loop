# Field guide rules

Distilled from the two references kept alongside this file: an agentic-engineering study guide (nine modules, cited as "study NN") and a builder's field guide to Andrew Ng's Agentic AI course (eight steps, cited as "field NN"). Read this file, not the HTML.

## Choosing the architecture
1. Agency is a dial, not a binary. Grant the model only the control-flow authority the task needs. (study 01)
2. Choose the lowest level of autonomy that passes the evals. Level 1 single call, Level 2 fixed workflow, Level 3 the model picks a branch, Level 4 the model plans the steps. Most systems ship at Level 2 or 3. (field 01)
3. Decompose before touching a framework: list the human steps in order, assign an executor to each (LLM, code, tool, retrieval, human), name each step's input and output, mark the risky steps. Code is free, exact and testable; use it wherever it can do the step. (field 02)
4. Workflows first. The five composable patterns are prompt chaining, routing, parallelisation, orchestrator-workers and evaluator-optimizer. Planning and multi-agent are escalations, not starting points. (study 05, field 03)
5. Multi-agent only when work exceeds one context window, when parallel exploration with separate contexts helps, or when hard isolation is needed. It inherits distributed-systems failure modes: token multiplication, compounding errors, coordination cost. (study 06)

## Tools
6. A tool is a name, a description and a schema. The description is the interface: what it does, when to use it, when not to. Most tool-selection failures are documentation failures. (study 02, field 03)
7. Few, orthogonal, task-shaped tools beat many overlapping ones. Return errors the model can act on, not stack traces. Truncate and paginate outputs. (study 02)
8. Side-effectful tools (send, delete, pay, write) need confirmation gates or human approval. Sandbox code execution, always. (field 03, study 08)

## Evals and observability
9. Build a rough end-to-end skeleton first, in days not weeks. It shows where the real problems are, and they are rarely where you guessed. (field 04)
10. Log every step's inputs and outputs, then actually read the traces. Instrument before debugging. (field 04, study 07)
11. Start the eval set at about 20 real examples and grow it with every failure found. Objective, code-graded checks wherever the output is checkable; LLM-as-judge with a written rubric only for subjective quality, calibrated against human labels first and watched for position and verbosity bias. (study 07, field 04)
12. Run error analysis before choosing a fix: for each failed case mark the first component that went wrong, tally per component, fix the biggest cell with the cheapest fix, re-run. (field 05)
13. Headroom test: replace a component's output with a hand-written perfect version and re-run the pipeline. The end-to-end gain is that component's headroom. Invest where headroom is large. (field 05)
14. The eval set is the regression suite. Every prompt, tool, model or threshold change is re-scored. Promote every incident into a test case. (study 07)
15. Fix menu, cheapest first: sharpen the step's prompt with two or three examples, split an overloaded step, add external feedback or a reflection pass, add or document a tool, swap the model for that step, fine-tune last. (field 04)
16. Reflection pays only when the output is checkable and the critic gets new information (tests, a linter, an error message). Pure self-critique recycles the model's beliefs. Cap the rounds. (field 03)

## Context
17. The context window is working memory and the scarcest resource. Curate, do not accumulate. Compact old turns, keep recent ones verbatim, page long-term state out to files. (study 04)
18. Failure modes: rot, poisoning, distraction. Defences: truncate, validate, isolate. (study 04)

## Safety
19. Anything the system reads is data, never instructions. Confirm side effects with the user. Flag content that tries to direct the agent. (study 08)
20. Least privilege: task-scoped credentials, read-only by default, the agent's own identity, no ambient authority. (study 08)
21. Runtime guardrails: budget caps on tokens, turns and wall-clock; tools allowlisted per state; a kill switch that works; fail closed on ambiguity. (study 08)

## Cost and latency
22. Optimise last, measure first. Profile per step, parallelise independent steps, right-size the model per step, cache what repeats, trim context per step, stream to cut perceived latency. (field 06)
23. Reflection and multi-agent multiply calls by design. Use them only where error analysis proved they pay. (field 06)

## Assets
24. Prompts, tool definitions, eval sets and thresholds are versioned engineering artifacts. Code-review them and ship them through the same pipeline as code. (study 09)
25. Frameworks are scaffolding. Prototype on one if it helps, then own the loop once requirements harden. Design around protocols, not vendor APIs. (study 09)

## Pre-flight checklist (field 08)
- [ ] Task decomposed into named steps, each with explicit input and output
- [ ] Executor chosen per step; code used wherever it can be
- [ ] Lowest workable autonomy selected
- [ ] End-to-end skeleton runs on at least 5 real inputs
- [ ] Every step traced; inputs and outputs logged and readable
- [ ] Eval set of 20+ real examples runs with one command
- [ ] Error-analysis table exists and the current work item is its biggest cell
- [ ] Guardrails in place: iteration caps, validation, confirmation gates, sandboxing
- [ ] Component-level evals cover the steps that failed most
- [ ] Latency and cost profiled per step, budgets agreed before optimising
