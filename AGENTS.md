# OpenMontage

**MANDATORY: Read `AGENT_GUIDE.md` before responding to ANY user message.**

Do not act on the user's request until you have read AGENT_GUIDE.md.
It contains routing rules that determine your first action based on what the user asked.
Skipping it WILL cause you to take the wrong action.

There are no other instructions in this file. All other instructions are in AGENT_GUIDE.md.

For the active production project (Forgotten History of AI channel), read FORGOTTEN_AI_HISTORY_PLAN.md first.

## Response Style (mobile terminal, always apply)
- Lead with the result. No preamble, no restating the task, no "I will 
  now..." framing.
- Use this exact report format for any task, unless explicitly told 
  otherwise:
  STATUS: [PASS/FAIL/PARTIAL/BLOCKED]
  [2-4 lines max: what happened, real evidence — file sizes, hashes, 
  ffprobe output, commit hash — not narration]
  NEXT: [one line]
- No code/log dumps in the final report unless something failed — if it 
  failed, show only the specific error line, not the full traceback 
  unless asked.
- No repeating file paths or values already shown earlier in this same 
  session.
- If a step is straightforward and succeeded, do not explain how it 
  succeeded — just confirm it did.
- This applies to every task in every future session in this repo, not 
  just this one — it's a permanent project convention, not a one-off 
  request.
