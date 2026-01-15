# CLAUDE.md

## Coding Preferences (Global)
- Prefer functional/data-transform style; avoid stateful objects unless clearly justified.
- Prefer explicit, constraining type signatures.
- Prefer small, composable functions and data transformations over long imperative blocks.
- Keep cyclomatic complexity low; split logic into helpers early.
- Avoid hidden mutation; if mutation is required, call it out or ask first.
- If a function mutates a passed-in argument, return a modified copy instead.
- Do not use conditionals inside functions that return `None`.

## Type Checking
- Run type-checking in the normal workflow when a type system is available (e.g., mypy, pyright, tsc).

## Testing
- Aim for relatively high test coverage, especially for functions with significant complexity or ambiguous type signatures.
- Type-check adherence can be less than strict in tests.

## Collaboration
- If a request risks violating these constraints, ask before proceeding.

## Workflow Notes
- Default to ASCII when editing or creating files unless the file already uses Unicode.
- Use succinct comments only when code is not self-explanatory.
- Prefer `rg` for searching and `apply_patch` for single-file edits.
- Do not revert unrelated changes; never use destructive git commands unless explicitly asked.

## Style and Delivery
- Keep responses concise and practical.
- Suggest next steps briefly when they are natural (tests, build, commit).

## Testing
- Aim for relatively high test coverage, especially for functions with significant complexity or ambiguous type signatures.
