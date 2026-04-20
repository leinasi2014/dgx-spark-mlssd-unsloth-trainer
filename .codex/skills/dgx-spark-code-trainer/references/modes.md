# Training modes

## Sequential mode

- Goal: maximize code performance first, then internalize repository skills.
- Training order: code SSD dataset -> code adapter -> skill0 dataset -> skill0 continuation.
- Best when code generation quality is the primary KPI.

## Mixed mode

- Goal: co-train code ability and skill internalization together.
- Training order: code SSD dataset + skill0 dataset -> mixed dataset -> one adapter.
- Best when runtime dependence on explicit skills should be reduced as early as possible.
