#!/usr/bin/env zx

import { existsSync, mkdirSync, rmSync } from 'fs';

$.shell = 'zsh';

await $`python q1-results.py`;
await $`python q2-results.py`;
await $`python q3-a-results.py`;
await $`python q3-b-results.py`;
await $`python q4-a-results.py`;
await $`python q4-b-results.py`;
await $`python q5-a-results.py`;
await $`python q5-b-results.py`;
await $`python q6-a-results.py`;