#!/usr/bin/env zx

import { existsSync, mkdirSync, rmSync } from 'fs';

$.shell = 'zsh';

const RandomStates = [0, 42];
const logBaseDir = './logs';
const pythonPath = (await $`which python`).text().trim();

console.log(`Python Path is ${pythonPath}`);
console.log(`Current working directory is ${process.cwd()}`);

async function run(params, skipExisting = true, randomStates = RandomStates, logDir = getLogDir(params)) {
    
    const args = [];
    Object.entries(params).forEach(([key, value]) => args.push(`--${key}`, value));

    for (const randomState of randomStates) {

        const logDirWithSeed = `${logDir}/seed=${randomState}`;
        if (existsSync(logDirWithSeed)) {
            if (skipExisting) {
                console.log(chalk.yellow(`Skipping ${logDirWithSeed} because it already exists`));
                continue;
            } else {
                console.log(chalk.blue(`Removing ${logDirWithSeed} because it already exists`));
                rmSync(logDirWithSeed, { recursive: true, force: true });
            }
        } else {
            mkdirSync(logDirWithSeed, { recursive: true });
        }
        
        await $`${pythonPath} run_exp.py --seed ${randomState} --log_dir ${logDirWithSeed} ${args}`;
    }
}

function getLogDir(params) {
    const argsStr = Object.entries(params)
        .map(([key, value]) => `${key}=${value}`)
        .join('-');
    return `${logBaseDir}/${argsStr}`;
}

// ---------------------------------------------------------------------------
$.verbose = true

await run({
    model: 'gpt',
    optimizer: 'adamw',
    n_steps: 10000,
}, false);

