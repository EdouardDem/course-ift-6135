#!/usr/bin/env zx

import { existsSync, mkdirSync, rmSync } from 'fs';

$.shell = 'zsh';

const RandomStates = [0, 42];
const logBaseDir = './logs';
const pythonPath = (await $`which python`).text().trim();

console.log(`Python Path is ${pythonPath}`);
console.log(`Current working directory is ${process.cwd()}`);

async function run(
    params,
    skipExisting = true,
    saveStates = false,
    randomStates = RandomStates,
    logDir = getLogDir(params)
) {

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

        // Add the arguments to the command
        const args = [];
        Object.entries(params).forEach(([key, value]) => args.push(`--${key}`, value));
        args.push('--seed', randomState);
        args.push('--log_dir', logDirWithSeed);
        if (!saveStates) {
            args.push('--save_model_step', '0');
        }

        await $`${pythonPath} run_exp.py ${args}`;
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

async function question1() {
    await run({
        model: 'gpt',
        optimizer: 'adamw',
        n_steps: 10000,
    }, false);

    await run({
        model: 'lstm',
        optimizer: 'adamw',
        n_steps: 10000,
    }, false);
}

await question1();

