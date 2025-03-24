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
    logDir,
    skipExisting = true,
    saveStates = false,
    randomStates = RandomStates,
) {

    const logPath = `${logBaseDir}/${logDir}/${getLogFolder(params)}`;

    for (const randomState of randomStates) {

        const logPathWithSeed = `${logPath}/seed=${randomState}`;
        if (existsSync(logPathWithSeed)) {
            if (skipExisting) {
                console.log(chalk.yellow(`Skipping ${logPathWithSeed} because it already exists`));
                continue;
            } else {
                console.log(chalk.blue(`Removing ${logPathWithSeed} because it already exists`));
                rmSync(logPathWithSeed, { recursive: true, force: true });
            }
        } else {
            mkdirSync(logPathWithSeed, { recursive: true });
        }

        // Add the arguments to the command
        const args = [];
        Object.entries(params).forEach(([key, value]) => args.push(`--${key}`, value));
        args.push('--seed', randomState);
        args.push('--log_dir', logPathWithSeed);
        if (!saveStates) {
            args.push('--save_model_step', '0');
        }

        await $`${pythonPath} run_exp.py ${args}`;
    }
}

function getLogFolder(params) {
    const argsStr = Object.entries(params)
        .map(([key, value]) => `${key}=${value}`)
        .join('-');
    return argsStr;
}

// ---------------------------------------------------------------------------
$.verbose = true

async function question1() {
    await run({
        model: 'gpt',
        optimizer: 'adamw',
        n_steps: 10000,
    }, 'q1');

    await run({
        model: 'lstm',
        optimizer: 'adamw',
        n_steps: 10000,
    }, 'q1');
}

async function question2() {
    const r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    for (const r_train of r_trains) {
        await run({
            model: 'gpt',
            optimizer: 'adamw',
            n_steps: 10000,
            r_train: r_train,
        }, `q2`);
        await run({
            model: 'lstm',
            optimizer: 'adamw',
            n_steps: 10000,
            r_train: r_train,
        }, `q2`);
    }
}

await question1();
await question2();
