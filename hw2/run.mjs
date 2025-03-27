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

async function question3() {
    const r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    for (const r_train of r_trains) {
        await run({
            model: 'gpt',
            optimizer: 'adamw',
            n_steps: 10000,
            r_train: r_train,
        }, `q3`);
        await run({
            model: 'lstm',
            optimizer: 'adamw',
            n_steps: 10000,
            r_train: r_train,
        }, `q3`);
    }
}

async function question4() {
    const operation_orders = ['2,3'];
    const p = 11;
    const reductions = [undefined, 'none'];
    for (const operation_order of operation_orders) {
        for (const reduction of reductions) {
            const args = {
                optimizer: 'adamw',
                n_steps: 10000,
                operation_orders: operation_order,
                p,
            }
            if (reduction) {
                args.reduction = reduction;
            }

            await run({
                model: 'gpt',
                ...args,
            }, `q4`);
            await run({
                model: 'lstm',
                ...args,
            }, `q4`);
        }
    }
}

async function question5() {
    const nums_layers = [1, 2, 3];
    const embedding_sizes = [64, 128, 256];

    for (const num_layers of nums_layers) {
        for (const embedding_size of embedding_sizes) {
            await run({
                model: 'gpt',
                optimizer: 'adamw',
                n_steps: 10000,
                num_layers,
                embedding_size,
            }, `q5`);
            await run({
                model: 'lstm',
                optimizer: 'adamw',
                n_steps: 10000,
                num_layers,
                embedding_size,
                hidden_size: embedding_size,
            }, `q5`);
        }
    }
}

async function question6() {
    const batch_sizes = [32, 64, 128, 256, 512];
    const n_steps = 2 * 10000 + 1;

    for (const batch_size of batch_sizes) {
        await run({
            model: 'gpt',
            optimizer: 'adamw',
            n_steps,
            train_batch_size: batch_size,
        }, `q6`);
        await run({
            model: 'lstm',
            optimizer: 'adamw',
            n_steps,
            train_batch_size: batch_size,
        }, `q6`);
    }
}

async function question7() {
    const weight_decays = [0.25, 0.5, 0.75, 1.0];
    const n_steps = 4 * 10000 + 1;

    for (const weight_decay of weight_decays) {
        await run({
            model: 'lstm',
            optimizer: 'adamw',
            n_steps,
            weight_decay,
        }, `q7`);
    }
}

async function question8() {
    const n_steps = 2 * 10000 + 1;
    await run({
        model: 'gpt',
        optimizer: 'adamw',
        n_steps: n_steps,
        save_model_step: n_steps * 2,
        p: 11,
    }, 'q8', true, true, [0]);
}

await question1();
await question3();
await question4();
await question5();
await question6();
await question7();
await question8();
