#!/bin/bash

set -e

# Définir les valeurs pour la grid search
embed_dims=(256 512 1024)
num_blocks=(4 6 8)
drop_rates=(0.0 0.3 0.5)

function generate_config() {
    # Paramètres pour le fichier de config
    local embed_dim=$1
    local num_blocks=$2
    local drop_rate=$3
    local config_name=$4

    # Créer le dossier de config s'il n'existe pas
    mkdir -p ./model_configs/question-6

    # Générer le fichier JSON
    cat > "./model_configs/question-6/${config_name}.json" << EOF
{
    "num_classes": 10,
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": ${embed_dim},
    "num_blocks": ${num_blocks},
    "drop_rate": ${drop_rate},
    "activation": "gelu"
}
EOF
}

function run_experiment() {
    local embed_dim=$1
    local num_blocks=$2
    local drop_rate=$3
    
    # Créer un nom unique pour l'expérience
    local exp_name="mlpmixer_e${embed_dim}_b${num_blocks}_d${drop_rate//.}"
    local config_path="./model_configs/question-6/${exp_name}.json"
    
    echo "Running experiment with embed_dim=${embed_dim}, num_blocks=${num_blocks}, drop_rate=${drop_rate}"
    
    # Générer le fichier de configuration
    generate_config $embed_dim $num_blocks $drop_rate $exp_name
    
    # Lancer l'expérience
    python main.py \
        --model mlpmixer \
        --model_config $config_path \
        --visualize \
        --logdir "./logs/question-6/${exp_name}"
}

# Exécuter la grid search
for embed_dim in "${embed_dims[@]}"; do
    for num_blocks in "${num_blocks[@]}"; do
        for drop_rate in "${drop_rates[@]}"; do
            run_experiment $embed_dim $num_blocks $drop_rate
        done
    done
done 