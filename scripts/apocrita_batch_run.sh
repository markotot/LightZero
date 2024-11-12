set APOC_USERNAME [lindex $argv 0]
set APOC_PASSPHRASE [lindex $argv 1];
set APOC_PASSWORD [lindex $argv 2];
set APOC_PRIVATE_KEY [lindex $argv 3];

set GIT_BRANCH [lindex $argv 4];
set PROJECT_NAME [lindex $argv 5];
set WANDB_API_KEY [lindex $argv 6];

set NUM_SEEDS [lindex $argv 7];
set RUN_NAME [lindex $argv 8];


environments=(
    "AlienNoFrameskip-v4"
    "AmidarNoFrameskip-v4"
    "AssaultNoFrameskip-v4"
    "AsterixNoFrameskip-v4"
    "BankHeistNoFrameskip-v4"
    "BattleZoneNoFrameskip-v4"
    "BoxingNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "ChopperCommandNoFrameskip-v4"
    "CrazyClimberNoFrameskip-v4"
    "DemonAttackNoFrameskip-v4"
    "FreewayNoFrameskip-v4"
    "FrostbiteNoFrameskip-v4"
    "GopherNoFrameskip-v4"
    "HeroNoFrameskip-v4"
    "JamesbondNoFrameskip-v4"
    "KangarooNoFrameskip-v4"
    "KrullNoFrameskip-v4"
    "KungFuMasterNoFrameskip-v4"
    "MsPacmanNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PrivateEyeNoFrameskip-v4"
    "QbertNoFrameskip-v4"
    "RoadRunnerNoFrameskip-v4"
    "SeaquestNoFrameskip-v4"
    "UpNDownNoFrameskip-v4"
)
for env in "${environments[@]}"; do
    echo "Submitting job for environment: $env"
done