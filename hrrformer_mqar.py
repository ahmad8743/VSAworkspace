import itertools

from zoology.config import (
    TrainConfig,
    ModelConfig,
    DataConfig,
    ModuleConfig,
    LoggerConfig,
)

from zoology.data.multiquery_ar import MQARConfig


input_seq_len = 64
vocab_size = 128
num_kv_pairs = 4

learning_rates = [
    1e-4,
    # 3e-4,
    # 1e-3,
]

d_models = [
    # 128,
    # 256,
    # 512,
    8192,
]


configs = []


for d_model, lr in itertools.product(
    d_models,
    learning_rates
):
    config = TrainConfig(
        run_id=f"HRROcto_d{d_model}_lr{lr}",
        max_epochs=10,
        learning_rate=lr,
        logger=LoggerConfig(
            project_name="vsa_attention",
            entity="a-h-hamzeh-washington-university-in-st-louis"
        ),
        data=DataConfig(
            train_configs=[
                MQARConfig(
                    num_examples=10_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            test_configs=[
                MQARConfig(
                    num_examples=1_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            batch_size=32,
        ),

        model=ModelConfig(
            name=f"HRROcto_d{d_model}_lr{lr}",
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            max_position_embeddings=input_seq_len,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.hrrformer.HRRFormerOctonionMixer",
                kwargs={
                    "num_heads": 1,
                }
            ),
        ),
    )

    configs.append(config)


for d_model, lr in itertools.product(
    d_models,
    learning_rates
):
    config = TrainConfig(
        run_id=f"MHA_d{d_model}_lr{lr}",
        max_epochs=10,
        learning_rate=lr,
        logger=LoggerConfig(
            project_name="vsa_attention",
            entity="a-h-hamzeh-washington-university-in-st-louis"
        ),
        data=DataConfig(
            train_configs=[
                MQARConfig(
                    num_examples=10_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            test_configs=[
                MQARConfig(
                    num_examples=1_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            batch_size=16,
        ),

        model=ModelConfig(
            name=f"MHA_d{d_model}_lr{lr}",
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            max_position_embeddings=input_seq_len,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.attention.MHA",
                kwargs={
                    "num_heads": 1,
                    "dropout": 0.0,
                }
            ),
        ),
    )

    configs.append(config)

for d_model, lr in itertools.product(
    d_models,
    learning_rates
):
    config = TrainConfig(
        run_id=f"MyHRR_d{d_model}_lr{lr}",
        max_epochs=10,
        learning_rate=lr,
        logger=LoggerConfig(
            project_name="vsa_attention",
            entity="a-h-hamzeh-washington-university-in-st-louis"
        ),
        data=DataConfig(
            train_configs=[
                MQARConfig(
                    num_examples=10_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            test_configs=[
                MQARConfig(
                    num_examples=1_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            batch_size=32,
        ),

        model=ModelConfig(
            name=f"MyHRR_d{d_model}_lr{lr}",
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            max_position_embeddings=input_seq_len,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.hrrformer.MyHRRFormerMixer",
                kwargs={
                    "num_heads": 1,
                    "dropout": 0.0,
                }
            ),
        ),
    )

    configs.append(config)

for d_model, lr in itertools.product(
    d_models,
    learning_rates
):
    config = TrainConfig(
        run_id=f"HRR_d{d_model}_lr{lr}",
        max_epochs=10,
        learning_rate=lr,
        logger=LoggerConfig(
            project_name="vsa_attention",
            entity="a-h-hamzeh-washington-university-in-st-louis"
        ),
        data=DataConfig(
            train_configs=[
                MQARConfig(
                    num_examples=10_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            test_configs=[
                MQARConfig(
                    num_examples=1_000,
                    vocab_size=vocab_size,
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                )
            ],
            batch_size=32,
        ),

        model=ModelConfig(
            name=f"HRR_d{d_model}_lr{lr}",
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            max_position_embeddings=input_seq_len,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.hrrformer.HRRFormerMixer",
                kwargs={
                    "num_heads": 1,
                    "dropout": 0.0,
                }
            ),
        ),
    )

    configs.append(config)