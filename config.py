from collections import OrderedDict
config={
    "data_type": "energy",
    "discriminator_type": "FDDiscriminator", # SigKerMMDDiscriminator, TruncatedDiscriminator, CDEDiscriminator, FDDiscriminator
    "model_save_root": "../checkpoints",
    "seed": 0,
    "device": "cuda:0",
    "data_config": {
        "output_dim": 1,
        "batch_size": 128,
        "path_length": 64, # 64
        "dataset_size": 128 * 256,
        "normalisation": "mean_var",
        "scale": 1e0,
        #"data_type": "forex",
        "forex_pairs": ["BRENT", "DIESEL", "GAS", "LIGHT"],
        "stride_length": 1,
        "frequency": "M1", # "H1"
        "filter_extremal_paths": False,
        "filter_extremal_pct": 0.95,
        "end_time": 2,  # 2
        "tt_split": 0.8,
        "gbm_params": [0., 0.2],
        "rB_params": [0.2**2, 1.5, -0.7, 0.2],
        #"cond_": "forex" == "rBergomi",
        "sde_parameters": [0.2**2, 1.5, -0.7, 0.2],
        "gen_sde_dt_scale": 1e-1,
        "gen_sde_integration_type": "ito",
        "gen_sde_method": "srk",
        "learning_type": "paths",
        "time_add_type": "basic",
        "filter_by_time": True,
        "initial_point": "scale",
        "do_transforms": True,
        "transformations": OrderedDict([
            ("visibility", False),
            ("time_difference", False),
            ("time_normalisation", False),
            ("lead_lag", False),
            ("basepoint", False)
        ]),
        "transformation_args": OrderedDict([
            ("visibility", {}),
            ("time_difference", {}),
            ("time_normalisation", {}),
            ("lead_lag", {
                "time_in": True,
                "time_out": False,
                "time_normalisation": False
            }),
            ("basepoint", {})
        ]),
        "subtract_start": True,
        "preserve_order": True,
    },
    "SigKerMMDDiscriminator_config": {
        "dyadic_order"   : 1,         # Mesh size of PDE solver used in loss function
        "kernel_type"    : "rbf",     # Type of kernel to use in the discriminator
        "sigma"          : 1e0,       # Sigma in RBF kernel
        "use_phi_kernel" : False,     # Whether to use the the phi(k) = (k/2)! scaling. Set "kernel_type" to "linear".
        "n_scalings"     : 3,         # Number of samples to draw from Exp(1). ~8 tends to be a good choice.
        "max_batch"      : 16         # Maximum batch size to pass through the discriminator.
    },
    "FDDiscriminator_config": {
        #"n_dims"         : "random",
        "sigma"          : 1.0,      # Sigma in RBF kernel
    },
    "TruncatedDiscriminator_config": {
        "order"         : 6,         # Truncation level
        "scalar_term"   : False,     # Whether to include the leading 1 term in the signature.
    },
    "CDEDiscriminator_config": {
        "hidden_size" : 16,          # Number of hidden states in CDE solver
        "num_layers"  : 3,           # Number of layers in MLPs of NCDE
        "mlp_size"    : 32           # Number of neurons in each layer
    },
    "generator_config": {
        "initial_noise_size" : 5,                 # How many noise dimensions to sample at the start of the SDE.
        "noise_size"         : 8,                 # How many dimensions the Brownian motion has.
        "hidden_size"        : 16,                # Size of the hidden state of the generator SDE.
        "mlp_size"           : 64,                # Size of the layers in the various MLPs.
        "num_layers"         : 3,                 # Numer of hidden layers in the various MLPs.
        "activation"         : "LipSwish",        # Activation function to use over hidden layers
        "tanh"               : True,              # Whether to apply final tanh activation
        "tscale"             : 1,                 # Clip parameter for tanh, i.e. [-1, 1] to [-c, c]
        "fixed"              : True,              # Whether to fix the starting point or not
        "noise_type"         : "general",         # Noise type argument for torchsde
        "sde_type"           : "ito",             # SDE integration type from torchsde
        "dt_scale"           : 1e0,               # Grid shrinking parameter. Lower values are computationally more expensive
        "integration_method" : "euler"            # Integration method for torchsde
    },
    "optimizer_config": {
        "steps_per_print": 100,
        "generator_lr": 1e-03,
        "discriminator_lr": 2e-03,
        "steps": 1000, #10000,
        "init_mult1": 1,
        "init_mult2_dr": 5e-1,
        "init_mult2_df": 5e-1,
        "init_mult3": 1,
        "weight_decay": 1e-2,
        "swa_step_start": int(20000 / 2),
        "gen_optim": "Adadelta", #"Adamax", "Adadelta"
        "disc_optim": "Adadelta",
        "loss_evals": 1,
        "adapting_lr": False, #True, False
        "adapting_lr_type": "StepLR",
        "lambda_lr_const": 0.5,
        "poly_exponent_smoother": -0.5,
        "mult_const": 1.01,
        "gamma_lr": 0.5,
        "steps_lr": int(20000 // 10),
        "max_lr": 1e-06,
        "anneal_strategy": "cos",
        "total_steps": 40,
        "pct_start": 0.3,
        "div_factor": 10,
        "early_stopping": True, # True for unconditional and False for conditional
        "early_stopping_type": "marginals",
        "crit_evals": 20,
        "crit_thresh": 0.99,
        "cutoff": 1.0,
        "mmd_ci": 0.95,
        "mmd_atoms": int(20000 // 10),
        "mmd_periods": 50
    }
}
